import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
from deptLoss import SL1Loss
from einops import rearrange
from tensorboardX import SummaryWriter


def decode_batch(batch):
    imgs = batch['imgs']
    proj_mats = batch['proj_mats']
    depths = batch['depths']
    masks = batch['masks']
    init_depth_min = batch['init_depth_min']
    depth_interval = batch['depth_interval']
    
    return imgs, proj_mats, depths, masks, init_depth_min, depth_interval

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()

        
        # self.depth_loss = SL1Loss()
        # self.depth_loss = self.depth_loss.to(device=args.device)
        logger = SummaryWriter("/root/autodl-tmp/VQGAN-pytorch/logs")
        self.logger = logger
        

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
    
    def train(self, args):
        from torchvision import transforms as T
        unpreprocess =  T.Compose([
            T.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        ])
        train_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, batch in zip(pbar, train_dataset):
                    imgs, proj_mats, depths, masks, init_depth_min, depth_interval = decode_batch(batch)
                    proj_mats = proj_mats.to(device=args.device)
                    init_depth_min = init_depth_min.to(device=args.device)
                    depth_interval = depth_interval.to(device=args.device)
                    for key in depths:
                        depths[key] = depths[key].to(device=args.device)
                    for key in masks:
                        masks[key] = masks[key].to(device=args.device)


                    imgs = rearrange(imgs,'b n c h w -> b c h (n w)')



                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    #depth_loss,loss_original = self.depth_loss(decoded_images, imgs, proj_mats, depths, masks, init_depth_min, depth_interval)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + \
                                            args.rec_loss_factor * rec_loss
                  
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ,_ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss,depth_loss=None)
                    vq_loss = perceptual_rec_loss + q_loss +\
                          disc_factor * λ * g_loss #+ miu * depth_loss * args.rec_loss_factor

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % 10 == 0:
                        with torch.no_grad():
                            imgs = unpreprocess(imgs)
                            decoded_images = unpreprocess(decoded_images)
                            
                            real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join("/root/autodl-tmp/VQGAN-pytorch/results", f"{epoch}_{i}.jpg"), nrow=4)
                            self.logger.add_scalar('VQ_Loss', vq_loss, epoch*steps_per_epoch+i)
                            self.logger.add_scalar('GAN_Loss', gan_loss, epoch*steps_per_epoch+i)
                            self.logger.add_scalar('Perceptual_Loss', perceptual_loss, epoch*steps_per_epoch+i)
                            # self.logger.add_scalar('Depth_Loss', depth_loss, epoch*steps_per_epoch+i)
                            # self.logger.add_scalar('Loss_Original', loss_original, epoch*steps_per_epoch+i)
                            # # ration depth_loss and loss_original
                            # self.logger.add_scalar('Ratio', depth_loss/loss_original, epoch*steps_per_epoch+i)
                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=(512,640), help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--split', type=str, default='train', help='Which split to use (default: train)')
    args = parser.parse_args()
    args.dataset_path = "/root/autodl-tmp/mvs_training/dtu"

    train_vqgan = TrainVQGAN(args)
    train_vqgan.train(args)



