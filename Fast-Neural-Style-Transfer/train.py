import argparse
import os
import sys
import random
from PIL import Image
import numpy as np
import torch
import glob
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from models import TransformerNet, VGG16
from utils import *
from deptLoss import SL1Loss
import sys
from einops import rearrange
from tqdm import tqdm
sys.path.append("/root/autodl-tmp/Fast-Neural-Style-Transfer/")
sys.path.append("/root/autodl-tmp/taming-transformers/")


from taming.data.dtu import DTUDataset  
def denormalize(tensors):
    unpreprocess =  T.Compose([
            T.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        ])
    tensors = unpreprocess(tensors)
    return tensors
def decode_batch(batch):
    imgs = batch['imgs']
    proj_mats = batch['proj_mats']
    depths = batch['depths']
    masks = batch['masks']
    init_depth_min = batch['init_depth_min']
    depth_interval = batch['depth_interval']
    
    return imgs, proj_mats, depths, masks, init_depth_min, depth_interval
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for Fast-Neural-Style")
    parser.add_argument("--dataset_path", type=str, required=True,default = "/root/autodl-tmp/mvs_training", help="path to training dataset")
    parser.add_argument("--style_image", type=str, default="style-images/train.jpg", help="path to style image")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=256, help="Size of training images")
    parser.add_argument("--style_size", type=int, help="Size of style image")
    parser.add_argument("--lambda_content", type=float, default=1e5, help="Weight for content loss")
    parser.add_argument("--lambda_style", type=float, default=1e10, help="Weight for style loss")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_model", type=str, help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=2000, help="Batches between saving model")
    parser.add_argument("--sample_interval", type=int, default=100, help="Batches between saving image samples")
    args = parser.parse_args(["--dataset_path","/root/autodl-tmp/mvs_training/dtu",
                             # "--checkpoint_model","/root/autodl-tmp/checkpoints/mse_44000.pth"
                              ])

    style_name = args.style_image.split("/")[-1].split(".")[0]
    os.makedirs(f"images/outputs/{style_name}-training", exist_ok=True)
    os.makedirs(f"checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataloader for the training data
    train_dataset = DTUDataset(root_dir = args.dataset_path,split="train")
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    val_dataset = DTUDataset(root_dir = args.dataset_path,split="val")
    valloader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Defines networks
    transformer = TransformerNet().to(device)
    #vgg = VGG16(requires_grad=False).to(device)

    # Load checkpoint model if specified
    if False:
        transformer.load_state_dict(torch.load(args.checkpoint_model))
        print(f"Loaded checkpoint model from {args.checkpoint_model}")

    # Define optimizer and loss
    optimizer = Adam(transformer.parameters(), args.lr)
    l2_loss = torch.nn.MSELoss().to(device)
    cal_depthloss = SL1Loss().to(device)

    # Load style image
    # style = style_transform(args.style_size)(Image.open(args.style_image))
    # style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # # Extract style features
    # features_style = vgg(style)
    # gram_style = [gram_matrix(y) for y in features_style]

    # Sample 8 images for visual evaluation of the model
    # image_samples = []
    # for path in random.sample(glob.glob(f"{args.dataset_path}/*/*.png"), 8):
    #     image_samples += [style_transform(args.image_size)(Image.open(path))]
    # image_samples = torch.stack(image_samples)

    def save_sample(batches_done,image_samples):
        """ Evaluates the model and saves image samples """
        transformer.eval()
        with torch.no_grad():
            output = transformer(image_samples.to(device))
        image_grid = denormalize(torch.cat((image_samples.cpu(), output.cpu()), 2))
        if not os.path.exists(f"/root/autodl-tmp/images/outputs/{style_name}-training"):
            os.makedirs(f"/root/autodl-tmp/images/outputs/{style_name}-training")
        
        save_image(image_grid, f"/root/autodl-tmp/images/outputs/{style_name}-training/{batches_done}.jpg", nrow=4)
        transformer.train()

    for epoch in range(args.epochs):
        epoch_metrics = {"content": [], "mse": [], "total": [],"depth":[],
        'abs/abs_original':[],
        "acc_1mm/acc_1mm_original":[],
        "acc_2mm/acc_2mm_original":[],
        "acc_3mm/acc_3mm_original":[],
        "acc_4mm/acc_4mm_original":[],
        }
        tqdm_bar = tqdm(dataloader)
        
        for batch_i, batch in enumerate(tqdm_bar):
            target_imgs = batch['target_imgs']

            images, proj_mats, depths, masks, init_depth_min, depth_interval = decode_batch(batch)

            target_imgs = rearrange(target_imgs, 'b n c h w -> b c h (n w)', n=3)
            images = rearrange(images, 'b n c h w -> b c h (n w)', n=3)

            optimizer.zero_grad()

            images_original = images.to(device)
            images_transformed = transformer(images_original)

            # Extract features
            # features_original = vgg(images_original)
            # features_transformed = vgg(images_transformed)

            # # Compute content loss as MSE between features
            # with torch.no_grad():
            #   content_loss =  l2_loss(features_transformed.relu2_2, features_original.relu2_2)

            # Compute style loss as MSE between gram matrices
            style_loss = 0
            style_loss = l2_loss(images_original, images_transformed)*10
            # for ft_y, gm_s in zip(features_transformed, gram_style):
            #     gm_y = gram_matrix(ft_y)
            #     style_loss += l2_loss(gm_y, gm_s[: images.size(0), :, :])
            
            depth_loss,depth_ori,log = cal_depthloss(images_transformed, images_original, proj_mats, depths, masks, init_depth_min, depth_interval)
            depthloss = (depth_loss-depth_ori)*10


            total_loss = depthloss+ style_loss
            total_loss.backward()
            optimizer.step()

            #epoch_metrics["mse"] += [content_loss.item()]
            epoch_metrics["depth"] += [depth_loss.item()/depth_ori.item()]
            epoch_metrics["total"] += [total_loss.item()]
            epoch_metrics["mse"] += [style_loss.item()]
            epoch_metrics['abs/abs_original'] += [log['abs/abs_original'].item()]
            epoch_metrics["acc_1mm/acc_1mm_original"] += [log['acc_1mm/acc_1mm_original'].item()]
            epoch_metrics["acc_2mm/acc_2mm_original"] += [log['acc_2mm/acc_2mm_original'].item()]
            epoch_metrics["acc_3mm/acc_3mm_original"] += [log['acc_3mm/acc_3mm_original'].item()]
            epoch_metrics["acc_4mm/acc_4mm_original"] += [log['acc_4mm/acc_4mm_original'].item()]




            # sys.stdout.write(
            #     "\r[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]"
            #     % (
            #         epoch + 1,
            #         args.epochs,
            #         batch_i,
            #         len(train_dataset),
            #         content_loss.item(),
            #         np.mean(epoch_metrics["content"]),
            #         depth_loss.item(),
            #         np.mean(epoch_metrics["depth"]),
            #         total_loss.item(),
            #         np.mean(epoch_metrics["total"]),
            #     )
            # )
            tqdm_bar.set_postfix_str(
                f"[Epoch {epoch + 1}/{args.epochs}] [Batch {batch_i}/{len(train_dataset)}] "
                #f"[Content: {np.mean(epoch_metrics['content']):.2f}] "
                f"[Depth: {np.mean(epoch_metrics['depth']):.2f}] "
                f"[MSE: {np.mean(epoch_metrics['mse']):.2f}] "
                f"[Total: {np.mean(epoch_metrics['total']):.2f}]"
        
                f"[abs/abs_original: {np.mean(epoch_metrics['abs/abs_original']):.2f}]"
                f"[acc_1mm/acc_1mm_original: {np.mean(epoch_metrics['acc_1mm/acc_1mm_original']):.2f}]"
                f"[acc_2mm/acc_2mm_original: {np.mean(epoch_metrics['acc_2mm/acc_2mm_original']):.2f}]"
                f"[acc_3mm/acc_3mm_original: {np.mean(epoch_metrics['acc_3mm/acc_3mm_original']):.2f}]"
                f"[acc_4mm/acc_4mm_original: {np.mean(epoch_metrics['acc_4mm/acc_4mm_original']):.2f}]"


            )
            batches_done = epoch * len(dataloader) + batch_i + 1
            if batches_done % args.sample_interval == 0:

                save_sample(batches_done,images)

            if args.checkpoint_interval > 0 and batches_done % args.checkpoint_interval == 0:
                style_name = os.path.basename(args.style_image).split(".")[0]
                torch.save(transformer.state_dict(), f"checkpoints/{style_name}_{batches_done}.pth")
        epoch_metrics = {"content": [], "mse": [], "total": [],"depth_ratio":[],
        'abs/abs_original':[],
        "acc_1mm/acc_1mm_original":[],
        "acc_2mm/acc_2mm_original":[],
        "acc_3mm/acc_3mm_original":[],
        "acc_4mm/acc_4mm_original":[],
        }
        tqdm_bar = tqdm(valloader)
        
        with torch.no_grad():
            for batch_i, batch in enumerate(tqdm_bar):
                target_imgs = batch['target_imgs']

                images, proj_mats, depths, masks, init_depth_min, depth_interval = decode_batch(batch)

                target_imgs = rearrange(target_imgs, 'b n c h w -> b c h (n w)', n=3)
                images = rearrange(images, 'b n c h w -> b c h (n w)', n=3)

                optimizer.zero_grad()

                images_original = images.to(device)
                images_transformed = transformer(images_original)

                # Extract features
                # features_original = vgg(images_original)
                # features_transformed = vgg(images_transformed)

                # # Compute content loss as MSE between features
                # with torch.no_grad():
                #   content_loss =  l2_loss(features_transformed.relu2_2, features_original.relu2_2)

                # Compute style loss as MSE between gram matrices
                style_loss = 0
                style_loss = l2_loss(images_original, images_transformed)*10
                # for ft_y, gm_s in zip(features_transformed, gram_style):
                #     gm_y = gram_matrix(ft_y)
                #     style_loss += l2_loss(gm_y, gm_s[: images.size(0), :, :])
                
                depth_loss,depth_ori,log = cal_depthloss(images_transformed, images_original, proj_mats, depths, masks, init_depth_min, depth_interval)
                depthloss = (depth_loss-depth_ori)*10


                total_loss = depthloss + style_loss
                

                #epoch_metrics["mse"] += [content_loss.item()]
                epoch_metrics["depth_ratio"] += [depth_loss.item()/depth_ori.item()]
                epoch_metrics["total"] += [total_loss.item()]
                epoch_metrics["mse"] += [style_loss.item()]
                epoch_metrics['abs/abs_original'] += [log['abs/abs_original'].item()]
                epoch_metrics["acc_1mm/acc_1mm_original"] += [log['acc_1mm/acc_1mm_original'].item()]
                epoch_metrics["acc_2mm/acc_2mm_original"] += [log['acc_2mm/acc_2mm_original'].item()]
                epoch_metrics["acc_3mm/acc_3mm_original"] += [log['acc_3mm/acc_3mm_original'].item()]
                epoch_metrics["acc_4mm/acc_4mm_original"] += [log['acc_4mm/acc_4mm_original'].item()]




                # sys.stdout.write(
                #     "\r[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]"
                #     % (
                #         epoch + 1,
                #         args.epochs,
                #         batch_i,
                #         len(train_dataset),
                #         content_loss.item(),
                #         np.mean(epoch_metrics["content"]),
                #         depth_loss.item(),
                #         np.mean(epoch_metrics["depth"]),
                #         total_loss.item(),
                #         np.mean(epoch_metrics["total"]),
                #     )
                # )
                tqdm_bar.set_postfix_str(
                    f"[VAL Epoch {epoch + 1}/{args.epochs}] [Batch {batch_i}/{len(val_dataset)}] "
                    #f"[Content: {np.mean(epoch_metrics['content']):.2f}] "
                    f"[Depth: {np.mean(epoch_metrics['depth_ratio']):.2f}] "
                    f"[MSE: {np.mean(epoch_metrics['mse']):.2f}] "
                    f"[Total: {np.mean(epoch_metrics['total']):.2f}]"
            
                    f"[abs/abs_original: {np.mean(epoch_metrics['abs/abs_original']):.2f}]"
                    f"[acc_1mm/acc_1mm_original: {np.mean(epoch_metrics['acc_1mm/acc_1mm_original']):.2f}]"
                    f"[acc_2mm/acc_2mm_original: {np.mean(epoch_metrics['acc_2mm/acc_2mm_original']):.2f}]"
                    f"[acc_3mm/acc_3mm_original: {np.mean(epoch_metrics['acc_3mm/acc_3mm_original']):.2f}]"
                    f"[acc_4mm/acc_4mm_original: {np.mean(epoch_metrics['acc_4mm/acc_4mm_original']):.2f}]"


                )
                batches_done = epoch * len(dataloader) + batch_i + 1
                if batches_done % args.sample_interval == 0:

                    save_sample(batches_done,images)
            
            
