from math import sqrt
import sys
sys.path.append('/root/autodl-tmp/project/dp_simple/')
#import ViT
from torchvision import transforms as T
from CasMVSNet_pl.models.mvsnet import CascadeMVSNet
from CasMVSNet_pl.utils import load_ckpt
from CasMVSNet_pl.datasets.dtu import DTUDataset    
from inplace_abn import ABN
from utils.utils import *
import pytorch_ssim
import pytorch_lightning as pl

sys.path.append('/root/autodl-tmp/D3Dnet/code')
import matplotlib.pyplot as plt
from dcn.modules.deform_conv import *
import functools
import torch.nn.functional as F
import torch
from collections import namedtuple
from torchvision import models
import torch.nn as nn
import sys
from einops import rearrange


class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, inputs, targets, masks):
        loss = 0
        for l in range(self.levels):
            depth_pred_l = inputs[f'depth_{l}']
            depth_gt_l = targets[f'level_{l}']
            mask_l = masks[f'level_{l}']
            loss += self.loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2**(1-l)
        return loss

loss_dict = {'sl1': SL1Loss}


class Net(pl.LightningModule):
    def __init__(self,configs):
        super(Net, self).__init__()
        self.upscale_factor = configs.upscale_factor
        self.in_channel = configs.in_channel
        out_channel = configs.out_channel
        nf = configs.nf
        in_channel = configs.in_channel
        upscale_factor = configs.upscale_factor



        self.input = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.residual_layer = self.make_layer(functools.partial(ResBlock_3d, nf), 5)
        self.TA = nn.Conv2d(3 * nf, nf, 1, 1, bias=True)
        ### reconstruct
        self.reconstruct = self.make_layer(functools.partial(ResBlock, nf), 6)
        ###upscale
        self.upscale = nn.Sequential(
            nn.Conv2d(nf, nf * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(nf, out_channel, 3, 1, 1, bias=False),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        )



        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.depthmodel = configs.model
        for param in self.depthmodel.parameters():
            param.requires_grad = False
        self.depth_loss = loss_dict['sl1'](levels=3)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        b, c, n, h, w = x.size()
        residual = rearrange(x,'b c n h w -> b (c n) h w')
        out = self.input(x)
        out = self.residual_layer(out)
        out = self.TA(out.permute(0,2,1,3,4).contiguous().view(b, -1, h, w))  # B, C, H, W
        out = self.reconstruct(out)
        ###upscale
        out = self.upscale(out)
        out = torch.add(out, residual)
        out = out.reshape(b,c,n,h,w)
        return out
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        monitor = 'val_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': monitor
            }
        }
    def decode_batch(self, batch):
        imgs = batch['imgs']
        proj_mats = batch['proj_mats']
        depths = batch['depths']
        masks = batch['masks']
        init_depth_min = batch['init_depth_min']
        depth_interval = batch['depth_interval']
        return imgs, proj_mats, depths, masks, init_depth_min, depth_interval
    def training_step(self, batch, batch_idx):
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
        imgs = imgs.transpose(1, 2)
        new_imgs = self.forward(imgs)
        results = self.depthmodel(new_imgs.transpose(1, 2), proj_mats, init_depth_min, depth_interval)
        result_original = self.depthmodel(imgs.transpose(1,2), proj_mats, init_depth_min, depth_interval)
        loss_original = self.calculate_depthloss(result_original, depths, masks)
        loss_depth = self.calculate_depthloss(results, depths, masks)



        #loss = content_loss+
        loss = loss_depth
        self.logger.experiment.add_scalar('loss', loss, self.global_step)
        #self.logger.experiment.add_scalar('content_loss', content_loss, self.global_step)
        self.logger.experiment.add_scalar('depth_loss', loss_depth, self.global_step)
        self.logger.experiment.add_scalar('ration: refined/original', loss_depth/(1e-10 + loss_original), self.global_step)
        self.log('loss', loss)

        return loss
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
       
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
        imgs = imgs.transpose(1, 2)
        new_imgs = self.forward(imgs)
        results = self.depthmodel(new_imgs.transpose(1, 2), proj_mats, init_depth_min, depth_interval)
        result_original = self.depthmodel(imgs.transpose(1,2), proj_mats, init_depth_min, depth_interval)
        loss_original = self.calculate_depthloss(result_original, depths, masks)
        loss_depth = self.calculate_depthloss(results, depths, masks)

        new_imgs = new_imgs.transpose(1, 2) # b, n, c, h, w
        if batch_idx%100 == 0:

            denormalize = T.Compose([T.Normalize(mean=[0., 0., 0.],
                                                std=[1/0.229, 1/0.224, 1/0.225]),
                                    T.Normalize(mean=[-0.485, -0.456, -0.406],
                                                std=[1., 1., 1.]),
                                    ])
            new_imgs = denormalize(new_imgs[0])
            
            #self.logger.experiment.add_image('val_output', new_imgs[0,0], self.current_epoch)
            new_imgs = rearrange(new_imgs, 'n c h w -> c h (n w) ')
            self.logger.experiment.add_image('val_output', new_imgs, self.global_step)

        #loss = content_loss+
        loss = loss_depth
        epochs = self.current_epoch
       # self.logger.experiment.add_scalar('val_loss', loss, batch_idx*epochs)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        #self.logger.experiment.add_scalar('content_loss', content_loss, self.global_step)
        #self.logger.experiment.add_scalar('val_depth_loss', loss_depth, batch_idx*epochs)
        self.log('val_depth_loss', loss_depth, on_step=True, on_epoch=True)

        #self.logger.experiment.add_scalar('val_ration: refined/original', loss_depth/(1e-10 + loss_original), self.g)
        self.log('val_ratio: refined/original', loss_depth/(1e-10 + loss_original), on_step=True, on_epoch=True)
        #self.log('val_loss', loss)
        return loss
    def calculate_depthloss(self, results, depths, masks):
        depth_loss = self.depth_loss(results, depths, masks)
        return depth_loss*0.01
    


class ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d, self).__init__()
        self.dcn0 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.dcn1 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x

class ResBlock(nn.Module):
    def __init__(self, nf):
        super(ResBlock, self).__init__()
        self.dcn0 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.dcn1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x
    


if __name__ == "__main__":
    
    from pytorch_lightning.loggers import TensorBoardLogger
   
    logger = TensorBoardLogger('/root/autodl-tmp/project/dp_simple/logs', name='d3c_net')
    model = CascadeMVSNet(n_depths=[8,32,48],
                      interval_ratios=[1.0,2.0,4.0],
                      num_groups=1,
                     
                      norm_act=ABN).cuda()
    load_ckpt(model, '/root/autodl-tmp/project/dp_simple/CasMVSNet_pl/ckpts/_ckpt_epoch_10.ckpt')
    class configs:
        lambda_content = 1
        lambda_style = 1
        model = model
        lambda_ssim = 1 
        logger = logger
        upscale_factor = 1
        in_channel = 3
        out_channel = 9
        nf = 64
    model = Net(configs)
    
    train_dataset = DTUDataset('/root/autodl-tmp/mvs_training/dtu/', 'train', img_wh=(256,256))
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    
    val_dataset = DTUDataset('/root/autodl-tmp/mvs_training/dtu/', 'val',img_wh=(256,256))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
    #train(train_loader,model,10)
    # save model

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/root/autodl-tmp/project/dp_simple/ckpts/',
        filename='d3c_net_{epoch}',
        save_top_k=1,
        mode='min',

    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=False,
        mode='min'
    )
    trainer = Trainer(max_epochs=200, gpus=2,
                    strategy='ddp',
                    callbacks=[checkpoint_callback, 
                               #early_stop_callback
                               ]
                    ,
                    
                    val_check_interval=1.0,
                    logger=logger,
                    resume_from_checkpoint='/root/autodl-tmp/project/dp_simple/ckpts/d3c_net_epoch=8.ckpt'
                    )
    trainer.fit(model, train_loader, val_loader)
    