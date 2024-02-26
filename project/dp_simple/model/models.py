import torch
from collections import namedtuple
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import sys
from einops import rearrange
sys.path.append('/root/autodl-tmp/project/dp_simple/')
from modules.conv import resConv21, C3D, R3D, Conv21,upsample,downsample
from CasMVSNet_pl.models.mvsnet import CascadeMVSNet
from CasMVSNet_pl.utils import load_ckpt
from CasMVSNet_pl.datasets.dtu import DTUDataset    
from inplace_abn import ABN
from utils.utils import *
import pytorch_ssim
import pytorch_lightning as pl



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



class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class TransformerNet(pl.LightningModule):
    def __init__(self,configs):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            Conv21(input_channels=3,output_channels=32,dropout=0.5,
                    kernel_size=9,depth=3,padding=4,
                    stride=1),
            Conv21(input_channels=32,output_channels=64,dropout=0.5,
                    kernel_size=3,depth=3,padding=1,
                    stride=2),
            Conv21(input_channels=64,output_channels=128,dropout=0.5,
                    kernel_size=3,depth=3,padding=1,
                    stride=2),
            resConv21(input_channels=128,output_channels=128,dropout=0.5,
                     kernel_size=3,depth=3,padding=1,
                     stride=1),
            resConv21(input_channels=128,output_channels=128,dropout=0.5,
                        kernel_size=3,depth=3,padding=1,
                        stride=1), 
            resConv21(input_channels=128,output_channels=128,dropout=0.5,
                        kernel_size=3,depth=3,padding=1,
                        stride=1),
            resConv21(input_channels=128,output_channels=128,dropout=0.5,
                        kernel_size=3,depth=3,padding=1,
                        stride=1),
            resConv21(input_channels=128,output_channels=128,dropout=0.5,
                        kernel_size=3,depth=3,padding=1,
                        stride=1),
            Conv21(input_channels=128,output_channels=64,dropout=0.5,
                    kernel_size=3,depth=3,padding=1,
                    stride=1),
            upsample(scale_factor=2,mode='bilinear'),
            Conv21(input_channels=64,output_channels=32,dropout=0.5,
                    kernel_size=3,depth=3,padding=1,
                    stride=1),
            upsample(scale_factor=2,mode='bilinear'),
            Conv21(input_channels=32,output_channels=3,dropout=0.5,
                    kernel_size=9,depth=3,padding=4,
                    stride=1,activation='linear'),
        )
        self.l2_loss = torch.nn.MSELoss()
        self.lambd_content = configs.lambda_content
        self.vgg = VGG16(requires_grad=False)
        self.depth_loss = loss_dict['sl1'](levels=3)
        self.depthmodel = configs.model
        for param in self.depthmodel.parameters():
            param.requires_grad = False
        self.training_parms = [(list(self.model.parameters()),1.0)]
       
    def configure_optimizers(self):
        for params,lr in self.training_parms:
            optimizer = torch.optim.Adam(params, lr=lr*0.0001)
        # define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=100,eta_min=1e-5,verbose=True)
        
        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'loss',
        }
        
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    def forward(self, x):
        return self.model(x)
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
        img_views = imgs.shape[1]
        imgs_new = self.forward(imgs.transpose(1,2))

        #content_loss = self.calculate_contentloss(imgs_new, imgs.transpose(1,2))
        imgs_new = imgs_new.transpose(1,2)


        results = self.depthmodel(imgs_new, proj_mats, init_depth_min, depth_interval)
        result_original = self.depthmodel(imgs, proj_mats, init_depth_min, depth_interval)
        loss_original = self.calculate_depthloss(result_original, depths, masks)
        loss_depth = self.calculate_depthloss(results, depths, masks)

        #loss = content_loss+
        loss = loss_depth
        self.logger.experiment.add_scalar('loss', loss, self.global_step)
        #self.logger.experiment.add_scalar('content_loss', content_loss, self.global_step)
        self.logger.experiment.add_scalar('depth_loss', loss_depth, self.global_step)
        self.logger.experiment.add_scalar('loss_original', loss_depth/(1e-10 + loss_original), self.global_step)
        self.log('loss', loss)
        return loss
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
        img_views = imgs.shape[1]
        imgs_new = self.forward(imgs.transpose(1,2))

        #content_loss = self.calculate_contentloss(imgs_new, imgs.transpose(1,2))
        imgs_new = imgs_new.transpose(1,2)


        results = self.depthmodel(imgs_new, proj_mats, init_depth_min, depth_interval)
        result_original = self.depthmodel(imgs, proj_mats, init_depth_min, depth_interval)
        loss_original = self.calculate_depthloss(result_original, depths, masks)
        loss_depth = self.calculate_depthloss(results, depths, masks)

        #loss = content_loss
        loss = loss_depth
        self.log('val_loss', loss,on_epoch=True)
        self.logger.experiment.add_scalar('val_loss', loss, self.global_step)
        #self.logger.experiment.add_scalar('val_content_loss', content_loss, self.global_step)
        self.logger.experiment.add_scalar('val_depth_loss', loss_depth, self.global_step)
        self.logger.experiment.add_scalar('val_loss_original', loss_depth/(1e-10 + loss_original), self.global_step)
        return loss
    def calculate_contentloss(self, x_transform, x_original):
        b,c,d,h,w = x_transform.shape
        x_transform = rearrange(x_transform,'b c d h w -> (b d) c h w')
        x_original = rearrange(x_original,'b c d h w -> (b d) c h w')

        # ssim_loss = pytorch_ssim.SSIM(window_size=11)
        # ssim_loss = 1 - ssim_loss(x_transform,x_original)

        
        features_original = self.vgg(x_original)
        features_transformed = self.vgg(x_transform)
        content_loss = self.l2_loss(features_transformed.relu2_2, features_original.relu2_2)
        return content_loss
    def calculate_depthloss(self, results, depths, masks):
        depth_loss = self.depth_loss(results, depths, masks)
        return depth_loss*0.01


if __name__ == "__main__":
    # import tensorboard as logger
    
  
    from pytorch_lightning.loggers import TensorBoardLogger
   
    logger = TensorBoardLogger('/root/autodl-tmp/project/dp_simple/logs', name='transformer_net')
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

    model = TransformerNet(configs)
    
    train_dataset = DTUDataset('/root/autodl-tmp/mvs_training/dtu/', 'train')
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    def train(train_loader,model,epochs):
        optimizer = torch.optim.Adam(model.parameters(), 1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-5,verbose=True)
        from tqdm import tqdm
        dataloader = tqdm(train_loader)
        count=0
        for epoch in range(epochs):
            
            for batch in dataloader:
                for key in batch:
                    try:
                        if type(batch[key]) == type(dict()):
                            for k in batch[key]:
                                batch[key][k] = batch[key][k].cuda()
                        else:
                            batch[key] = batch[key].cuda()
                        
                    except:
                       
                        batch[key] = batch[key]
                    
                loss = model.training_step(batch, count)
                count+=1
                optimizer.zero_grad()
                loss.backward()
                # print the loss on progress bar
                dataloader.set_postfix({'loss':loss})
                optimizer.step()

            scheduler.step()
            torch.save(model.state_dict(), 
                       f'/root/autodl-tmp/project/dp_simple/ckpts/transformer_net_{epoch}.pth')
    val_dataset = DTUDataset('/root/autodl-tmp/mvs_training/dtu/', 'val')
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
    #train(train_loader,model,10)
    # save model

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/root/autodl-tmp/project/dp_simple/ckpts/',
        filename='transformer_net_{epoch}',
        save_top_k=1,
        mode='min',

    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=False,
        mode='min'
    )
    trainer = Trainer(max_epochs=200, gpus=2,
                      strategy='ddp',
                    callbacks=[checkpoint_callback, early_stop_callback],
                    val_check_interval=1.0,
                    logger=logger,
                    resume_from_checkpoint='/root/autodl-tmp/project/dp_simple/ckpts/transformer_net_epoch=36.ckpt'
                    )
    trainer.fit(model, train_loader, val_loader)
    
    
