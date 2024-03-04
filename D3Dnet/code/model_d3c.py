from math import sqrt
import sys
import copy
sys.path.append('/root/autodl-tmp/project/dp_simple/')
sys.path.append("/root/autodl-tmp/taming-transformers")
from taming.data.dtu import DTUDataset
#import ViT
from torchvision import transforms as T
from CasMVSNet_pl.models.mvsnet import CascadeMVSNet
from CasMVSNet_pl.utils import load_ckpt

from CasMVSNet_pl.utils import *
from data_utils import *
from taming.data.dtu import DTUDataset 
from CasMVSNet_pl.metrics import *  
from inplace_abn import ABN
from torchvision import utils as vutils
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
from torchvision import models
def abs_error(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    return (depth_pred - depth_gt).abs()

def acc_threshold(depth_pred, depth_gt, mask, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    errors = abs_error(depth_pred, depth_gt, mask)
    acc_mask = errors < threshold
    return acc_mask.float()

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
        self.l2_loss = nn.MSELoss()


        self.input = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.vgg = VGG16()
        self.residual_layer = self.make_layer(functools.partial(ResBlock_3d, nf,dropout = 0.1), configs.num_groups1)
        self.TA = nn.Conv2d(3 * nf, nf, 1, 1, bias=True)
        ### reconstruct
        self.reconstruct = self.make_layer(functools.partial(ResBlock, nf,dropout = 0.1), configs.num_groups2)
        ###upscale
        self.upscale = nn.Sequential(
            nn.Conv2d(nf, nf * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(nf, out_channel, 3, 1, 1, bias=False),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        )
        self.lambda_content = configs.lambda_content



        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.depthmodel = configs.model
        for param in self.depthmodel.parameters():
            param.requires_grad = False
        self.depth_loss = loss_dict['sl1'](levels=3)
        self.lr = configs.lr
        self.unpreprocess = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                        std=[1/0.229, 1/0.224, 1/0.225])
        self.l2_loss = nn.MSELoss()

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
    
    def predict_step(self,imgs):
        imgs = imgs.transpose(1, 2)
        new_imgs = self.forward(imgs)
        return new_imgs.transpose(1, 2)
    def training_step(self, batch, batch_idx):
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
        target_imgs = batch['target_imgs']
        
        # target_imgs,imgs,masks,depths = random_crop(target_imgs,imgs, 256, 1,masks=copy.deepcopy(masks)
        #                                             ,depths=copy.deepcopy(depths)
        #                                             )
        imgs = rearrange(imgs, 'b n c h w -> b c h (n w)')
        target_imgs = rearrange(target_imgs, 'b n c h w -> b c h (n w)')
        imgs = F.interpolate(imgs, scale_factor=0.5, mode='bilinear', align_corners=False)
        target_imgs = F.interpolate(target_imgs, scale_factor=0.5, mode='bilinear', align_corners=False)
        imgs = rearrange(imgs, 'b c h (n w) -> b n c h w', n=3)
        target_imgs = rearrange(target_imgs, 'b c h (n w) -> b n c h w', n=3)

        imgs = imgs.transpose(1, 2)
        new_imgs = self.forward(imgs)
        

        
        target_imgs = target_imgs.transpose(1, 2)
        loss = self.l2_loss(new_imgs, target_imgs)


        # imgs = imgs.transpose(1, 2)
        # new_imgs = new_imgs.transpose(1, 2)
        # results = self.depthmodel(new_imgs, proj_mats, init_depth_min, depth_interval)
        # result_original = self.depthmodel(imgs, proj_mats, init_depth_min, depth_interval)
        # loss_original = self.calculate_depthloss(result_original, depths, masks)
        # loss_depth = self.calculate_depthloss(results, depths, masks)
        # depth_loss = loss_depth-loss_original
        # self.log('train/depth_loss', loss_depth, on_step=True, on_epoch=True)
        # self.log('train: refined/original', loss_depth/(1e-10 + loss_original), on_step=True, on_epoch=True)
        # loss = l2_loss + depth_loss
        





        self.log("train/loss", loss, on_step=True, on_epoch=True)
        with torch.no_grad():
            log ={}
            if batch_idx%20 == 0:
                imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
                target_imgs = batch['target_imgs']
                imgs = imgs.transpose(1, 2)
                new_imgs = self.forward(imgs)
                target_imgs = target_imgs.transpose(1, 2)


                results = self.depthmodel(new_imgs.transpose(1, 2), proj_mats, init_depth_min, depth_interval)
                result_original = self.depthmodel(imgs.transpose(1,2), proj_mats, init_depth_min, depth_interval)
                loss_original = self.calculate_depthloss(result_original, depths, masks)
                loss_depth = self.calculate_depthloss(results, depths, masks)
                content_loss = self.calculate_contentloss(new_imgs, target_imgs)*self.lambda_content
            
                self.log('train/content_loss', content_loss, on_step=True, on_epoch=True)
                self.log('train/depth_loss', loss_depth, on_step=True, on_epoch=True)
                self.log('train: refined/original', loss_depth/(1e-10 + loss_original), on_step=True, on_epoch=True)
        
        
       
            
                try:
                    imgs_new = new_imgs.transpose(1, 2)
                    img_ = self.unpreprocess(imgs_new[0]).cpu() # batch 0, ref image
                    img_ = rearrange(img_, 'n c h w -> c h (n w)')
                    depth_gt_ = visualize_depth(depths['level_0'][0])
                    depth_pred_ = visualize_depth(results['depth_0'][0]*masks['level_0'][0])
                    prob = visualize_prob(results['confidence_0'][0]*masks['level_0'][0])
                    stack1 = torch.stack([img_, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                    # vutils.save_image(stack, 
                    #                   f'/root/autodl-tmp/images/outputs/d3c/train/d3c_pred_net_{self.current_epoch}_{batch_idx}.png')
                    
                    imgs = imgs.transpose(1, 2)
                    img_ = self.unpreprocess(imgs[0]).cpu() # batch 0, ref image
                    img_ = rearrange(img_, 'n c h w -> c h (n w)')
                    depth_gt_ = visualize_depth(depths['level_0'][0])
                    depth_pred_ = visualize_depth(result_original['depth_0'][0]*masks['level_0'][0])
                    prob = visualize_prob(result_original['confidence_0'][0]*masks['level_0'][0])
                    stack2 = torch.stack([img_, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                    stack = torch.cat([stack1, stack2], dim=0)
                    vutils.save_image(stack, f'/root/autodl-tmp/images/outputs/d3c/train/d3c_ori_net_{self.current_epoch}_{batch_idx}.png'
                                      ,nrow = 4)
                    log['error'] =0
                    
                    

                except:
                    log['error'] = 1

                depth_pred = results['depth_0']
                depth_old = result_original['depth_0']
                depth_gt = depths['level_0']
                mask = masks['level_0']
                log['train/abs_err'] = abs_err = abs_error(depth_pred, depth_gt, mask).mean()
                log['train/acc_1mm'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()
                log['train/acc_2mm'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()
                log['train/acc_4mm'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()
                log['train/abs_err_old'] = abs_error(depth_old, depth_gt, mask).mean()
                log['train/acc_1mm_old'] = acc_threshold(depth_old, depth_gt, mask, 1).mean()
                log['train/acc_2mm_old'] = acc_threshold(depth_old, depth_gt, mask, 2).mean()
                log['train/acc_4mm_old'] = acc_threshold(depth_old, depth_gt, mask, 4).mean()
                # the ratio of the loss
                log['train/abs_err_ratio'] = abs_err/(1e-10 + abs_error(depth_old, depth_gt, mask).mean())
                log['train/acc_1mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 1).mean())
                log['train/acc_2mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 2).mean())
                log['train/acc_4mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 4).mean())
                self.log_dict(log, on_epoch=True, on_step=True)
        return loss
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
       
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)

        
        imgs = imgs.transpose(1, 2)
        target_imgs = batch['target_imgs']
        target_imgs = target_imgs.transpose(1, 2)
        new_imgs = self.forward(imgs)
        loss = self.l2_loss(new_imgs, target_imgs)

        
        with torch.no_grad():
            results = self.depthmodel(new_imgs.transpose(1, 2), proj_mats, init_depth_min, depth_interval)
            result_original = self.depthmodel(imgs.transpose(1,2), proj_mats, init_depth_min, depth_interval)
            loss_original = self.calculate_depthloss(result_original, depths, masks)
            loss_depth = self.calculate_depthloss(results, depths, masks)

        new_imgs = new_imgs.transpose(1, 2) # b, n, c, h, w
        imgs = imgs.transpose(1, 2)
        if batch_idx%100 == 0:

            denormalize = T.Compose([T.Normalize(mean=[0., 0., 0.],
                                                std=[1/0.229, 1/0.224, 1/0.225]),
                                    T.Normalize(mean=[-0.485, -0.456, -0.406],
                                                std=[1., 1., 1.]),
                                    ])
            new_img = denormalize(new_imgs[0])
            
            target_imgs = denormalize(target_imgs[0])

            target_imgs = rearrange(target_imgs, 'n c h w -> c h (n w)')
            new_img= rearrange(new_img, 'n c h w -> c h (n w) ')
            cat_imgs = torch.stack([target_imgs, new_img])
            vutils.save_image(cat_imgs, f'/root/autodl-tmp/images/outputs/d3c/val/d3c_net_{self.current_epoch}_{batch_idx}.png',
                              nrow = 2)

        
        
        epochs = self.current_epoch
       
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_depth_loss', loss_depth, on_step=True, on_epoch=True)
        self.log('val_ratio: refined/original', loss_depth/(1e-10 + loss_original), on_step=True, on_epoch=True)
        
        log= {}
        with torch.no_grad():
            if batch_idx%10 == 0:
                try:
                    img_ = self.unpreprocess(new_imgs[0,0]).cpu() # batch 0, ref image
                    depth_gt_ = visualize_depth(depths['level_0'][0])
                    depth_pred_ = visualize_depth(results['depth_0'][0]*masks['level_0'][0])
                    prob = visualize_prob(results['confidence_0'][0]*masks['level_0'][0])
                    stack = torch.stack([img_, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                    self.logger.experiment.add_images('val/image_pred_prob',
                                                    stack, self.global_step)
                    
                
                    img_ = self.unpreprocess(imgs[0,0]).cpu() # batch 0, ref image
                    depth_gt_ = visualize_depth(depths['level_0'][0])
                    depth_pred_ = visualize_depth(result_original['depth_0'][0]*masks['level_0'][0])
                    prob = visualize_prob(result_original['confidence_0'][0]*masks['level_0'][0])
                    stack = torch.stack([img_, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                    self.logger.experiment.add_images('val/image_GT_prob_old',
                                                    stack, self.global_step)
                    log['error'] =0
                except Exception as e:
                    print(e)
                    log['error'] = 1

            depth_pred = results['depth_0']
            depth_old = result_original['depth_0']
            depth_gt = depths['level_0']
            mask = masks['level_0']
            log['val/abs_err'] = abs_err = abs_error(depth_pred, depth_gt, mask).mean()
            log['val/acc_1mm'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()
            log['val/acc_2mm'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()
            log['val/acc_4mm'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()
            log['val/abs_err_old'] = abs_error(depth_old, depth_gt, mask).mean()
            log['val/acc_1mm_old'] = acc_threshold(depth_old, depth_gt, mask, 1).mean()
            log['val/acc_2mm_old'] = acc_threshold(depth_old, depth_gt, mask, 2).mean()
            log['val/acc_4mm_old'] = acc_threshold(depth_old, depth_gt, mask, 4).mean()
            # the ratio of the loss
            log['val/abs_err_ratio'] = abs_err/(1e-10 + abs_error(depth_old, depth_gt, mask).mean())
            log['val/acc_1mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 1).mean())
            log['val/acc_2mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 2).mean())
            log['val/acc_4mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 4).mean())
            self.log_dict(log, on_epoch=True, on_step=True)

        return {'loss': loss,
                'progress_bar': {'train_abs_err': abs_err},
                'log': log
               }

        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)

        img_views = imgs.shape[1]
        imgs = imgs.transpose(1, 2)

        imgs_new = self.forward(imgs)

        content_loss = self.calculate_contentloss(imgs_new, imgs.transpose(1,2))* self.lambda_content
        imgs_new = imgs_new.transpose(1, 2)
        imgs.transpose(1,2)

        results = self.depthmodel(imgs_new, proj_mats, init_depth_min, depth_interval)
        result_original = self.depthmodel(imgs, proj_mats, init_depth_min, depth_interval)
        loss_original = self.calculate_depthloss(result_original, depths, masks)
        loss_depth = self.calculate_depthloss(results, depths, masks)

       
       
        loss = loss_depth+content_loss
        self.log("val_content_loss", content_loss, on_epoch=True,on_step=True)

        self.log('val_loss', loss,on_epoch=True)
       
        self.log("val_depth_loss", loss_depth, on_step=True, on_epoch=True)
       
        ration_updata_loss = loss_depth/(1e-10 + loss_original)
        self.log('updated_ori_ratio', ration_updata_loss, on_epoch=True)
        log ={}
        with torch.no_grad():
            if batch_idx%10 == 0:
                try:
               
                    img_ = self.unpreprocess(imgs_new[0,0]).cpu() # batch 0, ref image
                    depth_gt_ = visualize_depth(depths['level_0'][0])
                    depth_pred_ = visualize_depth(results['depth_0'][0]*masks['level_0'][0])
                    prob = visualize_prob(results['confidence_0'][0]*masks['level_0'][0])
                    stack = torch.stack([img_, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                    self.logger.experiment.add_images('train/image_GT_pred_prob',
                                                    stack, self.global_step)
                    
            
                    img_ = self.unpreprocess(imgs[0,0]).cpu() # batch 0, ref image
                    depth_gt_ = visualize_depth(depths['level_0'][0])
                    depth_pred_ = visualize_depth(result_original['depth_0'][0]*masks['level_0'][0])
                    prob = visualize_prob(result_original['confidence_0'][0]*masks['level_0'][0])
                    stack = torch.stack([img_, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                    self.logger.experiment.add_images('train/image_GT_pred_prob_old',
                                                    stack, self.global_step)
                    log['error'] = 0
                except:
                    log['error'] = 1
            

            depth_pred = results['depth_0']
            depth_old = result_original['depth_0']
            depth_gt = depths['level_0']
            mask = masks['level_0']
            log['val/abs_err'] = abs_err = abs_error(depth_pred, depth_gt, mask).mean()
            log['val/acc_1mm'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()
            log['val/acc_2mm'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()
            log['train/acc_4mm'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()
            log['train/abs_err_old'] = abs_error(depth_old, depth_gt, mask).mean()
            log['train/acc_1mm_old'] = acc_threshold(depth_old, depth_gt, mask, 1).mean()
            log['train/acc_2mm_old'] = acc_threshold(depth_old, depth_gt, mask, 2).mean()
            log['train/acc_4mm_old'] = acc_threshold(depth_old, depth_gt, mask, 4).mean()
            # the ratio of the loss
            log['train/abs_err_ratio'] = abs_err/(1e-10 + abs_error(depth_old, depth_gt, mask).mean())
            log['train/acc_1mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 1).mean())
            log['train/acc_2mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 2).mean())
            log['train/acc_4mm_ratio'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()/(1e-10 + acc_threshold(depth_old, depth_gt, mask, 4).mean())
            # log metrics
            self.log_dict(log, on_epoch=True, on_step=True)


        return loss
    def calculate_depthloss(self, results, depths, masks):
        depth_loss = self.depth_loss(results, depths, masks)
        return depth_loss
    def calculate_contentloss(self, x_transform, x_original):

        # what we wanted is : the transformed image smooth, don't have many high frequency noise, abrubt changes
        # the changes on original images should be proportional to the difference of predicted depth and gt depth
        # gradient smootheness
        b,c,d,h,w = x_transform.shape
        x_transform = rearrange(x_transform,'b c d h w -> (b d) c h w')
        x_original = rearrange(x_original,'b c d h w -> (b d) c h w')

        # ssim_loss = pytorch_ssim.SSIM(window_size=11)
        # ssim_loss = 1 - ssim_loss(x_transform,x_original)

        
        features_original = self.vgg(x_original)
        features_transformed = self.vgg(x_transform)
        content_loss = self.l2_loss(features_transformed.relu2_2, features_original.relu2_2)
        return content_loss
    


class ResBlock_3d(nn.Module):
    def __init__(self, nf,dropout = 0.1):
        super(ResBlock_3d, self).__init__()
        self.dcn0 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.dcn1 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.dcn1(self.lrelu(self.dcn0(x)))) + x

class ResBlock(nn.Module):
    def __init__(self, nf,dropout = 0.1):
        super(ResBlock, self).__init__()
        self.dcn0 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.dcn1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        return self.dropout(self.dcn1(self.lrelu(self.dcn0(x)))) + x

    


if __name__ == "__main__":
    
    
    class configs:
        lambda_content = 50
        lambda_style = 1
        
        lambda_ssim = 1 
       
        upscale_factor = 1
        in_channel = 3
        out_channel = 9
        nf = 32

   
    
    
    #train(train_loader,model,10)
    # save model

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/root/autodl-tmp/ckpts/',
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
    

    import optuna
    
    def build_model(trial,
                     lambda_content, upscale_factor,
                       in_channel, out_channel, model,):
        
        nf = trial.nf
        lr = trial.lr
        num_groups2 =trial.num_groups2
        num_groups1 = trial.num_groups1

        print(f'nf: {nf}, lambda_content: {lambda_content}, \
                upscale_factor:{upscale_factor}, in_channel: {in_channel}, \
                out_channel: {out_channel}, lr: {lr}')

        configs = namedtuple('configs', ['nf',
                                          'lambda_content', 
                                          'upscale_factor', 'in_channel', 'out_channel', 'model','lr'])
        configs.nf = nf
        configs.lambda_content = lambda_content
        configs.upscale_factor = upscale_factor
        configs.in_channel = in_channel
        configs.out_channel = out_channel
        configs.model = model
        configs.num_groups1 = num_groups1
        configs.num_groups2 = num_groups2
        configs.lr = lr

        return Net(configs)
        
    
        
        

       
        
        
    from pytorch_lightning.loggers import TensorBoardLogger
   
    logger = TensorBoardLogger('/root/autodl-tmp/logs', name='d3c_net')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_ratio: refined/original',
        dirpath='/root/autodl-tmp/checkpoints/d3n',
        filename='d3c_net_128_{epoch}',
        save_top_k=1,
        mode='min',
        save_last=True

        )
    early_stop_callback = EarlyStopping(
            monitor='val_ratio: refined/original',
            patience=10,
            verbose=False,
            mode='min'
        )
    trainer = Trainer(max_epochs=200, 
                      gpus=1,
                    strategy='ddp',
                    
                    callbacks=[checkpoint_callback, 
                               #early_stop_callback
                               ]
                    ,      
                  
                    val_check_interval=1.0,
                    logger=logger,
                    # resume_from_checkpoint='/root/autodl-tmp/project/dp_simple/ckpts/d3c_net_epoch=54.ckpt'
                    )
    model = CascadeMVSNet(n_depths=[8,32,48],
                        interval_ratios=[1.0,2.0,4.0],
                        num_groups=1,
                        
                        norm_act=ABN).cuda()
    load_ckpt(model, '/root/autodl-tmp/project/dp_simple/CasMVSNet_pl/ckpts/_ckpt_epoch_10.ckpt')


    trial = namedtuple('trial', ['nf', 'lr', 'num_groups1', 'num_groups2'])
    trial.nf = 64
    trial.lr = 1e-4
    trial.num_groups1 = 5
    trial.num_groups2 = 6
    model = build_model(trial, 100,1, 3,9, model)

    train_dataset = DTUDataset('/root/autodl-tmp/mvs_training/dtu/', 'train')
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
        
    val_dataset = DTUDataset('/root/autodl-tmp/mvs_training/dtu/', 'val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

        
    trainer.fit(model, train_loader, val_loader)

    