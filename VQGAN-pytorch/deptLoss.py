import sys
sys.path.append('/root/autodl-tmp/project/dp_simple/')
#import ViT

from torchvision import transforms as T
from CasMVSNet_pl.models.mvsnet import CascadeMVSNet
from CasMVSNet_pl.utils import load_ckpt
from CasMVSNet_pl.datasets.dtu import DTUDataset  
from CasMVSNet_pl.utils import *
from CasMVSNet_pl.datasets.dtu import DTUDataset 
from CasMVSNet_pl.metrics import *  
from inplace_abn import ABN

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

class SL1Loss(nn.Module):
    def __init__(self, levels=3,nviews = 3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')
        model = CascadeMVSNet(n_depths=[8,32,48],
                        interval_ratios=[1.0,2.0,4.0],
                        num_groups=1,
                        
                        norm_act=ABN).cuda()
        load_ckpt(model, '/root/autodl-tmp/project/dp_simple/CasMVSNet_pl/ckpts/_ckpt_epoch_10.ckpt')
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.nviews = nviews
    def depthloss(self,inputs,targets,masks):
        loss = 0
        for l in range(self.levels):
            depth_pred_l = inputs[f'depth_{l}']
            depth_gt_l = targets[f'level_{l}']
            mask_l = masks[f'level_{l}']
            loss += self.loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2**(1-l)
        return loss

    def forward(self, preds, imgs, proj_mats, depths, masks, init_depth_min, depth_interval):
        imgs = rearrange(imgs, 'b c h (n w) -> b n c h w', n=self.nviews)
        preds = rearrange(preds, 'b c h (n w) -> b n c h w', n=self.nviews)
        

        results = self.model(preds, proj_mats, init_depth_min, depth_interval)
        result_original = self.model(imgs, proj_mats, init_depth_min, depth_interval)
        depth_loss = self.depthloss(results, depths, masks)
        with torch.no_grad():
            loss_original = self.depthloss(result_original, depths, masks)
        log = {}
        gt_depth = depths['level_0']
        mask = masks['level_0']
        pred_depth = results['depth_0']
        original_depth = result_original['depth_0']

        log['abs_error'] = abs_error(pred_depth, gt_depth, mask).mean()
        log['abs_error_original'] = abs_error(original_depth, gt_depth, mask).mean()
        log['acc_1mm'] = acc_threshold(pred_depth, gt_depth, mask, 1).mean()
        log['acc_1mm_original'] = acc_threshold(original_depth, gt_depth, mask, 1).mean()
        log['acc_2mm'] = acc_threshold(pred_depth, gt_depth, mask, 2).mean()
        log['acc_2mm_original'] = acc_threshold(original_depth, gt_depth, mask, 2).mean()
        log['acc_3mm'] = acc_threshold(pred_depth, gt_depth, mask, 3).mean()
        log['acc_3mm_original'] = acc_threshold(original_depth, gt_depth, mask, 3).mean()
        log['acc_4mm'] = acc_threshold(pred_depth, gt_depth, mask, 4).mean()
        log['acc_4mm_original'] = acc_threshold(original_depth, gt_depth, mask, 4).mean()
        # ratio
        log['abs/abs_original'] = log['abs_error']/log['abs_error_original']
        log['acc_1mm/acc_1mm_original'] = log['acc_1mm']/log['acc_1mm_original']
        log['acc_2mm/acc_2mm_original'] = log['acc_2mm']/log['acc_2mm_original']
        log['acc_3mm/acc_3mm_original'] = log['acc_3mm']/log['acc_3mm_original']
        log['acc_4mm/acc_4mm_original'] = log['acc_4mm']/log['acc_4mm_original']

        
        return depth_loss, loss_original,log
    
        
