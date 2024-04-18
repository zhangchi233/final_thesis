import os, sys
from opt import get_opts
import torch

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.mvsnet import CascadeMVSNet
from inplace_abn import InPlaceABN

from torchvision import transforms as T
from models.ddm import Net
# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger

class MVSSystem(LightningModule):
   
    def __init__(self, hparams,config = None):

        super(MVSSystem, self).__init__()
        self.argss = hparams
        # to unnormalize image for visualization
        self.unpreprocess = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                        std=[1/0.229, 1/0.224, 1/0.225])
        self.depth_transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        self.loss = loss_dict[hparams.loss_type](hparams.levels)

        self.model = CascadeMVSNet(n_depths=self.argss.n_depths,
                                   interval_ratios=self.argss.interval_ratios,
                                   num_groups=self.argss.num_groups,
                                   norm_act=InPlaceABN)
        self.refine_img_model = None
       

        # if num gpu is 1, print model structure and number of params
        if self.argss.num_gpus == 1:
            # print(self.model)
            print('number of parameters : %.2f M' % 
                  (sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6))
        
        # load model if checkpoint path is provided
        if self.argss.ckpt_path != '':
            print('Load model from', self.argss.ckpt_path)
            load_ckpt(self.model, self.argss.ckpt_path, self.argss.prefixes_to_ignore)

    def decode_batch(self, batch):
        imgs = batch['imgs']
        imgs = self.depth_transform(imgs)
        proj_mats = batch['proj_mats']
        depths = batch['depths']
        masks = batch['masks']
        init_depth_min = batch['init_depth_min']
        depth_interval = batch['depth_interval']
        return imgs, proj_mats, depths, masks, init_depth_min, depth_interval

    def forward(self, imgs, proj_mats, init_depth_min, depth_interval):
        return self.model(imgs, proj_mats, init_depth_min, depth_interval)

    def prepare_data(self):
        dataset = dataset_dict[self.argss.dataset_name]
        self.train_dataset = dataset(root_dir=self.argss.root_dir,
                                     split='train',
                                     n_views=self.argss.n_views,
                                     levels=self.argss.levels,
                                     depth_interval=self.argss.depth_interval)
        self.val_dataset = dataset(root_dir=self.argss.root_dir,
                                   split='val',
                                   n_views=self.argss.n_views,
                                   levels=self.argss.levels,
                                   depth_interval=self.argss.depth_interval)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.argss, self.model)
        scheduler = get_scheduler(self.argss, self.optimizer)
        
        return [self.optimizer]

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.argss.batch_size,
                          pin_memory=True)
        print('train dataset size:', len(self.train_dataset))
        
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                          
                          num_workers=4,
                          batch_size=self.argss.batch_size,
                          pin_memory=True)
        print('val dataset size:', len(self.train_dataset))
        return val_loader
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = \
            self.decode_batch(batch)
        results = self(imgs, proj_mats, init_depth_min, depth_interval)
        log['train/loss'] = loss = self.loss(results, depths, masks)
        
        with torch.no_grad():
            if batch_nb == 0:
                img_ = self.unpreprocess(imgs[0,0]).cpu() # batch 0, ref image
                depth_gt_ = visualize_depth(depths['level_0'][0])
                depth_pred_ = visualize_depth(results['depth_0'][0]*masks['level_0'][0])
                prob = visualize_prob(results['confidence_0'][0]*masks['level_0'][0])
                stack = torch.stack([img_, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                self.logger.experiment.add_images('train/image_GT_pred_prob',
                                                  stack, self.global_step)

            depth_pred = results['depth_0']
            depth_gt = depths['level_0']
            mask = masks['level_0']
            log['train/abs_err'] = abs_err = abs_error(depth_pred, depth_gt, mask).mean()
            log['train/acc_1mm'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()
            log['train/acc_2mm'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()
            log['train/acc_4mm'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()

        return {'loss': loss,
                'progress_bar': {'train_abs_err': abs_err},
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        log = {}
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = \
            self.decode_batch(batch)
        results = self(imgs, proj_mats, init_depth_min, depth_interval)
        log['val_loss'] = self.loss(results, depths, masks)
    
        if batch_nb == 0:
            img_ = self.unpreprocess(imgs[0,0]).cpu() # batch 0, ref image
            depth_gt_ = visualize_depth(depths['level_0'][0])
            depth_pred_ = visualize_depth(results['depth_0'][0]*masks['level_0'][0])
            prob = visualize_prob(results['confidence_0'][0]*masks['level_0'][0])
            stack = torch.stack([img_, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
            self.logger.experiment.add_images('val/image_GT_pred_prob',
                                                stack, self.global_step)

        depth_pred = results['depth_0']
        depth_gt = depths['level_0']
        mask = masks['level_0']

        log['val_abs_err'] = abs_error(depth_pred, depth_gt, mask).sum()
        log['val_acc_1mm'] = acc_threshold(depth_pred, depth_gt, mask, 1).sum()
        log['val_acc_2mm'] = acc_threshold(depth_pred, depth_gt, mask, 2).sum()
        log['val_acc_4mm'] = acc_threshold(depth_pred, depth_gt, mask, 4).sum()
        log['mask_sum'] = mask.float().sum()

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mask_sum = torch.stack([x['mask_sum'] for x in outputs]).sum()
        mean_abs_err = torch.stack([x['val_abs_err'] for x in outputs]).sum() / mask_sum
        mean_acc_1mm = torch.stack([x['val_acc_1mm'] for x in outputs]).sum() / mask_sum
        mean_acc_2mm = torch.stack([x['val_acc_2mm'] for x in outputs]).sum() / mask_sum
        mean_acc_4mm = torch.stack([x['val_acc_4mm'] for x in outputs]).sum() / mask_sum

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_abs_err': mean_abs_err},
                'log': {'val/loss': mean_loss,
                        'val/abs_err': mean_abs_err,
                        'val/acc_1mm': mean_acc_1mm,
                        'val/acc_2mm': mean_acc_2mm,
                        'val/acc_4mm': mean_acc_4mm,
                        }
               }


if __name__ == '__main__':
    hparams = get_opts()
   
    system = MVSSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(f'ckpts/{hparams.exp_name}',
                                                                '{epoch:02d}'),
                                          monitor='val/acc_2mm',
                                          mode='max',
                                          save_top_k=5,)

    logger = TestTubeLogger(
        save_dir="/openbayes/home/tf_dir",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      gpus= 2,
                      strategy = "ddp",
                      num_sanity_val_steps=0,
                      check_val_every_n_epoch=1,
                      precision=16)
    
    # trainer = Trainer(max_epochs=hparams.num_epochs,
    #                   checkpoint_callback=checkpoint_callback,
    #                   logger=logger,
    #                   early_stop_callback=None,
    #                   weights_summary=None,
    #                   progress_bar_refresh_rate=1,
    #                   gpus=hparams.num_gpus,
    #                   distributed_backend='ddp' if hparams.num_gpus>1 else None,
    #                   num_sanity_val_steps=0 if hparams.num_gpus>1 else 5,
    #                   benchmark=True,
    #                   precision=16 if hparams.use_amp else 32,
    #                   amp_level='O1')
    system.prepare_data()
    trainer.fit(system)