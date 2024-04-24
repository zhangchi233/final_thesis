import os, sys
from opt import get_opts
import torch

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.mvsnet import CascadeMVSNet,CascadeMVSNetIDW
from inplace_abn import InPlaceABN

from torchvision import transforms as T
from models.ddm import Net
# optimizer, scheduler, visualization
from utils import *
from einops import rearrange
# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping      
from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning.loggers import TestTubeLogger

class MVSSystem(LightningModule):

    def __init__(self,args,config = None):

        super(MVSSystem, self).__init__()
        
        self.depth_transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        
        self.config = config
        self.hparam = args
        self.loss = loss_dict["sl1"]

        self.model =  Net(args,config)
        if self.config.model.use_depth:
            self.depth_loss = loss_dict["sl1"]()
            self.depth_model = CascadeMVSNetIDW(n_depths=args.n_depths,
                                   interval_ratios=args.interval_ratios,
                                   num_groups=args.num_groups,
                                   norm_act=InPlaceABN)
            
        if self.hparam.ckpt_path != '':
            self.original_depth_model= CascadeMVSNet(n_depths=args.n_depths,
                                   interval_ratios=args.interval_ratios,
                                   num_groups=args.num_groups,
                                   norm_act=InPlaceABN)
            print('Load model from', self.hparam.ckpt_path)
            load_ckpt(self.original_depth_model, self.hparam.ckpt_path,"loss")
            for name,param in self.depth_model.named_parameters():
                if name in self.original_depth_model.state_dict():
                    if param.shape == self.original_depth_model.state_dict()[name].shape:
                        param.data = self.original_depth_model.state_dict()[name].data
            for param in self.original_depth_model.parameters():
                param.requires_grad = False
                    
                      
            
                    
            
            
        else:
            self.original_depth_model = None
        
        
        
       
        
        

        # if num gpu is 1, print model structure and number of params
        if self.hparam.num_gpus == 1:
            # print(self.model)
            print('number of parameters : %.2f M' % 
                  (sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6))
        
        # load model if checkpoint path is provided
        if self.hparam.resume != '':
            print('Load model from', self.hparam.resume)
            self.load_ddm_ckpt(self.hparam.resume, ema=False)
        
        
    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model = self.model.module
        # self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        # if ema:
        #     self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, 0))
    def decode_batch(self, batch):
        
        
        imgs = batch['imgs']

        proj_mats = batch['proj_mats']
        depths = batch['depths']
        masks = batch['masks']
        init_depth_min = batch['init_depth_min']
        depth_interval = batch['depth_interval']
        return imgs, proj_mats, depths, masks, init_depth_min, depth_interval

    def forward(self, batch):
        img_ori = batch["imgs"]
        img_gt = batch["imgs"]
        x = torch.cat([img_ori, img_gt], dim=2 if img_ori.ndim == 5 else 1)
               
        x = rearrange(x, 'b v c h w -> (b v) c h w ') if x.ndim == 5 else x

        y = batch["scan_vid"]
                
       
                
                

        
        
        output = self.model(x)
        # img_gt = self.depth_transform(img_ori)
        
        output["img"] = img_gt
       

        return output

    def prepare_data(self):
        dataset = dataset_dict[self.config.data.type]
        self.train_dataset = dataset(root_dir=self.config.data.root_dir,
                                     split='train',
                                    )
        self.val_dataset = dataset(root_dir=self.config.data.root_dir,
                                   split='test',
                                  )

    def configure_optimizers(self):
       
       
        self.optimizer = get_optimizer(self.config.type, self.depth_model.feature)
        for name,param in self.depth_model.named_parameters():
           
            if "feature" in name:
                
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False
        scheduler = get_scheduler(self.config.type, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.config.training.batch_size,
                          pin_memory=True)
        print('train dataset size:', len(self.train_dataset))
        
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                          
                          num_workers=4,
                          batch_size=self.config.training.batch_size,
                          pin_memory=True)
        print('val dataset size:', len(self.train_dataset))
        return val_loader
    def on_train_start(self) -> None:
        pth_path = f'/root/autodl-tmp/Diffusion/ckpts/{config.logger.exp_name}'
        pth_path = os.path.join(pth_path, 'version_encoder',f"epoch_{self.current_epoch}.pth")
        if os.path.exists(pth_path):
            self.depth_model.load_state_dict(torch.load(pth_path))
            print("load model from", pth_path)
        
       


        return super().on_train_start()

    def on_train_epoch_start(self) -> None:
        self.model.train()

    def on_validation_epoch_start(self) -> None:
        self.model.train()
    
    def predict_depth(self,batch):
        if len(batch["imgs"].shape) != 5:
            batch["imgs"] = batch["imgs"].unsqueeze(0)
            batch["proj_mats"] = batch["proj_mats"].unsqueeze(0)
            batch["imgs"] = batch["imgs"].to(self.device)
            batch["proj_mats"] = batch["proj_mats"].to(self.device)
            batch["init_depth_min"] = batch["init_depth_min"].unsqueeze(0)
            batch["depth_interval"] = batch["depth_interval"].unsqueeze(0)
            batch["init_depth_min"] = batch["init_depth_min"].to(self.device)
            batch["depth_interval"] = batch["depth_interval"].to(self.device)


        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
        outputs = self(batch)
        results = self.depth_model(outputs, proj_mats, init_depth_min, depth_interval)
        return results
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        outputs = self(batch)
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(batch)
       
        
       
        results = self.depth_model(outputs, proj_mats, init_depth_min, depth_interval)
        depth_loss = self.depth_loss(results, depths, masks)
        loss = depth_loss   
        log['train/loss'] = loss
        depth_pred = results['depth_0']
        depth_gt = depths['level_0']
        mask = masks['level_0']
        log['train/abs_err'] = abs_err = abs_error(depth_pred, depth_gt, mask).mean()
        log['train/acc_1mm'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()
        log['train/acc_2mm'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean()
        log['train/acc_4mm'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean()
        with torch.no_grad():
            if self.global_step % 50 == 0:
                imgs_ = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])(imgs)    
                original_results= self.original_depth_model(imgs_, proj_mats, init_depth_min, depth_interval)
                
                original_depth_loss = self.depth_loss(original_results, depths, masks)
            
                self.log('train/original', original_depth_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                log["train/original_depth_loss"] = original_depth_loss
                log["train/ori_abs_err"] = ori_abs_err = abs_error(original_results['depth_0'], depth_gt, mask).mean()
                log["train/ori_acc_1mm"] = ori_acc_1mm = acc_threshold(original_results['depth_0'], depth_gt, mask, 1).mean()
                log["train/ori_acc_2mm"] = ori_acc_2mm = acc_threshold(original_results['depth_0'], depth_gt, mask, 2).mean()
                log["train/ori_acc_4mm"] = ori_acc_4mm = acc_threshold(original_results['depth_0'], depth_gt, mask, 4).mean()

                log["train/depth_loss_ratio"] = depth_loss/(original_depth_loss+1e-7)
                log["train/abs_err_ratio"] = abs_err/(ori_abs_err+1e-7)
                log["train/acc_1mm_ratio"] = log["train/acc_1mm"]/(ori_acc_1mm+1e-7)
                log["train/acc_2mm_ratio"] = log["train/acc_2mm"]/(ori_acc_2mm+1e-7)
                log["train/acc_4mm_ratio"] = log["train/acc_4mm"]/ (ori_acc_4mm+1e-7)

                pred_x = outputs["pred_x"]
                if len(imgs.shape) == 5:
                    imgs = rearrange(imgs, 'b v c h w ->b c h (v w)')
                    img_gt =batch["imgs_gt"]
                    img_gt = rearrange(img_gt, 'b v c h w ->(b v) c h w')

                
                # self.logger.experiment.add_images('train/input_gt_pred',imgs, self.global_step)
                
                # self.logger.experiment.add_images('train/gt',img_gt, self.global_step,)
                # self.logger.experiment.add_images('train/pred',pred_x, self.global_step)
                # log image modifying scenario
                
                img_save_directiory = f'/root/autodl-tmp/Diffusion/ckpts/{config.logger.exp_name}/images'
                if not os.path.exists(img_save_directiory):
                    os.makedirs(img_save_directiory)
                save_image(img_gt, os.path.join(img_save_directiory, f'gt_{self.global_step}.png')
                           )
                            
                save_image(pred_x,os.path.join(img_save_directiory, f'pred_{self.global_step}.png'), 
                          )
                

                
                
                
        self.log_dict(log, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            
            
        return {'loss': loss,
                'progress_bar': {'train_abs_err': abs_err},
                'log': log
               }
    @torch.no_grad()
    def validation_step(self, batch, batch_nb):
        self.model.train()
        log = {'lr': get_learning_rate(self.optimizer)}
        imgs, proj_mats, depths, masks, init_depth_min, depth_interval = \
            self.decode_batch(batch)
        outputs = self(batch)
        results = self.depth_model(outputs, proj_mats, init_depth_min, depth_interval)
        log['val_loss'] = self.depth_loss(results, depths, masks)
        self.logger.experiment.add_text('val/loss', str(log['val_loss'].mean()), self.global_step)
    
        if batch_nb%50 == 0:
            img_ = imgs[0,0].cpu() # batch 0, ref image
            depth_gt_ = visualize_depth(depths['level_0'][0])
            depth_pred_ = visualize_depth(results['depth_0'][0]*masks['level_0'][0])
            prob = visualize_prob(results['confidence_0'][0]*masks['level_0'][0])
            stack = torch.stack([img_, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
            self.logger.experiment.add_images('val/image_GT_pred_prob',
                                                stack, self.global_step)

        depth_pred = results['depth_0']
        depth_gt = depths['level_0']
        mask = masks['level_0']

        log['val_abs_err'] = abs_error(depth_pred, depth_gt, mask).mean()
        log['val_acc_1mm'] = acc_threshold(depth_pred, depth_gt, mask, 1).mean()
        log['val_acc_2mm'] = acc_threshold(depth_pred, depth_gt, mask, 2).mean() 
        log['val_acc_4mm'] = acc_threshold(depth_pred, depth_gt, mask, 4).mean() 
        log['mask_sum'] = mask.float().sum()
       
        self.logger.experiment.add_scalar('val/loss', log['val_loss'].mean(), self.global_step)
        self.logger.experiment.add_scalar('val/abs_err', log['val_abs_err'], self.global_step)
        self.logger.experiment.add_scalar('val/acc_1mm', log['val_acc_1mm'], self.global_step)
        self.logger.experiment.add_scalar('val/acc_2mm', log['val_acc_2mm'], self.global_step)
        self.logger.experiment.add_scalar('val/acc_4mm', log['val_acc_4mm'], self.global_step)
        self.log_dict(log, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return log

    def validation_epoch_end(self, outputs):
        # save the depth_model's weight
        save_path = f'/root/autodl-tmp/Diffusion/ckpts/{config.logger.exp_name}/version_encoder'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_pth = os.path.join(save_path, f'epoch_{self.current_epoch}.pth')
        torch.save(self.depth_model.state_dict(), save_pth)


        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
       
        mean_abs_err = torch.stack([x['val_abs_err'] for x in outputs]).mean() 
        mean_acc_1mm = torch.stack([x['val_acc_1mm'] for x in outputs]).mean() 
        mean_acc_2mm = torch.stack([x['val_acc_2mm'] for x in outputs]).mean() 
        mean_acc_4mm = torch.stack([x['val_acc_4mm'] for x in outputs]).mean() 
        self.log('val/loss', mean_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/abs_err', mean_abs_err, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc_1mm', mean_acc_1mm, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc_2mm', mean_acc_2mm, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc_4mm', mean_acc_4mm, on_epoch=True, prog_bar=True, logger=True)

        
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
    import argparse
    import yaml

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    def parse_args_and_config():
        parser = argparse.ArgumentParser(description='Training Wavelet-Based Diffusion Model')
        parser.add_argument('--n_views', type=int, default=3,
                        help='number of views (including ref) to be used in training')
        parser.add_argument('--levels', type=int, default=3, choices=[3],
                            help='number of FPN levels (fixed to be 3!)')
        parser.add_argument('--depth_interval', type=float, default=2.65,
                            help='depth interval for the finest level, unit in mm')
        parser.add_argument('--n_depths', nargs='+', type=int, default=[8,32,48],
                            help='number of depths in each level')
        parser.add_argument('--interval_ratios', nargs='+', type=float, default=[1.0,2.0,4.0],
                            help='depth interval ratio to multiply with --depth_interval in each level')
        parser.add_argument('--num_groups', type=int, default=1, choices=[1, 2, 4, 8],
                        help='number of groups in groupwise correlation, must be a divisor of 8')
        parser.add_argument("--config", default='/root/autodl-tmp/Diffusion/configs/DTU2.yml', type=str,
                            help="Path to the config file")
        parser.add_argument('--resume', default='/root/autodl-tmp/model.pth.tar', type=str,
                            help='Path for checkpoint to load and resume')
        parser.add_argument("--optimizer", default='adam', type=str,)
        parser.add_argument("--lr_scheduler", default='steplr', type=str,)
        parser.add_argument("--sampling_timesteps", type=int, default=10,
                            help="Number of implicit sampling steps for validation image patches")
        parser.add_argument("--image_folder", default='results/', type=str,
                            help="Location to save restored validation image patches")
        parser.add_argument('--seed', default=230, type=int, metavar='N',
                            help='Seed for initializing training (default: 230)')
        parser.add_argument('--num_gpus', default=2, type=int, metavar='N',
                            help='Number of GPUs to use')
        parser.add_argument('--use_amp', action='store_true', default=True,
                            help='Use Automatic Mixed Precision')
        parser.add_argument('--num_epochs', default=5, type=int, metavar='N',
                            help='Number of epochs to train')
        parser.add_argument("--ckpt_path", default='/root/autodl-tmp/project/dp_simple/CasMVSNet_pl/ckpts/_ckpt_epoch_10.ckpt', type=str,)

        args = parser.parse_args()

        with open( args.config, "r") as f:
            config = yaml.safe_load(f)
        new_config = dict2namespace(config)

        return args, new_config
    args, config = parse_args_and_config()

    system = MVSSystem( args, config)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(f'/root/autodl-tmp/Diffusion/ckpts/{config.logger.exp_name}','version_encoder'),
                                          monitor='val/acc_2mm',
                                          mode='max',
                                          every_n_epochs=1.0,
                                          save_last=True,
                                          save_top_k=5,)
    
    early_stop_callback = EarlyStopping(monitor='val/acc_2mm',
                                        patience=10,
                                        mode='max')  
    lr_scheduler = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_scheduler, early_stop_callback]
    logger = TestTubeLogger(
        save_dir="/root/tf-logs",
        name=config.logger.exp_name,
        debug=False,
        create_git_tag=False
    )
    trainer = Trainer(max_epochs=args.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      gpus= 1,
                      strategy = "ddp",
                      num_sanity_val_steps=5,
                      check_val_every_n_epoch=1,

                    

                      
                      
                      precision=16)
    # trainer = Trainer(max_epochs=args.num_epochs,
    #                   checkpoint_callback=checkpoint_callback,
    #                   logger=logger,
    #                   early_stop_callback=None,
    #                   weights_summary=None,
    #                   progress_bar_refresh_rate=1,
    #                   gpus=args.num_gpus,
    #                   distributed_backend='ddp' if args.num_gpus>1 else None,
    #                   num_sanity_val_steps=0 if args.num_gpus>1 else 3,
    #                   benchmark=True,
    #                   precision= 16,
    #                   amp_level='O1')
    system.prepare_data()
    trainer.fit(system)