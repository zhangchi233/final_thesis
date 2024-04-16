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
from einops import rearrange
# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger

class MVSSystem(LightningModule):
   
    def __init__(self, args,config = None):

        super(MVSSystem, self).__init__()
        
        self.depth_transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        
        self.config = config
        self.hparams = args
        self.loss = loss_dict["refine_loss"]

        self.model =  Net(args,config)
        
        if self.hparams.cas_ckpt_path != '':
            self.depth_model = CascadeMVSNet(n_depths=self.args.n_depths,
                                   interval_ratios=self.args.interval_ratios,
                                   num_groups=self.args.num_groups,
                                   norm_act=InPlaceABN)
            print('Load model from', self.hparams.ckpt_path)
            load_ckpt(self.depth_model, self.hparams.ckpt_path, self.hparams.prefixes_to_ignore)
        
        
        
       
        
        

        # if num gpu is 1, print model structure and number of params
        if self.hparams.num_gpus == 1:
            # print(self.model)
            print('number of parameters : %.2f M' % 
                  (sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6))
        
        # load model if checkpoint path is provided
        if self.hparams.resume != '':
            print('Load model from', self.hparams.resume)
            self.load_ddm_ckpt(self.hparams.resume, ema=False)
        if self.config.model.use_depth:
            self.depth_loss = loss_dict["sl1"]
    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))
    def decode_batch(self, batch):
        
        return imgs,gt_imgs

    def forward(self, batch):
        img_ori = batch["imgs"]
        img_gt = batch["imgs_gt"]
        x = torch.cat([img_ori, img_gt], dim=2 if img_ori.ndim == 5 else 1)
               
        x = rearrange(x, 'b v c h w -> (b v) c h w ') if x.ndim == 5 else x

        y = batch["scan_vid"]
                
       
                
                

        

        output = self.model(x)
       

        return output

    def prepare_data(self):
        dataset = dataset_dict[self.config.data.type]
        self.train_dataset = dataset(root_dir=self.config.data.root_dir,
                                     split='train',
                                    )
        self.val_dataset = dataset(root_dir=self.config.data.root_dir,
                                   split='val',
                                  )

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.config.type, self.model)
        scheduler = get_scheduler(self.config.type, self.optimizer)
        
        return [self.optimizer]

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
    def on_train_epoch_start(self) -> None:
        self.model.train()

    def on_validation_epoch_start(self) -> None:
        self.model.train()
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        outputs = self(batch)
        img_ori = batch["imgs"]
        img_gt = batch["imgs_gt"]
        x = torch.cat([img_ori, img_gt], dim=2 if img_ori.ndim == 5 else 1)
        x = rearrange(x, 'b v c h w -> (b v) c h w ') if x.ndim == 5 else x 
        noise_loss, photo_loss, frequency_loss =  self.loss(x, outputs)
        if self.config.model.use_depth:


        loss = noise_loss + photo_loss + frequency_loss
        log['train/loss'] = loss 
        log['train/photo_loss'] = photo_loss
        log['train/noise_loss'] = noise_loss
        log['train/frequency_loss'] = frequency_loss

        
        with torch.no_grad():
            if batch_nb%20 == 0:
                pred_x = outputs["pred_x"]
                if len(img_ori.shape) == 5:
                    img_ori = rearrange(img_ori, 'b v c h w ->b c h (v w)')
                    img_gt = rearrange(img_gt, 'b v c h w ->b c h (v w)')
                    
                
                
                self.logger.experiment.add_images('train/input_gt_pred',img_ori, self.global_step)
                self.logger.experiment.add_images('train/gt',img_gt, self.global_step)
                self.logger.experiment.add_images('train/pred',pred_x, self.global_step)

            
            
        return {'loss': loss,
                'progress_bar': {'train_abs_err': photo_loss},
                'log': log
               }
    @torch.no_grad()
    def validation_step(self, batch, batch_nb):
        self.model.train()
        log = {'lr': get_learning_rate(self.optimizer)}
        outputs = self(batch)
        img_ori = batch["imgs"]
        img_gt = batch["imgs_gt"]
        x = torch.cat([img_ori, img_gt], dim=2 if img_ori.ndim == 5 else 1)
        
        x = rearrange(x, 'b v c h w -> (b v) c h w ') if x.ndim == 5 else x
        noise_loss, photo_loss, frequency_loss =  self.loss(x, outputs)
        loss = noise_loss + photo_loss + frequency_loss
        log['val/loss'] = loss 
        log['val/photo_loss'] = photo_loss
        log['val/noise_loss'] = noise_loss
        log['val/frequency_loss'] = frequency_loss
        
        pred_x = outputs["pred_x"]
        pred_x = pred_x.reshape(img_ori.shape)
        with torch.no_grad():
            if batch_nb%20 == 0:
                if len(img_ori.shape) == 5:
                    img_ori = rearrange(img_ori, 'b v c h w ->b c h (v w)')
                    img_gt = rearrange(img_gt, 'b v c h w ->b c h (v w)')
                    pred_x = rearrange(pred_x, 'b v c h w ->b c h (v w)')
               
                
                self.logger.experiment.add_images('val/input_gt_pred',img_ori, self.global_step)
                self.logger.experiment.add_images('val/gt',img_gt, self.global_step)
                self.logger.experiment.add_images('val/pred',pred_x, self.global_step)

        return log

    def validation_epoch_end(self, outputs):
        photo_loss = torch.stack([x['val/photo_loss'] for x in outputs]).mean()
        loss = torch.stack([x['val/loss'] for x in outputs]).sum()
        noise_loss = torch.stack([x['val/noise_loss'] for x in outputs]).sum() 
        frequency_loss = torch.stack([x['val/frequency_loss'] for x in outputs]).sum()
        
        return {'progress_bar': {'val_loss': loss,
                                 'photo_loss': photo_loss},
                'log': {'val/loss': loss,
                        'val/photo_loss': photo_loss,
                        'val/noise_loss': noise_loss,
                        'val/frequency_loss': frequency_loss,
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
        parser.add_argument('--resume', default='', type=str,
                            help='Path for checkpoint to load and resume')
        parser.add_argument("--optimizer", default='adam', type=str,)
        parser.add_argument("--lr_scheduler", default='steplr', type=str,)
        parser.add_argument("--sampling_timesteps", type=int, default=10,
                            help="Number of implicit sampling steps for validation image patches")
        parser.add_argument("--image_folder", default='results/', type=str,
                            help="Location to save restored validation image patches")
        parser.add_argument('--seed', default=230, type=int, metavar='N',
                            help='Seed for initializing training (default: 230)')
        parser.add_argument('--num_gpus', default=1, type=int, metavar='N',
                            help='Number of GPUs to use')
        parser.add_argument('--use_amp', action='store_true', default=True,
                            help='Use Automatic Mixed Precision')
        parser.add_argument('--num_epochs', default=100, type=int, metavar='N',
                            help='Number of epochs to train')
        parser.add_argument("--cas_ckpt_path", default='/root/autodl-tmp/project/dp_simple/CasMVSNet_pl/ckpts/_ckpt_epoch_10.ckpt', type=str,)

        args = parser.parse_args()

        with open( args.config, "r") as f:
            config = yaml.safe_load(f)
        new_config = dict2namespace(config)

        return args, new_config
    args, config = parse_args_and_config()
    system = MVSSystem( args, config)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'ckpts/{config.logger.exp_name}',
                                                                '{epoch:02d}'),
                                          monitor='val/photo_loss',
                                          mode='max',
                                          save_top_k=5,)

    logger = TestTubeLogger(
        save_dir="/root/tf-logs",
        name=config.logger.exp_name,
        debug=False,
        create_git_tag=False
    )
    # trainer = Trainer(max_epochs=hparams.num_epochs,
    #                   callbacks=[checkpoint_callback],
    #                   logger=logger,
    #                   gpus= 1,
    #                   strategy = "ddp",
    #                   num_sanity_val_steps=0,
    #                   check_val_every_n_epoch=1,
    #                   precision=16)
    trainer = Trainer(max_epochs=args.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      early_stop_callback=None,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=args.num_gpus,
                      distributed_backend='ddp' if args.num_gpus>1 else None,
                      num_sanity_val_steps=0 if args.num_gpus>1 else 5,
                      benchmark=True,
                      precision= 32,
                      amp_level='O1')

    trainer.fit(system)