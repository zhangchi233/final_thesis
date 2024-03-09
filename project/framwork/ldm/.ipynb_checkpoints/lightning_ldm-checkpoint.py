import pytorch_lightning as pl
from diffusers import VQModel,UNet2DModel,DDIMScheduler
import torch
import os
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer,CLIPVisionModel
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau
#from models.pano.MVGenModel import MultiViewBaseModel
from einops import rearrange
from torchvision import transforms as T
import open_clip 
import sys
import math
sys.path.append("/root/autodl-tmp/project")
from data.dtu import DTUDataset
from modules.unClip import ReprogrammingLayer
sys.path.append("/root/autodl-tmp/project/framwork/ldm")
from denoise.denoise import denoiseModel
class PanoOutpaintGenerator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs'] if 'max_epochs' in config['train'] else 0
        self.diff_timestep = config['model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']

        self.views = [0,1,2]



        self.vae, self.scheduler, unet = self.load_model(
            config['model']['model_id'])
        


     
        unet.conv_in = torch.nn.Conv2d(12, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        

        self.model = denoiseModel(unet)
        self.trainable_params = [(
            list(self.model.parameters()), 1
        )]
      
       


        

        #self.mv_base_model = MultiViewBaseModel(
        #    unet, config['model'])
        #self.trainable_params = self.mv_base_model.trainable_parameters
        #self.trainable_params = [(list(self.text_encoder.parameters())+list(self.vae.parameters()),1)]
        
        #self.save_hyperparameters()
        
    def forward(self, latents, timestep):
        return self.model(latents, timestep)
    def load_model(self, model_id):
        vae = VQModel.from_pretrained(
            #"/root/autodl-tmp/inpaiting_model/vae"
            model_id, subfolder="vqvae",
            )
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(
            #"/root/autodl-tmp/inpaiting_model/scheduler")
            model_id, subfolder="scheduler")
        unet = UNet2DModel.from_pretrained(
            #"/root/autodl-tmp/inpaiting_model/unet"
            model_id, subfolder="unet"
            )
        return vae, scheduler, unet

    

    
    


    @torch.no_grad()
    def encode_image(self, x_input, vae):
       
        x_input = x_input.float().to(vae.dtype)
        x_input = x_input.to(vae.device)

        z = vae.encode(x_input).latents  # (bs, 2, 4, 64, 64) # z vector mean and variance
        # vqvae
        

        # use the scaling factor from the vae config
       
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        #latents = (1 / vae.config.scaling_factor * latents)

        images = []
            
        image = vae.decode(latents).sample
        
            

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype('uint8')
        return image

    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            param_groups.append({"params": params, "lr": self.lr * lr_scale})
        optimizer = torch.optim.AdamW(param_groups)
        
        scheduler = {
            #
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
            #ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, threshold=0.0001),
            #CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
            'interval': 'epoch',  # update the learning rate after each epoch
            'name': 'cosine_annealing_lr',
            "monitor": "val_loss"
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    
    def training_step(self, batch, batch_idx):


        
       
        meta = {
           
            "mask": batch['mask'],
           
        }

        images=batch['imgs']
        blur_images = batch['dark_imgs'] # input, blur image, condition mask
       
        #images=rearrange(images, 'bs m h w c -> bs m c h w')
        #blur_images=rearrange(blur_images, 'bs m c h w -> bs m c h w')

        images = images[:,0]
        blur_images = blur_images[:,self.views]
        
       
    
        
        
        
        
        latents = self.encode_image(images, self.vae)
        
        blur_images= rearrange(blur_images, 'bs m c h w -> bs (m c) h w')
        
        blur_images = torch.nn.functional.interpolate(blur_images, size=(
                                                                         128, 128),
                                                    mode='bilinear', align_corners=False)


        
       
        t = torch.randint(0, self.scheduler.num_train_timesteps,
                        (latents.shape[0],), device=latents.device).long()
        

        noise = torch.randn_like(latents)
        
        noise_z = self.scheduler.add_noise(latents, noise, t)
        
        # b,m,c,h,w 
        # the input should be latents_input,
        latents_input = torch.cat([noise_z,blur_images ], dim=1)
        latents_input = self.scheduler.scale_model_input(latents_input, t)
        denoise = self.forward(latents_input, t)
        target = noise
        target = target.to(torch.float32)
        denoise = denoise.to(torch.float32)
        

       

        # eps mode
        loss = torch.nn.functional.mse_loss(denoise, target)
        print("loss is nan",loss.isnan())
        self.log("train_loss", loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        
        
        return loss
    
        

    

    def psnr(self,img1,img2):
        mse = np.mean((img1/255. - img2/255.) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred = self.inference(batch)
        #images = ((batch['images']/2+0.5)
        #                  * 255).cpu()
        #blur_images = ((batch['dark_imgs']/2+0.5)
        #                  * 255).cpu()
        
        mask = batch["mask"]
        gt_imgs = batch["imgs"][:,0].permute(0,2,3,1).cpu().numpy()
        gt_imgs = gt_imgs/2+0.5
        dark_imgs = batch["dark_imgs"]/2+0.5
        dark_imgs = rearrange(dark_imgs, 'bs m c h w -> bs c h (m w)')
        dark_imgs = dark_imgs.permute(0,2,3,1).cpu().numpy()
        gt_imgs = (gt_imgs*255).round().astype('uint8')
        dark_imgs = (dark_imgs*255).round().astype('uint8')
        val_loss = np.mean(np.square(images_pred - gt_imgs))
        sample ={}
        psnr = self.psnr(images_pred,gt_imgs)
        sample["psnr"] = psnr
    
        sample["gt_imgs"] = gt_imgs
        sample["dark_imgs"] = dark_imgs
        sample["pred_imgs"] = images_pred
        sample["mask"] = mask
        sample["val_loss"] = val_loss
    
        self.log("psnr", psnr,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        self.log("val_loss", val_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        
   
        return sample
        # compute image & save
        #if self.trainer.global_rank == 0:
        #    self.save_image(images_pred, images,blur_images, batch['prompt'], batch_idx)

    @torch.no_grad()
    def inference(self, batch):
       
        images = batch['dark_imgs'][:,self.views]
      
        mask = batch["mask"]

        bs, m, c,h, w = images.shape
        device = images.device
        latents_shape = (bs, c, h//4, w//4)
        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps
        images = rearrange(images, 'bs m c h w -> bs (m c) h w')

        images = torch.nn.functional.interpolate(images,
                                                 size = (128, 128),
                                                    mode='bilinear', align_corners=False)
        


        latents = torch.randn(latents_shape, device=device)
        
        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]], dim=1)
            latent_model_input = torch.cat([latents,images], dim=1)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.forward(latent_model_input,t)
            
            latents = self.scheduler.step(
                noise_pred, t, latents,eta = 1).prev_sample

        images_pred = self.decode_latent(
            latents, self.vae)
       
        return images_pred
    