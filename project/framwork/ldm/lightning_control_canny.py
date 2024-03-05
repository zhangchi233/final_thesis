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
import cv2
sys.path.append("/root/autodl-tmp/project")
from data.dtu import DTUDataset
from modules.unClip import ReprogrammingLayer
sys.path.append("/root/autodl-tmp/project/framwork/ldm")
from denoise.denoise_controlnet import denoiseModel
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



     
        #unet.conv_in = torch.nn.Conv2d(12, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        

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

        if len(x_input.shape) == 5:
            x_input = rearrange(x_input, 'b m c h w -> (b m) c h w')
            m = x_input.shape[1]
            z = vae.encode(x_input).latents  # (bs, 2, 4, 64, 64) # z vector mean and variance
            # vqvae
            z = rearrange(z, '(b m) c h w -> b m c h w', m=2)
        else:
            z = vae.encode(x_input).latents
        

        # use the scaling factor from the vae config
       
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        
        #latents = (1 / vae.config.scaling_factor * latents)

        images = []
        latents = rearrange(latents, 'b m c h w -> (b m) c h w')
        image = vae.decode(latents).sample
        image = rearrange(image, '(b m) c h w -> b m h w c', m=m)
        
            

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float().numpy()
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

    @torch.no_grad()
    def canny_edge(self,imgs):
        imgs = imgs.permute(0,2,3,1).cpu().numpy()/2+0.5
        

        imgs = (imgs*255).round().astype('uint8')
        
        # applying histogram equalization
        #imgs = [cv2.equalizeHist(img) for img in imgs]
        # reduce noise gaussian blur
        imgs = [cv2.GaussianBlur(img,(3,3),0) for img in imgs]
        
        
        # canny edge detection
       
        imgs = np.array([cv2.Canny(img,100,150) for img in imgs])
        
        
        imgs = imgs/127.5-1
       
        imgs = torch.from_numpy(imgs).unsqueeze(1)
        
        imgs = imgs.repeat(1,3,1,1)#.permute(0,3,1,2)
        
        return imgs

    def training_step(self, batch, batch_idx):


        
       
        meta = {
           
            "mask": batch['mask'],
           
        }
        
        images=batch['imgs']
        blur_images = batch['dark_imgs'] # input, blur image, condition mask
        bs, m, c,h, w = images.shape
        #images=rearrange(images, 'bs m h w c -> bs m c h w')
        #blur_images=rearrange(blur_images, 'bs m c h w -> bs m c h w')

        
        blur_images = blur_images[:,self.views]
        
        #blur_images[:,0] = self.canny_edge(blur_images[:,0])
       
    
        
        
        
        images = rearrange(images, 'bs m c h w -> (bs m) c h w')
        latents = self.encode_image(images, self.vae)
        latents = rearrange(latents, '(bs m) c h w -> bs m c h w',m=m)
       
        noise = torch.randn_like(latents)
        #latents[:,0] = blur_images
        noise[:,1:] = 0

        t = torch.randint(0, self.scheduler.num_train_timesteps,
                        (latents.shape[0],), device=latents.device).long()
        noise_z = self.scheduler.add_noise(latents, noise, t)

        
        blur_images= rearrange(blur_images, 'bs m c h w -> (bs m) c h w')
        
        blur_images = torch.nn.functional.interpolate(blur_images,
                                                 size = (128, 128),
                                                    mode='bilinear', align_corners=False)
        blur_images = rearrange(blur_images, '(bs m) c h w -> bs m c h w',m=m)
        
        #blur_images = rearrange(blur_images, '(bs m) c h w -> bs m c h w',m=m)


        
       
        
        

        

        # b,m,c,h,w
        # the input should be latents_input,
        latents_input = torch.cat([noise_z,blur_images], dim=2)
        latents_input = self.scheduler.scale_model_input(latents_input, t)
        denoise = self.forward(latents_input, torch.cat([t[:, None]]*m, dim=1))
        target = noise[:,0]
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
        dark_imgs = rearrange(dark_imgs, 'bs m c h w -> (bs m) c h w')
        dark_imgs = dark_imgs.permute(0,2,3,1).cpu().numpy()
        gt_imgs = (gt_imgs*255).round().astype('uint8')
        dark_imgs = (dark_imgs*255).round().astype('uint8')
        val_loss = np.mean(np.square(images_pred[:,0] - gt_imgs))

        sample ={}
        psnr = self.psnr(images_pred,gt_imgs)
        sample["psnr"] = psnr
    
        sample["gt_imgs"] = gt_imgs
        sample["dark_imgs"] = dark_imgs
        sample["condition"] = self.condition
        
        sample["pred_imgs"] = images_pred.reshape(-1,512,512,3)
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
      
       

        bs, m, c,h, w = images.shape
        device = images.device
        latents_shape = (bs,m, c, h//4, w//4)
        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps
        #images[:,0] = self.canny_edge(images[:,0])

        #images = rearrange(images, 'bs m c h w -> bs (m c) h w')
        encoded_images = self.encode_image(images[:,1:], self.vae)
        latents = torch.randn(latents_shape, device=device)
        latents[:,1:] = encoded_images

        images = rearrange(images, 'bs m c h w -> (bs m) c h w')
        images = torch.nn.functional.interpolate(images,
                                                 size = (128, 128),
                                                    mode='bilinear', align_corners=False)
        
        self.condition = images.cpu().numpy()/2+0.5
        images = rearrange(images, '(bs m) c h w -> bs m c h w',m=m)
        # images b,m,c,h,w
        


        

        for i, t in enumerate(timesteps):
          
            
            _timestep = torch.cat([t[None, None]]*m*bs, dim=1)
            latent_model_input = torch.cat([latents,images], dim=2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = self.forward(latent_model_input,t)
            
            latents[:,0] = self.scheduler.step(
                noise_pred, t, latents[:,0],eta = 1).prev_sample

        images_pred = self.decode_latent(
            latents, self.vae)
       
        return images_pred
if __name__=="__main__":
    import yaml 
    config = yaml.load(open("/root/autodl-tmp/project/configs/config_sr_ldm.yaml", "r"), Loader=yaml.FullLoader)
    model = PanoOutpaintGenerator(config).cuda()
    batch ={}
    path = "/root/autodl-tmp/project/ldm_generated_image.png"
    from PIL import Image
    img = Image.open(path)
    img = np.array(img)
    img = img/127.5-1
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda()
    img = img.unsqueeze(1).repeat(3,3,1,1,1)
    print(img.shape)
    


    batch["dark_imgs"] = img.float()
    batch["imgs"] = img.float()
    batch["mask"] = torch.ones_like(img).float()
    pred = model.training_step(batch,0)
    for k,v in pred.items():
        try:
            print(k,v.shape)
        except:
            print(k,v)

    