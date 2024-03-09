import pytorch_lightning as pl
from diffusers import VQModel,UNet2DModel,DDIMScheduler
import torch
import os
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer,CLIPVisionModel
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import torch
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers import DDIMScheduler
import matplotlib.pyplot as plt
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau
#from models.pano.MVGenModel import MultiViewBaseModel
from einops import rearrange
from torchvision import transforms as T
from torchvision.utils import save_image

import sys
import math

#from modules.unClip import ReprogrammingLayer


class PanoOutpaintGenerator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs'] if 'max_epochs' in config['train'] else 0
        self.diff_timestep = config['model']['diff_timestep']
        

        self.views = 3



        vae, scheduler, unet, text_encoder, tokenizer = self.load_model(
            config)
        self.vae = vae
        self.scheduler = scheduler
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.inferece_steps = config['model']['diff_timestep']
        self.begin_steps = config['model']['begin_steps']
        self.guidance_scale = config['model']['guidance_scale']

        

        


     
        

        lora_layers = filter(lambda p: p.requires_grad, self.unet.parameters())

       
        self.trainable_params = [(lora_layers, 1
        )]
      
       


        

        #self.mv_base_model = MultiViewBaseModel(
        #    unet, config['model'])
        #self.trainable_params = self.mv_base_model.trainable_parameters
        #self.trainable_params = [(list(self.text_encoder.parameters())+list(self.vae.parameters()),1)]
        
        #self.save_hyperparameters()
        
   
    def load_model(self, config):

        model_id = config["model"]['model_id']


        
        tokenizer = CLIPTokenizer.from_pretrained(model_id,subfolder= config["model"]['tokenizer_folder'])
        text_encoder = CLIPTextModel.from_pretrained(model_id,subfolder= config["model"]['text_folder'])
        vae = AutoencoderKL.from_pretrained(model_id,subfolder= config["model"]['vae_folder'])
        unet = UNet2DConditionModel.from_pretrained(model_id,subfolder= config["model"]['unet_folder'])
        scheduler = DDIMScheduler.from_pretrained(model_id,subfolder= config["model"]['scheduler_folder']) 
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)
        unet.enable_xformers_memory_efficient_attention()

        return vae, scheduler, unet, text_encoder, tokenizer


    

    
    


    @torch.no_grad()
    def encode_image(self,imgs,depths):
        unpreprocess = T.Compose([
            T.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            
        ])
       
        b,v,c,h,w = imgs.shape
        imgs = rearrange(imgs,"b v c h w -> b c h (v w)")
        depths = rearrange(depths,"b v h w -> b h (v w)")
        mi = depths.min()
        ma = depths.max()
        depths = (depths-mi)/(ma-mi+1e-8)
        depths = depths*2-1
        imgs = unpreprocess(imgs)
        
        imgs = imgs*2-1
        depths = depths.unsqueeze(1)
        
        imgs = torch.cat([imgs,depths],dim=1)
        imgs = imgs.to(self.vae.device)
        image_embeds = self.vae.tiled_encode(imgs)
        mean = image_embeds.latent_dist.mean
        std = image_embeds.latent_dist.std
        # reparameter
        alpha = torch.randn_like(mean)
        z = mean # + std*alpha
        
        return z

    @torch.no_grad()
    def decode_latent(self, latents):
        v = self.views
        b,c,h,vw = latents.shape
   
        image = self.vae.decode(latents #/vae.config.scaling_factor
                        ,
                        return_dict=False)[0]
        imgs = rearrange(image,"b c h (v w) -> b v c h w",v=v)
        image = imgs[:,:,:3]
        depth =  imgs[:,:,3]
        return image,depth

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


        
       
       

        images=batch['imgs']
        target_imgs = batch['target_imgs']
        depths = batch['depth']
       
        #images=rearrange(images, 'bs m h w c -> bs m c h w')
        #blur_images=rearrange(blur_images, 'bs m c h w -> bs m c h w')

       
        
        latents = self.encode_image(target_imgs, depths)
        
        input_latents = self.encode_image(images, depths)
        #input_latents[:,3:] = torch.randn_like(input_latents[:,3:])

        
       
        t = torch.randint(0,self.scheduler.num_train_timesteps,(latents.shape[0],), device=latents.device).long()
                       
        prompt_embedding = self.encode_prompt([""]*images.shape[0])

        noise = torch.randn_like(latents)
        
        noise_z = self.scheduler.add_noise(latents, noise, t)


        
        # b,m,c,h,w 
        # the input should be latents_input,
        # the target should be latents
        denoise = self.unet(noise_z,t, encoder_hidden_states=prompt_embedding.cuda())[0]

       

        # eps mode
        loss = torch.nn.functional.mse_loss(denoise, noise)
        print("loss is nan",loss.isnan())
        self.log("train_loss", loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        
        
        return loss
    
        

    def encode_prompt(self,prompt):
        text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        text_features = self.text_encoder(text_input_ids)
        
        text_features = text_features[0]
        
        return text_features

    def psnr(self,img1,img2):
        mse = np.mean((img1/255. - img2/255.) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images=batch['imgs']
        target_imgs = batch['target_imgs']
        depths = batch['depth']
       
        #images=rearrange(images, 'bs m h w c -> bs m c h w')
        #blur_images=rearrange(blur_images, 'bs m c h w -> bs m c h w')

       
        
        latents = self.encode_image(target_imgs, depths)
        
        input_latents = self.encode_image(images, depths)
        #input_latents[:,3:] = torch.randn_like(input_latents[:,3:])

        
       
        t = torch.randint(0,self.scheduler.config.num_train_timesteps
                          
                          ,(latents.shape[0],), device=latents.device).long()
                       
        prompt_embedding = self.encode_prompt([" "]*images.shape[0])

        noise = torch.randn_like(latents)
        
        noise_z = self.scheduler.add_noise(latents, noise, t)


        
        # b,m,c,h,w 
        # the input should be latents_input,
        # the target should be latents
        denoise = self.unet(noise_z,t, encoder_hidden_states=prompt_embedding.cuda())[0]

       

        # eps mode
        loss = torch.nn.functional.mse_loss(denoise, noise)

        self.log("loss is nan",loss.isnan(),on_step=True,prog_bar=True,logger=True)
        self.log("val_temp_loss", loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)

        if batch_idx%50 == 0:
            latents_input = self.encode_image(images, depths)
            latents_input[:,3:] = torch.randn_like(latents_input[:,3:])
            z = self.encode_image(images,torch.randn_like(depths))

            noise = torch.randn_like(z)

            
            self.scheduler.set_timesteps(self.inferece_steps, device="cuda")
            timesteps = self.scheduler.timesteps

            latents =self.scheduler.add_noise(z, noise, torch.tensor([self.begin_steps]).to(z.device))
            prompt_embedding = self.encode_prompt([""]*images.shape[0])

            from tqdm import tqdm
            for t in tqdm(timesteps[timesteps<self.begin_steps]):
                
                
                latents_input = self.scheduler.scale_model_input(latents_input, t)
                denoise_pred = self.unet(latents,t, encoder_hidden_states=prompt_embedding.cuda())[0]

                latents =self.scheduler.step(denoise_pred, t, latents, return_dict=False)[0]
            img,depth = self.decode_latent(latents/self.vae.config.scaling_factor)

            target_imgs = self.denormalize(target_imgs) # b,c,h,w
            depth = self.denormalize(depth) # b,c,h,w
            img = self.denormalize(img) # b,1,h,w
            images = self.denormalize(images) 
            depths = self.denormalize(depths) # b,1,h,w
            img_loss = torch.nn.functional.mse_loss(img, target_imgs)
            depth_loss = torch.nn.functional.mse_loss(depth, depths)
        
            stack1 = torch.stack([img[0],target_imgs[0],images[0]])
            stack2 = torch.stack([depth[0],depths[0]])
            path_dir = "/root/autodl-tmp/images/ldm3d/val"
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            epoch = self.current_epoch
            save_image(stack1, path_dir+f"/epoch_{epoch}_val_img_"+str(batch_idx)+".png",nrow=6)
            save_image(stack2, path_dir+f"/epoch_{epoch}_val_depth_"+str(batch_idx)+".png",nrow=6)

            self.log("val_loss", img_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
            self.log("val_depth_loss", depth_loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)


        
        
   
        
        # compute image & save
        #if self.trainer.global_rank == 0:
        #    self.save_image(images_pred, images,blur_images, batch['prompt'], batch_idx)

    @torch.no_grad()
    def denormalize(self, img):
        if img.shape[2] != 3:
            img = rearrange(img,"b v  h w -> b  h (v w)")
            mi = img.min()
            ma = img.max()
            img = (img-mi)/(ma-mi+1e-8)
            return img.unsqueeze(1)
        
        else:
            unpreprocess = T.Compose([
                T.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
                T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            ])
            img = rearrange(img,"b v c h w -> b c h (v w)")
            img = unpreprocess(img)
        return img
    @torch.no_grad()
    def inference(self, batch):
       
        images=batch['imgs']
        depths = batch['depth']
        z = self.encode_image(images,torch.randn_like(depths))

        noise = torch.randn_like(z)

        
        self.scheduler.set_timesteps(self.inferece_steps, device="cuda")
        timesteps = self.scheduler.timesteps

        latents =self.scheduler.add_noise(z, noise, torch.tensor([self.begin_steps]).to(z.device))
        prompt_embedding = self.encode_prompt([" "]*images.shape[0])

        from tqdm import tqdm
        for t in tqdm(timesteps[timesteps<self.begin_steps]):
            
            

            denoise_pred = self.unet(latents,t, encoder_hidden_states=prompt_embedding.cuda())[0]
            latents =self.scheduler.step(denoise_pred, t, latents, return_dict=False)[0]
        img,depth = self.decode_latent(latents)
        
        return img,depth
        
    