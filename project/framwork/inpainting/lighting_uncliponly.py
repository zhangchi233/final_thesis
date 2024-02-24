import pytorch_lightning as pl
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
import torch
import os
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer,CLIPVisionModel
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
#from models.pano.MVGenModel import MultiViewBaseModel
from einops import rearrange
from torchvision import transforms as T
import open_clip 
import sys
import math
sys.path.append("/root/autodl-tmp/project")
from data.dtu import DTUDataset
from modules.unClip import ReprogrammingLayer
sys.path.append("/root/autodl-tmp/project/framwork/inpainting")
from denoise.denoise import denoiseModel
class PanoOutpaintGenerator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs'] if 'max_epochs' in config['train'] else 0
        self.diff_timestep = config['model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']



        self.tokenizer = CLIPTokenizer.from_pretrained(
            #"/root/autodl-tmp/inpaiting_model/tokenizer"
            config['model']['model_id'], subfolder="tokenizer"
            )
        self.text_encoder = CLIPTextModel.from_pretrained(
            #"/root/autodl-tmp/text_encoder"
            config['model']['model_id'], subfolder="text_encoder"
            )


        self.vae, self.scheduler, unet = self.load_model(
            config['model']['model_id'])


        preprocess = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
                        
        ])
        imgEncoder = open_clip.create_model('ViT-H-14', 'laion2b_s32b_b79k')
        self.imgEncoder = imgEncoder
        self.preprocess = preprocess
        self.word_embeddings = self.text_encoder.get_input_embeddings().weight 
        self.map_layer = torch.nn.Linear(self.word_embeddings.shape[0], 1000)
        self.image_reprogramming = ReprogrammingLayer(d_model=1024, n_heads=4, d_keys=1024, d_llm=1024, attention_dropout=0.1)
        self.text_reprogramming = ReprogrammingLayer(d_model=1024, n_heads=4, d_keys=1024, d_llm=1024, attention_dropout=0.1)
        

        self.trainable_params = [
            (list(self.image_reprogramming.parameters())+\
             list(self.map_layer.parameters())+\
             list(self.text_reprogramming.parameters()),1)]

        self.model = denoiseModel(unet)
      
       


        

        #self.mv_base_model = MultiViewBaseModel(
        #    unet, config['model'])
        #self.trainable_params = self.mv_base_model.trainable_parameters
        #self.trainable_params = [(list(self.text_encoder.parameters())+list(self.vae.parameters()),1)]
        
        #self.save_hyperparameters()
        
    def forward(self, latents, timestep, prompt_embd):
        return self.model(latents, timestep, prompt_embd)
    def load_model(self, model_id):
        vae = AutoencoderKL.from_pretrained(
            #"/root/autodl-tmp/inpaiting_model/vae"
            model_id, subfolder="vae",
            )
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(
            #"/root/autodl-tmp/inpaiting_model/scheduler")
            model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(
            #"/root/autodl-tmp/inpaiting_model/unet"
            model_id, subfolder="unet"
            )
        return vae, scheduler, unet

    

    @torch.no_grad()
    def clipEncoder(self,images):
        encodings = []
        for image in images:
            image_processed = self.preprocess(image)
            encoding = self.imgEncoder.encode_image(image_processed)
            encodings.append(encoding)
        return torch.stack(encodings, dim=0)

    def img2prompt(self,images):
        B,L,C,H,W = images.shape
        images = self.clipEncoder(images)
        ref_imgs = images[:,[0]]
        source_imgs = images[:,1:]
        


        repro_imgs = self.image_reprogramming(ref_imgs, source_imgs, source_imgs)
        repro_imgs = repro_imgs.repeat(1,self.tokenizer.model_max_length,1)
        word_embeddings = self.word_embeddings.repeat(B,1,1)
        word_embeddings = self.map_layer(word_embeddings.permute(0,2,1)).permute(0,2,1)
        prompt = self.text_reprogramming(repro_imgs, word_embeddings, word_embeddings)
        return prompt
    


    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]
        x_input = x_input.float().to(vae.dtype)
        x_input = x_input.to(vae.device)

        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64) # z vector mean and variance

        z = z.sample() # sample

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        z = z.float()
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)

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
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
            'interval': 'epoch',  # update the learning rate after each epoch
            'name': 'cosine_annealing_lr',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width
    ):
    
        mask = torch.nn.functional.interpolate(
            mask, size=(height // 8, width // 8)
        )
        masked_image_latents = self.encode_image(masked_image, self.vae)
        
        return mask, masked_image_latents
    def prepare_mask_image(self, images,meta_mask):
        bs, m, _, h, w = images.shape
        mask=torch.zeros(bs, m, 1, h, w, device=images.device)
        mask[:,0]=meta_mask.unsqueeze(1)
        masked_image=images*(mask<0.5)
        #mask[:,0] = torch.ones_like(mask[:,0]).to(mask.device).float()
        mask[:,1:] = torch.zeros_like(mask[:,1:]).to(mask.device).float()
        self.masked_image = rearrange(masked_image, 'bs m c h w -> bs c h (m w)')
        mask_latnets=[]
        masked_image_latents=[]
        self.mask = rearrange(mask, 'bs m c h w -> bs c h (m w)')

        
        for i in range(1  #m
                       ):
            _mask, _masked_image_latent = self.prepare_mask_latents(
                mask[:,i],
                masked_image[:,i],
                bs,
                h,
                w,
            )
            mask_latnets.append(_mask)
            masked_image_latents.append(_masked_image_latent)
        mask_latnets = torch.stack(mask_latnets, dim=1)
        masked_image_latents = torch.stack(masked_image_latents, dim=1)
        return mask_latnets, masked_image_latents
    def training_step(self, batch, batch_idx):


        
       
        meta = {
           
            "mask": batch['mask'],
           
        }

        images=batch['imgs']
        blur_images = batch['dark_imgs'] # input, blur image, condition mask
       
        #images=rearrange(images, 'bs m h w c -> bs m c h w')
        #blur_images=rearrange(blur_images, 'bs m c h w -> bs m c h w')

        prompt_embds = self.img2prompt(blur_images)

        mask_latnets, masked_image_latents=self.prepare_mask_image(blur_images,meta["mask"]) 
        # prepare mask via the depth difference 
        
        
        
        images = images[:,0]
        
        blur_images=blur_images[:,0]

        mask_latnets=mask_latnets[:,0]
        masked_image_latents=masked_image_latents[:,0]
        #depths = batch['depths']

        latents=self.encode_image(images, self.vae)
       
        t = torch.randint(0, self.scheduler.num_train_timesteps,
                        (latents.shape[0],), device=latents.device).long()
        

        noise = torch.randn_like(latents)
        noise_z = self.scheduler.add_noise(latents, noise, t)
        
        # b,m,c,h,w 
        # the input should be latents_input,
        latents_input = torch.cat([noise_z, mask_latnets, masked_image_latents], dim=1)
        denoise = self.forward(latents_input, t, prompt_embds)
        target = noise
        target = target.to(torch.float32)
        denoise = denoise.to(torch.float32)
       

        # eps mode
        loss = torch.nn.functional.mse_loss(denoise, target)
        self.log("train_loss", loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        
        
        return loss
    
        

    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch):
        latents = torch.cat([latents]) # one conditional, one unconditional
        timestep = torch.cat([timestep]) # one conditional, one unconditional
        

        masks = torch.cat([batch["mask"]])
        meta = {
            
            "mask": masks.float()#.to(self.device)
            ,
            
        }

        return latents, timestep, prompt_embd, meta

    @torch.no_grad()
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model):
        latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latents_high_res, _timestep, prompt_embd, batch)

        noise_pred = model(
            latents, _timestep, _prompt_embd, meta)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred


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
        dark_imgs = batch["dark_imgs"]
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = batch['dark_imgs'].to(device)
        prompt_embds = self.img2prompt(images)
        mask = batch["mask"]

        bs, m, _,h, w = images.shape
        
        mask_latnets, masked_image_latents=self.prepare_mask_image(images,mask)
        
        device = images.device

        latents= torch.randn(
            bs, 4, h//8, w//8, device=device)
        
        prompt_embds = self.img2prompt(images)

        images = images[:,0]
        mask_latnets=mask_latnets[:,0]
        masked_image_latents=masked_image_latents[:,0]


       
        
      
        
     
       

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        
        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]], dim=1)
            latent_model_input = torch.cat([latents, mask_latnets, masked_image_latents], dim=1)

            noise_pred = self.forward(latent_model_input, _timestep, prompt_embds)
            
            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample

        images_pred = self.decode_latent(
            latents, self.vae)
       
        return images_pred
    
   