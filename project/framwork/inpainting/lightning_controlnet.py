import pytorch_lightning as pl
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
import torch
import os
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.append("/root/autodl-tmp/project/framwork/inpainting")
from denoise.denoise_controlnet import MultiViewBaseModel
from einops import rearrange
# import scaler

class PanoOutpaintGenerator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs'] if 'max_epochs' in config['train'] else 0
        self.diff_timestep = config['model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']

        self.tokenizer = CLIPTokenizer.from_pretrained(
            config['model']['model_id'], subfolder="tokenizer", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(
            config['model']['model_id'], subfolder="text_encoder", torch_dtype=torch.float16)

        self.vae, self.scheduler, unet = self.load_model(
            config['model']['model_id'])
        self.mv_base_model = MultiViewBaseModel(
            unet)
        self.trainable_params = self.mv_base_model.trainable_parameters

        self.save_hyperparameters()
       
    def load_model(self, model_id):
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae",
            #torch_dtype=torch.float16
            )
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet",
            #torch_dtype=torch.float16
            )
        return vae, scheduler, unet

    @torch.no_grad()
    def encode_text(self, text, device):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask)

        return prompt_embeds[0].float(), prompt_embeds[1]

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]

        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        z = z.float()
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)

        images = []
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
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
        # define scaler for automatic mixed precision


        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width
    ):
    
        mask = torch.nn.functional.interpolate(
            mask, size=(height // 8, width // 8)
        )
        masked_image_latents = self.encode_image(masked_image, self.vae)
        
        return mask, masked_image_latents

    def training_step(self, batch, batch_idx):

        meta = {
            "mask": batch['mask'],
            "proj_mats": batch['proj_mats'],
            "init_depth_min": batch['init_depth_min'],
            'depth_interval': batch['depth_interval'],
                }
        device = batch['imgs'].device
        images=batch['imgs']
        dark_images=batch['dark_imgs']

       
        mask_latnets, masked_image_latents=self.prepare_mask_image(dark_images,meta)
        
        prompt_embds = []
        for prompt in range(3):
            prompt_embds.append(self.encode_text(
                "", device)[0])
        m=images.shape[1]
        dark_images=rearrange(dark_images, 'bs m c h w -> (bs m) c h w')
        latents=self.encode_image(dark_images, self.vae)
        latents=rearrange(latents, '(bs m) c h w -> bs m c h w', m=m)
        t = torch.randint(0, self.scheduler.num_train_timesteps,
                        (latents.shape[0],), device=latents.device).long()
        prompt_embds = torch.stack(prompt_embds, dim=1).repeat(latents.shape[0], 1, 1, 1)

        noise = torch.randn_like(latents)
        noise[:,1:] = 0
        noise_z = self.scheduler.add_noise(latents, noise, t)
        t = t[:, None].repeat(1, latents.shape[1])

        latents_input = torch.cat([noise_z, mask_latnets, masked_image_latents], dim=2)
        denoise = self.mv_base_model(
            latents_input, t, prompt_embds, meta)
        target = noise[:, 0]
       

        # eps mode
        loss = torch.nn.functional.mse_loss(denoise, target)
        self.log('train_loss', loss)
        return loss

    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch):
        latents = torch.cat([latents])
        timestep = torch.cat([timestep])
        
    
      
        meta = {
            "mask": batch['mask'],
            'init_depth_min': batch['init_depth_min'],
            'depth_interval': batch['depth_interval'],
            "proj_mats": batch['proj_mats'],
        }

        return latents, timestep, prompt_embd, meta

    @torch.no_grad()
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model):
        latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latents_high_res, _timestep, prompt_embd, batch)

        noise_pred = model(
            latents, _timestep, _prompt_embd, meta)

        #noise_pred_uncond = noise_pred#.chunk(2)
        #noise_pred_text = noise_pred#.chunk(2)
        #noise_pred = noise_pred_uncond #+ self.guidance_scale * \
            #(noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred = self.inference(batch)
        images = batch['imgs']
        images = rearrange(images, 'bs m c h w -> (bs m) c h w')
        images = ((images.permute(0, 2, 3, 1)/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)
      
        # compute image & save
        print(images_pred.shape)
        dark_imgs = (batch["dark_imgs"][:,0]/2+0.5).permute(0,2,3,1).cpu().numpy()
        val_loss = np.mean(np.square(images_pred[0,0] - images[0,0]))

        sample={}
        sample['gt_imgs'] = images
        sample['pred_imgs'] = images_pred[:,0]
        self.log('val_loss', val_loss)
        sample["dark_imgs"] = dark_imgs
        sample["mask"] = batch["mask"]

        return sample
    
    def prepare_mask_image(self, images,meta):
        bs, m, _, h, w = images.shape
        mask=torch.ones(bs, m, 1, h, w, device=images.device)
        mask[:,0,0]=meta['mask']
        masked_image=images*(mask<0.5)
        mask_latnets=[]
        masked_image_latents=[]
        for i in range(m):
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

    @torch.no_grad()
    def inference(self, batch):
        images = batch['dark_imgs']
        
        device = images.device
        bs, m, c, h, w = images.shape
        mask_latnets, masked_image_latents=self.prepare_mask_image(images,batch)
        images=rearrange(images, 'bs m c h w -> (bs m) c h w')
        latents=self.encode_image(images, self.vae)
        latents=rearrange(latents, '(bs m) c h w -> bs m c h w', m=m)
        latents[:,0]= torch.randn(
            bs, 4, h//8, w//8, device=device)

      
        
        

        

        #prompt_embds = []
        #for prompt in range(3):
        #    prompt_embds.append(self.encode_text(
        #        "", device)[0])
        #prompt_embds = torch.stack(prompt_embds, dim=1)
        
        prompt_null = self.encode_text('', device)[0]
        prompt_embd = prompt_null[:, None].repeat(latents.shape[0], m, 1, 1)
        #torch.cat(
        #    [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        
        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]]*m, dim=1).repeat(latents.shape[0], 1)
            latent_model_input = torch.cat([latents, mask_latnets, masked_image_latents], dim=2)

            noise_pred = self.forward_cls_free(
                latent_model_input, _timestep, prompt_embd, batch, self.mv_base_model)
            
            latents[:,0] = self.scheduler.step(
                noise_pred, t, latents[:,0]).prev_sample

        images_pred = self.decode_latent(
            latents, self.vae)
       
        return images_pred
    
    
