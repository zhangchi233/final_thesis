from calendar import c
from os import times
import time
import torch
import torch.nn as nn
import sys
sys.path.append("/root/autodl-tmp/project")
from models.modules.utils import CPBlock,CPBlock2
from einops import rearrange
import copy

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
class MultiViewBaseModel(nn.Module):
    def __init__(self, unet):
        super().__init__()

        self.unet = unet
        self.single_image_ft = False
        self.unet.train()
        
        self.overlap_filter=0.1
       
        if self.single_image_ft:
            self.trainable_parameters = [(self.unet.parameters(), 0.01)]
        else:
            self.cp_blocks_encoder = nn.ModuleList()
            for i in range(len(self.unet.down_blocks)):
                if i == 0:
                    self.cp_blocks_encoder.append(
                        #CPAttn(
                        CPBlock(
                        self.unet.down_blocks[i].resnets[-1].out_channels, flag360=False))
                else:
                    self.cp_blocks_encoder.append(
                        #CPAttn(
                        CPBlock(
                        self.unet.down_blocks[i].resnets[-1].out_channels, flag360=False))

            self.cp_blocks_mid =  \
                CPBlock(  #CPAttn( \  
                self.unet.mid_block.resnets[-1].out_channels, flag360=False)

                    
            #for i, downsample_block in enumerate(self.unet.down_blocks):
            #    if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
            #        training_parameters+=list(downsample_block.resnets.parameters())
            #        training_parameters+=list(downsample_block.attentions.parameters())
                    
            #    else:
            #        if i<2:
            #            training_parameters+=list(downsample_block.resnets.parameters())
            
           
            
           

            self.zero_layers = []
                #conv_channels = self.unet.conv_in.in_channels
            channels = self.unet.conv_in.out_channels
            self.zero_layers.append(zero_module(conv_nd(2,channels,channels,3,padding=1)))
            for i, downsample_block in enumerate(self.unet.down_blocks):
                channels = downsample_block.resnets[-1].out_channels
                self.zero_layers.append(zero_module(conv_nd(2,channels,channels,3,padding=1)))

            mid_channels =  self.unet.mid_block.resnets[-1].out_channels

            self.zero_layers.append(zero_module(conv_nd(2,mid_channels,mid_channels,3,padding=1)))
            self.zero_layers = nn.ModuleList(self.zero_layers)#.cuda(1)
                
            
            self.cp_blocks = nn.ModuleList()
            self.cp_blocks.append(self.cp_blocks_encoder)
            self.cp_blocks.append(self.cp_blocks_mid)
            self.conv_in = self.unet.conv_in
            self.trainable_parameters = [
                (   list(self.conv_in.parameters())+
                    list(self.cp_blocks.parameters()) +\
                    list(self.zero_layers.parameters()),0.01)]
            #self.trainable_parameters+=training_parameters 
    def build_zeroconv(self,channels_in,channels_out,dim,num):
        layer = zero_module(conv_nd(dim,channels_in,channels_out,num,paddint = 1))
        return layer
    
    def forward_trainable(self, latents, timestep, prompt_embd,
                           meta,trainable_res,m,
                           #correspondences
                           ):
        count = 1
       
        proj_mats,init_depth_min,depth_interval = meta['proj_mats'],meta['init_depth_min'],meta['depth_interval']

        
        target_dtype = self.unet.conv_in.weight.dtype
        #latents = latents.to(target_dtype)
        #prompt_embd = prompt_embd.to(target_dtype)
        #timestep = timestep.to(target_dtype)
        #correspondences = correspondences.to(target_device)
    



        for i,downsample_block in enumerate(self.unet.down_blocks):
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for resnet, attn in zip(downsample_block.resnets, downsample_block.attentions):
                    latents = resnet(latents, timestep)

                    latents = attn(
                        latents, encoder_hidden_states=prompt_embd
                    ).sample
                    
                    #zero_conv = self.zero_layers[count+1]
                    #trainable_res.append(zero_conv(latents))
                    
            else:
                for resnet in downsample_block.resnets:
                    latents = resnet(latents, timestep)
                    
            if m > 1:
                latents = self.cp_blocks[0][i](
                   latents, 
                   #correspondences, 
                   proj_mats,init_depth_min,depth_interval,m, 
                   #meta
                   )
            trainable_res.append(self.zero_layers[count](latents[0::m])#.to(self.unet.dtype)
                                 )
            count+=1
            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    
                    latents = downsample(latents)# divide by 2
                
                    
                
            
        if m > 1:
            latents = latents#.to(self.unet.dtype)
            lstents = self.cp_blocks[1](
            latents,
             # correspondences,
              proj_mats,init_depth_min,depth_interval,m 
            #meta
            )
        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            latents = latents#.to(self.unet.dtype)
            latents = attn(
            latents, encoder_hidden_states=prompt_embd).sample
            latents = resnet(latents, timestep)
        zero_conv = self.zero_layers[count]
        trainable_res.append(zero_conv(latents[0::m])#.to(self.unet.dtype)
                             )
        return trainable_res
   
    def forward(self, latents, timestep, prompt_embd, meta):
        
        latents = latents#.to(self.unet.dtype)
       
        proj_mats,init_depth_min,depth_interval = meta['proj_mats'],meta['init_depth_min'],meta['depth_interval']
        
        
        b, m, c, h, w = latents.shape
       



        #correspondences = get_correspondences(meta, img_h, img_w)

        # bs*m, 4, 64, 64
        hidden_states = rearrange(latents, 'b m c h w -> (b m) c h w')

        prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')

        # 1. process timesteps

        timestep = timestep.reshape(-1)
        t_emb = self.unet.time_proj(timestep)  # (bs, 320)
        #t_emb = t_emb.to(self.unet.dtype).to(self.unet.device)
        emb = self.unet.time_embedding(t_emb)  # (bs, 1280)
        #hidden_states = hidden_states.to(self.unet.dtype)
        
        
        ### 
       
        
        hidden_trainable =  self.unet.conv_in(hidden_states#.to(self.unet.conv_in.weight.dtype)
                                              )
        
        zero_layer = self.zero_layers[0]
        hidden_trainable = zero_layer(hidden_trainable#.to(zero_layer.weight.device)
                                      )

       
            
        hidden_states = self.unet.conv_in(
            hidden_states[0::m]#.to(self.unet.device)
            )  # bs*m, 320, 64, 64
        hidden_trainable = hidden_states.repeat(m,1,1,1)+hidden_trainable#.to(hidden_states.dtype)



        # unet
        # a. downsample
        #prompt_embd = prompt_embd.to(self.unet.dtype)
        #prompt_embd = prompt_embd.to(self.unet.device)

        down_block_res_samples = (hidden_states,)
        training_add_res = []
        training_add_res.append(hidden_trainable#.to(self.unet.dtype)
                                )

        training_add_res = self.forward_trainable(hidden_trainable, emb, prompt_embd,meta,
                                                  training_add_res,m,
                                                  #correspondences
                                                  )
        prompt_embd = prompt_embd[0::m]
        emb = emb[0::m]

        for i, downsample_block in enumerate(self.unet.down_blocks):
            

            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for resnet, attn in zip(downsample_block.resnets, downsample_block.attentions):
                    hidden_states = resnet(hidden_states, emb)
                    
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample

                    down_block_res_samples += (hidden_states,)
            else:
                for resnet in downsample_block.resnets:
                    hidden_states = resnet(hidden_states, emb)
                    down_block_res_samples += (hidden_states,)
            # if m > 1:

            #     hidden_states = hidden_states.to(torch.float32)
            #     hidden_states = self.cp_blocks[0][i](
            #         hidden_states,
            #         # correspondences,
            #         proj_mats,init_depth_min,depth_interval, m,
            #         #meta
            #         )
            #     hidden_states = hidden_states.to(self.unet.dtype)

            if downsample_block.downsamplers is not None:

                for downsample in downsample_block.downsamplers:
                    hidden_states = hidden_states#.to(self.unet.dtype)
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)

        # b. mid
        hidden_states = hidden_states#.to(self.unet.dtype)
        hidden_states = self.unet.mid_block.resnets[0](
            hidden_states, emb)

        # if m > 1:
        #     hidden_states = hidden_states.to(torch.float32)
           
        #     hidden_states = self.cp_blocks[1](
        #        hidden_states, 
        #        #correspondences, 
        #        proj_mats,init_depth_min,depth_interval, m,
        #        #meta
        #        )
        #     hidden_states = hidden_states.to(self.unet.dtype)

        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            hidden_states = hidden_states#.to(self.unet.dtype)
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embd
            ).sample
            hidden_states = resnet(hidden_states, emb)
        hidden_trainable = training_add_res[-1]
        training_add_res = training_add_res[:-1]
        hidden_states+=hidden_trainable

        h, w = hidden_states.shape[-2:]

        # c. upsample
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]

            hidden_states = hidden_states#.to(self.unet.dtype)
            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                hidden_states = hidden_states#.to(self.unet.dtype)
                for resnet, attn in zip(upsample_block.resnets, upsample_block.attentions):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample
                
            else:
                for resnet in upsample_block.resnets:
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
                
            
            # if m > 1:
            #     hidden_states = hidden_states.to(self.unet.dtype)
                
            #     hidden_states = self.cp_blocks[2][i](
            #         hidden_states,
            #         # correspondences,
            #         proj_mats,init_depth_min,depth_interval, m,
            #         #meta
            #         )
            #     hidden_states = hidden_states.to(self.unet.dtype)
            if upsample_block.upsamplers is not None:
                hidden_trainable = training_add_res[-1]
            
                training_add_res = training_add_res[:-1]
                hidden_states+=hidden_trainable
                
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)
            

                
                

        # 4.post-process
        hidden_states = hidden_states#.to(self.unet.dtype)
        hidden_states += training_add_res[-1]
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        #hidden_states += training_add_res[-2]
        sample = self.unet.conv_out(sample)
        #sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)
        return sample


