import torch
from torch import nn
class denoiseModel(nn.Module):
    def __init__(self,unet):
        super(denoiseModel,self).__init__()
        self.unet = unet
        self.conv_in = self.unet.conv_in
        self.down_blocks = self.unet.down_blocks
        self.mid_blocks = self.unet.mid_block
        self.zero_layers = []
        self.zero_layers.append(self.zero_module(
            self.conv_nd(2,
                self.unet.conv_in.out_channels
                ,self.unet.conv_in.out_channels,3,padding=1)
                        
        ))
        self.condition_convin = self.conv_nd(2, 3,self.unet.conv_in.out_channels, 3, padding=1)

        
        for block in self.down_blocks:
            input_channels = block.resnets[-1].out_channels
            for i in range(2):
                    self.zero_layers.append(
                    self.zero_module(
                        self.conv_nd(2,
                            input_channels
                            ,input_channels,3,padding=1)
                                    ))

            if not hasattr(block, "attentions"):
                self.zero_layers.append(
                    self.zero_module(
                        self.conv_nd(2,
                            input_channels
                            ,input_channels,3,padding=1)
                                    ))
        
        self.zero_layers = nn.ModuleList(self.zero_layers)
        
                                             
            

    def zero_module(self,module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()
        return module
    def conv_nd(self,dims, *args, **kwargs):
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
    def trainable_forward(self,sample,time_embdding):
        #build unet architecture 
        # latents shape == (batch_size, 3, 6, 128, 128)
        b,c,h,w = sample.shape

        
        down_block_res_samples = (self.zero_layers[0](sample),)

        
        for i,block in enumerate(self.down_blocks):
            
            sample, residuals= block(hidden_states=sample, temb=time_embdding)
            length = len(down_block_res_samples)
            
            for i,res in enumerate(residuals):
                
                res = self.zero_layers[length + i](res)
                down_block_res_samples += (res,)        
            
            

           
            
            
           
        return down_block_res_samples
        
    
    def forward(self, latents, timestep,condition_latents = None):

        b,c,h,w = latents.shape
        
        
        sample = self.unet.conv_in(latents)

        time_embddings = self.unet.time_proj(timestep.flatten())
        time_embedding = self.unet.time_embedding(time_embddings)

        condition_latents = self.condition_convin(condition_latents)
        condition_latents += sample
        added_res = self.trainable_forward(condition_latents,time_embedding)
       
        sample = sample
        down_block_res_samples = (sample,)
      
        for block in self.unet.down_blocks:
            
            sample, res_sample = block(hidden_states=sample, temb=time_embedding)
            down_block_res_samples += res_sample
            

        sample = self.unet.mid_block(sample, time_embedding)
        skip_sample = None
        
        ziped_res = zip(down_block_res_samples,added_res)
        down_samples_residual = []
        for res,added in ziped_res:
            down_samples_residual.append(res+added)

        for upsample_block in self.unet.up_blocks:
            
            res_samples = down_samples_residual[-len(upsample_block.resnets) :]
            down_samples_residual = down_samples_residual[: -len(upsample_block.resnets)]

            
            sample = upsample_block(sample, res_samples, time_embedding)
        
        sample = self.unet.conv_norm_out(sample)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)

        

        if self.unet.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        

        return sample
if __name__ == "__main__":
    from diffusers import VQModel,UNet2DModel,DDIMScheduler
    unet = UNet2DModel.from_pretrained(
            #"/root/autodl-tmp/inpaiting_model/unet"
            "CompVis/ldm-super-resolution-4x-openimages", subfolder="unet"
            )
    model = denoiseModel(unet)
    noise_pred = model.forward(torch.rand(1,6,128,128),torch.randint(0,10,(1,)),torch.rand(1,3,128,128))
    print(noise_pred.shape)