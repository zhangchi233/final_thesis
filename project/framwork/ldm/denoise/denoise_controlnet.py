from torch import nn
from einops import rearrange
import sys
sys.path.append("/root/autodl-tmp/project")
from models.modules.utils import CPBlock

class CPBlock(CPBlock):
    def __init__(self, dim, flag360=False):
        super().__init__(dim,flag360)
        
    def forward(self,x,m):
        #x = self.attn1.forward(x, correspondences, img_h, img_w, R, K, m)
        x = self.attn1.forward(x, m)
        #x = self.attn2(x, correspondences, img_h, img_w, R, K, m)
        x = self.attn2.forward(x, m)
        x = self.resnet(x)
        return x
    
    
class denoiseModel(nn.Module):
    def __init__(self,unet):
        super(denoiseModel,self).__init__()
        self.unet = unet

        self.conv_in = self.unet.conv_in
        self.down_blocks = self.unet.down_blocks
        self.mid_blocks = self.unet.mid_block
        self.blocks = []
        self.zero_layers = []
        for down_block in self.down_blocks:
            self.blocks.append(
                CPBlock(
                    down_block.resnets[-1].out_channels, flag360=False)
                    )
            self.zero_layers.append(
                self.zero_module(
                    self.conv_nd(3,
                        down_block.resnets[-1].out_channels
                        ,down_block.resnets[-1].out_channels,3,padding=1
                                )
                                )
                                )
        self.zero_layers = nn.ModuleList(self.zero_layers)
        self.blocks = nn.ModuleList(self.blocks)
        self.condition_convin = self.conv_nd(3, 6, 3, 3, padding=1)

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
    def trainable_forward(self,latents,time_embdding):
        #build unet architecture 
        # latents shape == (batch_size, 3, 6, 128, 128)
        b,m,c,h,w = latents.shape
        latents = rearrange(latents,'b m c h w -> (b m) c h w')
        sample = self.conv_in(latents)
        down_block_res_samples = ()
        for i,block in enumerate(self.down_blocks):

            sample, _= block(hidden_states=sample, temb=time_embdding)
           
            

            sample_add = sample
            sample_add = self.blocks[i](sample_add,m)
            
            sample_add = rearrange(sample_add,'(b m) c h w -> b c m h w',m =m)
            sample_add = self.zero_layers[i](sample_add)
            down_block_res_samples += (sample_add,)
        return down_block_res_samples
        

        
        

    
    def forward(self, latents, timestep):

        b,m,c,h,w = latents.shape
        latents = rearrange(latents,'b m c h w -> (b m) c h w')
        
        sample = self.unet.conv_in(latents)
        sample = rearrange(sample,'(b m) c h w -> b m c h w',b=b,m=m)
        latents = rearrange(latents,'(b m) c h w -> b m c h w',b=b,m=m)
       

        time_embddings = self.unet.time_proj(timestep.flatten())
        time_embedding = self.unet.time_embedding(time_embddings)
        

        self.trainable_forward(latents,time_embedding)
       
        sample = sample[:,0]
        down_block_res_samples = (sample,)
        time_embedding = time_embedding[0::m]
        for block in self.unet.down_blocks:
            
            sample, res_sample = block(hidden_states=sample, temb=time_embedding)
            down_block_res_samples += res_sample
            

        sample = self.unet.mid_block(sample, time_embedding)
        skip_sample = None
        for upsample_block in self.unet.up_blocks:
            
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            
            sample = upsample_block(sample, res_samples, time_embedding)
        
        sample = self.unet.conv_norm_out(sample)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)

        

        if self.unet.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        

        return sample