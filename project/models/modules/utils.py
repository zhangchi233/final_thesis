import os
os.chdir("/root/autodl-tmp")
print(os.getcwd())
import sys
sys.path.append("/root/autodl-tmp/project")
from modules.CVA import CVAtten
import numpy as np
from PIL import Image
import torch

#from src.dataset import MP3Ddataset, Scannetdataset
import pytorch_lightning as pl
from einops import rearrange
import torch.nn.functional as F

import torch.nn as nn
import torchvision.transforms as T


from models.modules.resnet import BasicResNetBlock
from models.modules.transformer import BasicTransformerBlock, PosEmbedding
#from src.models.pano.utils import get_query_value


class CPBlock(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.attn1 = CPAttn(dim, flag360=flag360)
        self.attn2 = CPAttn(dim, flag360=flag360)
        self.resnet = BasicResNetBlock(dim, dim, zero_init=True)

    def forward(self, x,proj_mats,init_depth_min,depth_interval,m):
        #x = self.attn1.forward(x, correspondences, img_h, img_w, R, K, m)
        x = self.attn1.forward(x, m)
        #x = self.attn2(x, correspondences, img_h, img_w, R, K, m)
        x = self.attn2.forward(x, m)
        x = self.resnet(x)
        return x
class CPBlock2(nn.Module):
    def __init__(self, dim, flag360=False,levels = 2):
        super().__init__()
        self.attn1 = CVAtten(dim, flag360=flag360,levels = levels)
        #self.attn2 = CPAttn(dim, flag360=flag360)
        self.attn3 = CVAtten(dim, flag360=flag360,levels = levels)
        
        self.resnet = BasicResNetBlock(dim, dim, zero_init=True)

    def forward(self, x, 
                #correspondences, 
                
                proj_mats,init_depth_min,depth_interval,m):
        #x = self.attn1.forward(x, correspondences, img_h, img_w, R, K, m)
        x = rearrange(x, '(b m) c h w -> b m c h w',m = m)
        x = self.attn1.forward(x, proj_mats, init_depth_min, depth_interval,m)
        #x = self.attn2(x, correspondences, img_h, img_w, R, K, m)
        #x = self.attn2._forward(x,  m)
       
        x = self.attn3(x, proj_mats, init_depth_min, depth_interval,m)
        x = rearrange(x, 'b m c h w -> (b m) c h w',m = m)
        x = self.resnet(x)
        return x


class CPAttn(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.flag360 = flag360
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = PosEmbedding(2, dim//4)
    def forward(self, x, m):
        # not use multi view correspondences
        b, c, h, w = x.shape
        x = rearrange(x, '(b m) c h w -> b m c h w', m=m)
        query = x[:,[0]]
        query = rearrange(query, 'b m c h w -> b (h w m) c')
        values = x[:,1:]
        values = rearrange(values, 'b m c h w -> b (h w m) c')
        query_pe = self.pe(torch.zeros(
        query.shape[0], 1, 2, device=query.device))

        out = self.transformer(query, values, query_pe=query_pe)
        out = rearrange(out, 'b (h w m) c -> b m c h w', h=h, w=w)
        values = rearrange(values, 'b (h w m) c -> b m c h w', h=h, w=w)
        out = torch.cat([out,values],dim = 1)

        

        out = rearrange(out, 'b m c h w -> (b m) c h w')

        #out = rearrange(out, 'b m c h w -> b (m c h) w')
        return out
        #print("query shape is:",query.shape,values.shape)

    
if __name__=="__main__":
    print("hello debuger,now start debugging")
    layer = CPBlock2(320)
    layer2 = CPBlock(320)
    images = torch.randn(6,320,64,64)
    proj_mats = torch.randn(2,2,3,3,4)
    init_depth_min = torch.randn(2,1)
    depth_interval = torch.randint(1,10,(2,1))
    out = layer(images,proj_mats,init_depth_min,depth_interval,3)
    print(out.shape)
    x = torch.randn(6,320,64,64)
    print(layer2(x,3).shape)
