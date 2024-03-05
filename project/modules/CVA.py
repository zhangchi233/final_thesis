# conduct cost volume attention
# calculate feature b,m,c,h,w --> b,m,c,h,w,
# calculate cost volume b,m,c,h,w ---> b,m,d,c,h,w
# calculate cost homo warpped feature on each depth, b,1,d,c,h,w, and b,2,d,c,h,w
# calculate the cost variance for each volume
# 3d variance to 
# softmax
# sum of each feature on value and add that back to the feature layer of the image
# output original feature and recover the image
# send the image to the unet, the diffusion network
# the loss should be two parts
from einops import reduce, rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
#from inplace_abn import InPlaceABN
from kornia.utils import create_meshgrid
import sys

sys.path.append("/root/autodl-tmp/project")
from models.modules.resnet import BasicResNetBlock
from models.modules.transformer import BasicTransformerBlock, PosEmbedding



def homo_warp(src_feat, proj_mat, depth_values):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device

    R = proj_mat[:, :, :3] # (B, 3, 3)
    T = proj_mat[:, :, 3:] # (B, 3, 1)
    # create grid from the ref frame
    ref_grid = create_meshgrid(H, W, normalized_coordinates=False,
                               device=device) # (1, H, W, 2)
    ref_grid = rearrange(ref_grid, '1 h w c -> 1 c (h w)') # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1) # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:,:1])), 1) # (B, 3, H*W)
    ref_grid_d = repeat(ref_grid, 'b c x -> b c (d x)', d=D) # (B, 3, D*H*W)
    src_grid_d = R @ ref_grid_d + T/rearrange(depth_values, 'b d h w -> b 1 (d h w)')
    del ref_grid_d, ref_grid, proj_mat, R, T, depth_values # release (GPU) memory
    
    # project negative depth pixels to somewhere outside the image
    negative_depth_mask = src_grid_d[:, 2:] <= 1e-7
    src_grid_d[:, 0:1][negative_depth_mask] = W
    src_grid_d[:, 1:2][negative_depth_mask] = H
    src_grid_d[:, 2:3][negative_depth_mask] = 1

    src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:] # divide by depth (B, 2, D*H*W)
    del src_grid_d
    src_grid[:, 0] = src_grid[:, 0]/((W-1)/2) - 1 # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1]/((H-1)/2) - 1 # scale to -1~1
    src_grid = rearrange(src_grid, 'b c (d h w) -> b d (h w) c', d=D, h=H, w=W)

    warped_src_feat = F.grid_sample(src_feat, src_grid,
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True) # (B, C, D, H*W)
    warped_src_feat = rearrange(warped_src_feat, 'b c d (h w) -> b c d h w', h=H, w=W)
    src_grid = rearrange(src_grid, 'b d g c -> (b g) d c')

    return warped_src_feat,src_grid

class CVAtten(nn.Module):
    def __init__(self, dim, flag360=False,interval_ratio=1,n_depths=12,levels = 2):
        super().__init__()
        self.flag360 = flag360
        self.n_depths = n_depths
        self.interval_ratio = interval_ratio
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = PosEmbedding(2, dim//4)
        self.levels = levels
    def forward(self,imgs,proj_mats,init_depth_min,depth_interval,m):
        b, m, c, h, w = imgs.shape
        depth_interval_l = depth_interval * self.interval_ratio
        D = self.n_depths
        proj_mats = proj_mats[:,:,self.levels]

        if isinstance(init_depth_min, float):
            depth_values = init_depth_min + depth_interval_l * \
                                   torch.arange(0, D,
                                                device=imgs.device,
                                                dtype=imgs.dtype) # (D)
            depth_values = rearrange(depth_values, 'd -> 1 d 1 1')
            depth_values = repeat(depth_values, '1 d 1 1 -> b d h w', b=b, h=h, w=w)
        else:
            depth_values = init_depth_min.reshape(-1,1) + depth_interval_l * \
                                   (rearrange(torch.arange(0, D,
                                                           device=imgs.device,
                                                          dtype=imgs.dtype),
                                             'd -> 1 d').repeat(b,1)) # (B, D)
            depth_values = rearrange(depth_values, 'b d -> b d 1 1')
            depth_values = repeat(depth_values, 'b d 1 1 -> b d h w', h=h, w=w)
        key_values,key_value_xy = self.get_key_values(imgs,proj_mats,depth_values)
        key_pe = self.pe(key_value_xy)
        key_values+=key_pe
        query = imgs[:,[0]]
        query = rearrange(query,'b m c h w -> (b h w) m c')
        query_pe = self.pe(torch.zeros(query.shape[0],1,2,device = query.device))
        out = self.transformer(query, key_values, query_pe=query_pe)
        out = rearrange(out,'(b h w) m c -> b m c h w',h=h,w=w)
        output = torch.cat([out,imgs[:,1:]],dim=1)
        return output
        


    def get_key_values(self, feats, proj_mats, depth_values):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V-1, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]

        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = rearrange(src_feats, 'b vm1 c h w -> vm1 b c h w') # (V-1, B, C, h, w)
        proj_mats = rearrange(proj_mats, 'b vm1 x y -> vm1 b x y') # (V-1, B, 3, 4)

      
        #del ref_feats
        volumes = []
        key_grids = []
        for src_feat, proj_mat in zip(src_feats, proj_mats):
            warped_volume,key_grid = homo_warp(src_feat, proj_mat, depth_values) # (B, C, D, h, w)
            warped_volume = warped_volume.to(ref_feats.dtype)
            volumes.append(warped_volume)
            key_grids.append(key_grid)

            
            del src_feat, proj_mat
        del src_feats, proj_mats
        # aggregate multiple feature volumes by variance
        key_value_volume = torch.cat(volumes,dim=2) # b,c,2d,h,w
        key_value_volume = rearrange(key_value_volume,'b c d h w -> (b h w) d c')
        key_grids = torch.cat(key_grids,dim =1) # (b h w),2d,c

        
       
        return  key_value_volume, key_grids






if __name__=="__main__":
    import torch
    images = torch.randn(2,3,3,64,80)
    
    proj_mats = torch.randn(2,2,3,3,4)
    init_depth_min = torch.randn(2,1)
    depth_interval = torch.randint(1,10,(2,1))
    

    layer = CVAtten(dim = 320)
    img = torch.randn(2,3,320,64,80)*100
    output  = layer(img,proj_mats,init_depth_min,depth_interval,3)
    
    print(torch.equal(output[:,0],img[:,0]))
    
