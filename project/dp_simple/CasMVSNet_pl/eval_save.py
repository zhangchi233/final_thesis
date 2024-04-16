# %% [markdown]
# # refine model

# %%
from math import sqrt


import pytorch_lightning as pl
import sys
sys.path.append('/openbayes/input/input0/D3Dnet/code')
import functools
import torch.nn.functional as F
import torch
from collections import namedtuple
from torchvision import models
import torch.nn as nn
import sys
from einops import rearrange
from torchvision import models
import sys


# %%
sys.path.append('/openbayes/input/input0/project/dp_simple/')
#import ViT
from torchvision import transforms as T
from CasMVSNet_pl.models.mvsnet import CascadeMVSNet
from CasMVSNet_pl.utils import load_ckpt
from CasMVSNet_pl.datasets.dtu import DTUDataset  
from CasMVSNet_pl.utils import *
from CasMVSNet_pl.datasets.dtu import DTUDataset 
from CasMVSNet_pl.metrics import *  
from inplace_abn import ABN


import pytorch_lightning as pl


# %%
from datasets import dataset_dict
from datasets.utils import save_pfm, read_pfm
import cv2
import torch
import os, shutil
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

# for depth prediction
from models.mvsnet import CascadeMVSNet
from utils import load_ckpt
from inplace_abn import ABN

# for point cloud fusion
from numba import jit
from plyfile import PlyData, PlyElement

torch.backends.cudnn.benchmark = True # this increases inference speed a little

# %%
torch.backends.cudnn.benchmark = True # this increases inference speed a little

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/openbayes/input/input0/mvs_training/dtu',
                        help='root directory of dtu dataset')
    parser.add_argument('--dataset_name', type=str, default='dtu',
                        choices=['dtu', 'tanks', 'blendedmvs'],
                        help='which dataset to train/val')
    parser.add_argument('--split', type=str, default='train',
                        help='which split to evaluate')
    parser.add_argument('--scan', type=str, default='scan7',
                        help='specify scan to evaluate (must be in the split)')
    parser.add_argument('--cpu', default=False, action='store_true',
                        help='''use cpu to do depth inference.
                                WARNING: It is going to be EXTREMELY SLOW!
                                about 37s/view, so in total 30min/scan. 
                             ''')
    # for depth prediction
    parser.add_argument('--n_views', type=int, default=3,
                        help='number of views (including ref) to be used in testing')
    parser.add_argument('--depth_interval', type=float, default=2.65,
                        help='depth interval unit in mm')
    parser.add_argument('--n_depths', nargs='+', type=int, default=[8,32,48],
                        help='number of depths in each level')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[1.0,2.0,4.0],
                        help='depth interval ratio to multiply with --depth_interval in each level')
    parser.add_argument('--num_groups', type=int, default=1, choices=[1, 2, 4, 8],
                        help='number of groups in groupwise correlation, must be a divisor of 8')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[640,512],
                        help='resolution (img_w, img_h) of the image, must be multiples of 32')
    parser.add_argument('--ckpt_path', type=str, default='/openbayes/input/input0/project/dp_simple/CasMVSNet_pl/ckpts/_ckpt_epoch_10.ckpt',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--save_visual', default=False, action='store_true',
                        help='save depth and proba visualization or not')

    # for point cloud fusion
    parser.add_argument('--conf', type=float, default=0.999,
                        help='min confidence for pixel to be valid')
    parser.add_argument('--min_geo_consistent', type=int, default=5,
                        help='min number of consistent views for pixel to be valid')
    parser.add_argument('--max_ref_views', type=int, default=400,
                        help='max number of ref views (to limit RAM usage)')
    parser.add_argument('--skip', type=int, default=1,
                        help='''how many points to skip when creating the point cloud.
                                Larger = fewer points and smaller file size.
                                Ref: skip=10 creates ~= 3M points = 50MB file
                                     skip=1 creates ~= 30M points = 500MB file
                             ''')
    args, _ = parser.parse_known_args()
    return args
  


def decode_batch(batch):
    imgs = batch['imgs']
    proj_mats = batch['proj_mats']
    init_depth_min = batch['init_depth_min'].item()
    depth_interval = batch['depth_interval'].item()
    scan, vid = batch['scan_vid']
    return imgs, proj_mats, init_depth_min, depth_interval, \
           scan, vid


# define read_image and read_proj_mat for each dataset

def read_image(dataset_name, root_dir, scan, vid):
    if dataset_name == 'dtu':
        return cv2.imread(os.path.join(root_dir,
                    f'Rectified/{scan}/rect_{vid+1:03d}_3_r5000.png'))
    if dataset_name == 'tanks':
        return cv2.imread(os.path.join(root_dir, scan,
                    f'images/{vid:08d}.jpg'))
    if dataset_name == 'blendedmvs':
        return cv2.imread(os.path.join(root_dir, scan,
                    f'blended_images/{vid:08d}.jpg'))


def read_refined_image(dataset_name, scan, vid):
    return cv2.imread(f'results/{dataset_name}/image_refined/{scan}/{vid:08d}.png')


def save_refined_image(image_refined, dataset_name, scan, vid):
    cv2.imwrite(f'results/{dataset_name}/image_refined/{scan}/{vid:08d}.png',
                image_refined)


def read_proj_mat(dataset_name, dataset, scan, vid):
    if dataset_name == 'dtu':
        return dataset.proj_mats[vid][0][0].numpy()
    if dataset_name in ['tanks', 'blendedmvs']:
        return dataset.proj_mats[scan][vid][0][0].numpy()


@jit(nopython=True, fastmath=True)
def xy_ref2src(xy_ref, depth_ref, P_world2ref,
               depth_src, P_world2src, img_wh):
    # create ref grid and project to ref 3d coordinate using depth_ref
    xyz_ref = np.vstack((xy_ref, np.ones_like(xy_ref[:1]))) * depth_ref
    xyz_ref_h = np.vstack((xyz_ref, np.ones_like(xy_ref[:1])))

    P = (P_world2src @ np.ascontiguousarray(np.linalg.inv(P_world2ref)))[:3]
    # project to src 3d coordinate using P_world2ref and P_world2src
    xyz_src_h = P @ xyz_ref_h.reshape(4,-1)
    xy_src = xyz_src_h[:2]/xyz_src_h[2:3]
    xy_src = xy_src.reshape(2, img_wh[1], img_wh[0])

    return xy_src


@jit(nopython=True, fastmath=True)
def xy_src2ref(xy_ref, xy_src, depth_ref, P_world2ref,
               depth_src2ref, P_world2src, img_wh):
    # project xy_src back to ref view using the sampled depth
    xyz_src = np.vstack((xy_src, np.ones_like(xy_src[:1]))) * depth_src2ref
    xyz_src_h = np.vstack((xyz_src, np.ones_like(xy_src[:1])))
    P = (P_world2ref @ np.ascontiguousarray(np.linalg.inv(P_world2src)))[:3]
    xyz_ref_h = P @ xyz_src_h.reshape(4,-1)
    depth_ref_reproj = xyz_ref_h[2].reshape(img_wh[1], img_wh[0])
    xy_ref_reproj = xyz_ref_h[:2]/xyz_ref_h[2:3]
    xy_ref_reproj = xy_ref_reproj.reshape(2, img_wh[1], img_wh[0])

    # check |p_reproj-p_1| < 1
    pixel_diff = xy_ref_reproj - xy_ref
    mask_pixel_reproj = (pixel_diff[0]**2+pixel_diff[1]**2)<1

    # check |d_reproj-d_1| / d_1 < 0.01
    mask_depth_reproj = np.abs((depth_ref_reproj-depth_ref)/depth_ref)<0.01

    mask_geo = mask_pixel_reproj & mask_depth_reproj

    return depth_ref_reproj, mask_geo


def check_geo_consistency(depth_ref, P_world2ref,
                          depth_src, P_world2src,
                          image_ref, image_src,
                          img_wh):
    """
    Check the geometric consistency between ref and src views.
    """
    xy_ref = np.mgrid[:img_wh[1],:img_wh[0]][::-1].astype(np.float32)
    xy_src = xy_ref2src(xy_ref, depth_ref, P_world2ref,
                        depth_src, P_world2src, img_wh)

    # Sample the depth of xy_src using bilinear interpolation
    depth_src2ref = cv2.remap(depth_src,
                              xy_src[0].astype(np.float32),
                              xy_src[1].astype(np.float32),
                              interpolation=cv2.INTER_LINEAR)

    image_src2ref = cv2.remap(image_src,
                              xy_src[0].astype(np.float32),
                              xy_src[1].astype(np.float32),
                              interpolation=cv2.INTER_LINEAR)

    depth_ref_reproj, mask_geo = \
        xy_src2ref(xy_ref, xy_src, depth_ref, P_world2ref, 
                   depth_src2ref, P_world2src, img_wh)

    depth_ref_reproj[~mask_geo] = 0
    image_src2ref[~mask_geo] = 0
    
    return depth_ref_reproj, mask_geo, image_src2ref

# %%
args = get_opts()
dataset = dataset_dict['dtu'] \
            (args.root_dir, "train",
                n_views=args.n_views, depth_interval=args.depth_interval,
                )
dataset2 = dataset_dict['dtu'] \
            (args.root_dir,"val",
                n_views=args.n_views, depth_interval=args.depth_interval,
                )
dataset3 = dataset_dict['dtu'] \
            (args.root_dir,"test",
                n_views=args.n_views, depth_interval=args.depth_interval,
                )



# %%
metas = dataset.metas+dataset2.metas+dataset3.metas

# %%
args.scan = 'scan70'

# %%
refine = False
read_gt = False

# %%
scans = [f"scan{i}" for i in range(1,129)]

# %%
scan_bbox_file = "bbox.txt"
bboxes = {}

# %%
True
metas[6]

# %%
args.img_wh

# %%
# Step 2. Perform depth filtering and fusion
point_dir = f'results/{args.dataset_name}/points'
os.makedirs(point_dir, exist_ok=True)
print('Fusing point clouds...')
scans = os.listdir("/openbayes/input/input0/mvs_training/dtu/Rectified")
print(scans)
for scan in scans:
    print(f'Processing {scan} ...')
    # buffers for the final vertices of this scan
    vs = []
    v_colors = []

    image_refined = set()
    depth_refined = {}
    for meta in tqdm(list(filter(lambda x: (x[0]+"_train")==scan and x[1]==3, metas))[:args.max_ref_views]):
 
        ref_vid = meta[2]

        image_ref = read_image(args.dataset_name, args.root_dir, scan, ref_vid)
        image_ref = cv2.resize(image_ref, tuple(args.img_wh),
                                interpolation=cv2.INTER_LINEAR)[:,:,::-1] # to RGB

        depth_ref = read_pfm(f'{args.root_dir}/Depths/{scan}/depth_map_{ref_vid:04d}.pfm')[0]
        depth_ref = cv2.resize(depth_ref, tuple(args.img_wh),
                                interpolation=cv2.INTER_LINEAR)
            

        
        proba_ref = np.ones_like(depth_ref)

        mask_conf = proba_ref > args.conf # confidence mask
        P_world2ref = read_proj_mat(args.dataset_name, dataset, scan, ref_vid)
        
        src_vids = meta[3]
        mask_geos = []
        depth_ref_reprojs = [depth_ref]
        image_src2refs = [image_ref]
        # for each src view, check the consistency and refine depth
        for src_vid in src_vids:

            image_src = read_image(args.dataset_name, args.root_dir, scan, src_vid)
            image_src = cv2.resize(image_src, tuple(args.img_wh),
                                    interpolation=cv2.INTER_LINEAR)[:,:,::-1] # to RGB
            depth_src = read_pfm(f'{args.root_dir}/Depths/{scan}/depth_map_{src_vid:04d}.pfm')[0]
            depth_src = cv2.resize(depth_src, tuple(args.img_wh),
                                interpolation=cv2.INTER_LINEAR)
            depth_refined[src_vid] = depth_src
            P_world2src = read_proj_mat(args.dataset_name, dataset, scan, src_vid)
            depth_ref_reproj, mask_geo, image_src2ref = \
                check_geo_consistency(depth_ref, P_world2ref,
                                        depth_src, P_world2src,
                                        image_ref, image_src, tuple(args.img_wh))
            depth_ref_reprojs += [depth_ref_reproj]
            image_src2refs += [image_src2ref]
            mask_geos += [mask_geo]
        mask_geo_sum = np.sum(mask_geos, 0)
        mask_geo_final = mask_geo_sum >= args.min_geo_consistent
        depth_refined[ref_vid] = \
            (np.sum(depth_ref_reprojs, 0)/(mask_geo_sum+1)).astype(np.float32)
        image_refined_ = \
            np.sum(image_src2refs, 0)/np.expand_dims((mask_geo_sum+1), -1)

        image_refined.add(ref_vid)
        save_refined_image(image_refined_, args.dataset_name, scan, ref_vid)
        mask_final = mask_conf & mask_geo_final
        
        # create the final points
        xy_ref = np.mgrid[:args.img_wh[1],:args.img_wh[0]][::-1]
        xyz_ref = np.vstack((xy_ref, np.ones_like(xy_ref[:1]))) * depth_refined[ref_vid]
        xyz_ref = xyz_ref.transpose(1,2,0)[mask_final].T # (3, N)
        color = image_refined_[mask_final] # (N, 3)
        xyz_ref_h = np.vstack((xyz_ref, np.ones_like(xyz_ref[:1])))
        xyz_world = (np.linalg.inv(P_world2ref) @ xyz_ref_h).T # (N, 4)
        xyz_world = xyz_world[::args.skip, :3]
        color = color[::args.skip]
        
        # append to buffers
        vs += [xyz_world]
        v_colors += [color]



    # clear refined buffer
    image_refined.clear()
    depth_refined.clear()
    

    # process all points in the buffers
    vs = np.ascontiguousarray(np.vstack(vs).astype(np.float32))
    v_colors = np.vstack(v_colors).astype(np.uint8)
    print(f'{scan} contains {len(vs)/1e6:.2f} M points')
    vs.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    vertex_all = np.empty(len(vs), vs.dtype.descr+v_colors.dtype.descr)
    bboxes[scan] = {}
    for prop in vs.dtype.names:
        vertex_all[prop] = vs[prop][:, 0]
        bboxes[scan][prop]={}
       
        bboxes[scan][prop]["min"] =  vertex_all[prop].min()
        bboxes[scan][prop]["max"] =  vertex_all[prop].max()
        
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]
    print(vertex_all)
   


print('Done!')

# %%

# %%
import pickle as pkl
with open("bbox.pkl","wb") as f:
    pkl.dump(bboxes,f)

# %%

# %%


# %%
import subprocess
import os



# %%
import open3d


# %%



