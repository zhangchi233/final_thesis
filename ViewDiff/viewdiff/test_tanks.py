
# In[2]:


from dataclasses import dataclass, asdict
from typing import Union, Optional, Literal, Tuple
import os
from pathlib import Path
import json
import tyro
import copy
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.utils.checkpoint

from torch.utils.tensorboard import SummaryWriter

from accelerate.utils import set_seed
from accelerate.logging import get_logger
from diffusers import DPMSolverMultistepScheduler, UniPCMultistepScheduler, DDPMScheduler, DDIMScheduler
# %%
from torch.utils.data import Dataset
import sys
sys.path.append('/root/autodl-tmp/project/dp_simple/')
from CasMVSNet_pl.datasets.utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
from train_util import FinetuneConfig
from diffusers.loaders import LoraLoaderMixin

from dacite import from_dict, Config

from train import test_step

logger = get_logger(__name__, log_level="INFO")
import os
os.getcwd()
from model.custom_unet_2d_condition import (
    UNet2DConditionCrossFrameInExistingAttnModel,
)
from model.util import (
    replace_self_attention_with_cross_frame_attention,
    add_pose_cond_to_attention_layers,
    update_cross_frame_attention_config,
    update_last_layer_mode,
    update_vol_rend_inject_noise_sigma,
    update_n_novel_images,
    CrossFrameAttentionConfig,
    ModelConfig,
)
from model.custom_stable_diffusion_pipeline import CustomStableDiffusionPipeline
from model.custom_stable_instructPix2pix_pipeline import CustomInstructPix2pixDiffusionPipeline

from io_util import (
    setup_output_directories,
    make_output_directories,
    convert_to_tensorboard_dict,
    SaveConfig
)

from metrics.image_metrics import load_lpips_vgg_model

from model.util import (
    replace_self_attention_with_cross_frame_attention,
    update_last_layer_mode,
    update_vol_rend_inject_noise_sigma,
    update_n_novel_images,
    update_cross_frame_attention_config,
    add_pose_cond_to_attention_layers,
    collapse_prompt_to_batch_dim,
    collapse_tensor_to_batch_dim,
    expand_output_to_k,
    expand_tensor_to_k,
    tokenize_captions,
    ModelConfig,
    CrossFrameAttentionConfig,
    build_cross_attention_kwargs,
)



# In[3]:


import subprocess
import os
from torch.utils.data import Dataset
import sys
ROOTDIR = "root/autodl-tmp"
sys.path.append(f'/{ROOTDIR}/project/dp_simple/')
from CasMVSNet_pl.datasets.utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T


import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import crop
from torch.nn.functional import interpolate

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection
from pytorch3d.io import IO
import sys
import os
path = os.getcwd()
print(path)
from torch.utils.data import Dataset

import os
import numpy as np
import cv2
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms as T


class TanksDataset(Dataset):
    def __init__(self, root_dir, split='intermediate', n_views=3, levels=3, depth_interval=-1,
                 img_wh=(640,640)):
        """
        For testing only! You can write training data loader by yourself.
        @depth_interval has no effect. The depth_interval is predefined for each view.
        """
        self.root_dir = root_dir
        self.img_wh = img_wh
        assert img_wh[0]%32==0 and img_wh[1]%32==0, \
            'img_wh must both be multiples of 32!'
        self.split = split
        self.build_metas()
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.build_proj_mats()
        self.define_transforms()

    def build_metas(self):
        self.metas = []
        if self.split == 'intermediate':
            self.scans = ['Family', 'Francis', 'Horse', 'Lighthouse',
                          'M60', 'Panther', 'Playground', 'Train']
            self.image_sizes = {'Family': (1920, 1080),
                                'Francis': (1920, 1080),
                                'Horse': (1920, 1080),
                                'Lighthouse': (2048, 1080),
                                'M60': (2048, 1080),
                                'Panther': (2048, 1080),
                                'Playground': (1920, 1080),
                                'Train': (1920, 1080)}
            self.depth_interval = {'Family': 2.5e-3,
                                   'Francis': 1e-2,
                                   'Horse': 1.5e-3,
                                   'Lighthouse': 1.5e-2,
                                   'M60': 5e-3,
                                   'Panther': 5e-3,
                                   'Playground': 7e-3,
                                   'Train': 5e-3} # depth interval for each scan (hand tuned)
        elif self.split == 'advanced':
            self.scans = ['Auditorium', 'Ballroom', 'Courtroom',
                          'Museum', 'Palace', 'Temple']
            self.image_sizes = {'Auditorium': (1920, 1080),
                                'Ballroom': (1920, 1080),
                                'Courtroom': (1920, 1080),
                                'Museum': (1920, 1080),
                                'Palace': (1920, 1080),
                                'Temple': (1920, 1080)}
            self.depth_interval = {'Auditorium': 3e-2,
                                   'Ballroom': 2e-2,
                                   'Courtroom': 2e-2,
                                   'Museum': 2e-2,
                                   'Palace': 1e-2,
                                   'Temple': 1e-2} # depth interval for each scan (hand tuned)
        self.ref_views_per_scan = defaultdict(list)

        for scan in self.scans:
            with open(os.path.join(self.root_dir, self.split, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    self.metas += [(scan, -1, ref_view, src_views)]
                    self.ref_views_per_scan[scan] += [ref_view]

    def build_proj_mats(self):
        self.proj_mats = {} # proj mats for each scan
        for scan in self.scans:
            self.proj_mats[scan] = {}
            img_w, img_h = self.image_sizes[scan]
            for vid in self.ref_views_per_scan[scan]:
                proj_mat_filename = os.path.join(self.root_dir, self.split, scan,
                                                 f'cams/{vid:08d}_cam.txt')
                intrinsics, extrinsics, depth_min = \
                    self.read_cam_file(proj_mat_filename)
                intrinsics[0] *= self.img_wh[0]/img_w/4
                intrinsics[1] *= self.img_wh[1]/img_h/4
                # self.depth_interval[scan][vid] = depth_interval

                # multiply intrinsics and extrinsics to get projection matrix
                proj_mat_ls = []
                for l in reversed(range(self.levels)):
                    proj_mat_l = np.eye(4)
                    proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                    intrinsics[:2] *= 2 # 1/4->1/2->1
                    proj_mat_ls += [torch.FloatTensor(proj_mat_l)]
                # (self.levels, 4, 4) from fine to coarse
                proj_mat_ls = torch.stack(proj_mat_ls[::-1])
                self.proj_mats[scan][vid] = (proj_mat_ls,intrinsics,extrinsics, depth_min)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        return intrinsics, extrinsics, depth_min

    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    # T.Normalize(mean=[0.485, 0.456, 0.406], 
                                    #             std=[0.229, 0.224, 0.225]),
                                    ])

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        sample = {}
        scan, _, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + [12,11] #src_views[:self.n_views-1]
        task = np.random.choice([
                                 "flare",
                                 # "overexposed",
                                  #"shadow","different dark"
                                 ],1)
        sample['prompt'] = [f"modify the images for task {task}"]
        sample['prompt'] = [f"modify the lightness of image to light_class_{4} style"]
        intensity_stats=[]
        imgs = []
        Rs = []
        Ks = []
        proj_mats = [] # record proj mats between views
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.root_dir, self.split, scan, f'images/{vid:08d}.jpg')

            img = Image.open(img_filename)
            img = img.resize(self.img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]

            proj_mat_ls,K,R, depth_min = self.proj_mats[scan][vid]
            Ks += [K]
            Rs += [R]
            if i == 0:  # reference view
                ref_proj_inv = torch.inverse(proj_mat_ls)
                sample['init_depth_min'] = torch.FloatTensor([depth_min])
                sample['depth_interval'] = torch.FloatTensor([self.depth_interval[scan]])
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]
            var, mean = torch.var_mean(img)
            intensity_stat = torch.stack([mean, var], dim=0)
            intensity_stats.append(intensity_stat)

        imgs = torch.stack(imgs) # (V, 3, H, W)
        proj_mats = torch.stack(proj_mats)[:,:,:3] # (V-1, self.levels, 3, 4) from fine to coarse
        Ks = np.stack(Ks)
        Rs = np.stack(Rs)
        sample['pose'] = torch.tensor(Rs)
        sample['K'] = torch.tensor(Ks)
        sample['images'] = imgs
        sample["target_imgs"] = imgs
        sample['proj_mats'] = proj_mats
        sample['scan_vid'] = (scan, [ref_view]+src_views[:self.n_views-1])
        x_min, y_min, z_min = -1, -1, -1
        x_max, y_max, z_max = 1, 1, 1
        sample["bbox"] =torch.tensor([[x_min,y_min,z_min], 
                                    [x_max,y_max,z_max]], dtype=torch.float32)
        sample["intensity_stats"] = torch.stack(intensity_stats)
        

        return sample


        





# %%

# %%
val_data = TanksDataset("/root/autodl-tmp/tankandtemples")

# %%
class runfig:
    pretrained_model_name_or_path =  "/root/autodl-tmp/output_var_unproj_eth2/all/subset_all/input_3/train/class6/saved_model_from_checkpoint-6000"
    n_input_images =3
    n_output_noise =3

# %%
config_path = config_path = os.path.join(runfig.pretrained_model_name_or_path, "config.json")
if not os.path.isfile(str(config_path)):
        raise ValueError("cannot find config.json in ", config_path)
with open(config_path, "r") as f:
    config_data = json.load(f)
finetune_config = from_dict(FinetuneConfig, data=config_data, config=Config(cast=[tuple, int]))
runfig.cross_frame_attention = finetune_config.cross_frame_attention
runfig.model = finetune_config.model


# %%
pipeline = CustomInstructPix2pixDiffusionPipeline.from_pretrained(
        runfig.pretrained_model_name_or_path
    )
pipeline.scheduler.config.prediction_type = finetune_config.training.noise_prediction_type

# %%
replace_self_attention_with_cross_frame_attention(
                unet=pipeline.unet,
                n_input_images=runfig.n_input_images,
                to_k_other_frames=runfig.cross_frame_attention.to_k_other_frames,
                with_self_attention=runfig.cross_frame_attention.with_self_attention,
                random_others=runfig.cross_frame_attention.random_others,
                use_lora_in_cfa="cfa" in runfig.model.pose_cond_mode or "sa" in runfig.model.pose_cond_mode,
                use_temb_in_lora=runfig.cross_frame_attention.use_temb_cond,
                temb_out_size=8,
                pose_cond_dim=runfig.model.pose_cond_dim,
                rank=runfig.model.pose_cond_lora_rank,
            )

# %%
runfig.cross_frame_attention.last_layer_mode,finetune_config.training.changed_cfa_last_layer

# %%
runfig.cross_frame_attention.with_self_attention,runfig.cross_frame_attention.random_others,

# %%
def update_cfa_config(
   
    pipeline: CustomInstructPix2pixDiffusionPipeline,
):
    if runfig.cross_frame_attention.mode == "add_in_existing_block":
        update_cross_frame_attention_config(
            pipeline.unet,
            runfig.n_input_images,
            runfig.cross_frame_attention.to_k_other_frames,
            runfig.cross_frame_attention.with_self_attention,
            runfig.cross_frame_attention.random_others,
            change_self_attention_layers=False,  # should have custom cfa layers
        )
    elif runfig.cross_frame_attention.mode == "pretrained":
        update_cross_frame_attention_config(
            pipeline.unet,
            3,
            2,
            runfig.cross_frame_attention.with_self_attention,
            runfig.cross_frame_attention.random_others,
            change_self_attention_layers=True,  # should have cfa is sa layers
        )
    else:
        raise NotImplementedError(
            f"did not implement different n_input_images for cfa.mode={runfig.cross_frame_attention.mode}"
        )

# %%
if finetune_config.training.changed_cfa_last_layer != runfig.cross_frame_attention.last_layer_mode:
        print("Change last-layer-mode to", finetune_config.training.changed_cfa_last_layer)
        update_last_layer_mode(
            pipeline.unet,
            finetune_config.training.changed_cfa_last_layer,
        )
update_vol_rend_inject_noise_sigma(
        pipeline.unet, 0.0
    )
    # disable n_novel_images
update_n_novel_images(
        pipeline.unet, 0

    )
update_cfa_config(pipeline)

# %%
if runfig.model.pose_cond_mode != "none":
        # Set correct lora layers
        unet_lora_attn_procs, unet_lora_parameters = add_pose_cond_to_attention_layers(
            pipeline.unet,
            rank=runfig.model.pose_cond_lora_rank,
            pose_cond_dim=runfig.model.pose_cond_dim,
            only_cross_attention="sa" not in runfig.model.pose_cond_mode,
        )

        if unet_lora_parameters is not None:
            in_dir = os.path.join(runfig.pretrained_model_name_or_path, "unet")
            try:
                lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(in_dir, weight_name="pytorch_lora_weights.safetensors")
            except:
                lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(in_dir, weight_name="pytorch_lora_weights.bin")
            lora_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items()}
            pipeline.unet.load_state_dict(lora_state_dict, strict=False)
            print("Loaded LoRA weights into model")

# %%
pipeline = pipeline.to("cuda")

# %%
runfig.model.pose_cond_mode

# %%
import torch
@torch.no_grad()
def process_batch(
    
   
    pipeline,
   
    batch,
    guidance_scale=8,
    image_guidance_scale: float = 2.0,
    
):
    
    model_config=runfig.model
    cfa_config=runfig.cross_frame_attention
    io_config=runfig
    orig_hw=(640,640)
    num_inference_steps=50
    n_repeat_generation=1
    generator = torch.Generator("cuda").manual_seed(42)

    # combine
    
    batch["images"] = batch["images"].to("cuda").unsqueeze(0)
    batch["target_imgs"] = batch["target_imgs"].to("cuda").unsqueeze(0) 
    batch["pose"] = batch["pose"].to("cuda").unsqueeze(0)
    batch["K"] = batch["K"].to("cuda").unsqueeze(0)
    batch["intensity_stats"] = batch["intensity_stats"].to("cuda").unsqueeze(0)
    batch["bbox"] = batch["bbox"].to("cuda").unsqueeze(0)

    # check if need to change n_input_images
    if runfig.n_input_images != batch["pose"].shape[1]:
        runfig.n_input_images = batch["pose"].shape[1]
        runfig.cross_frame_attention.to_k_other_frames = batch["pose"].shape[1] - 1
        runfig.model.n_input_images = batch["pose"].shape[1]
        update_cfa_config(runfig, pipeline)

    # alwasy set to 0
    batch["intensity_stats"] *= 0

    # create images
    batch_size = len(batch["prompt"])

    batch["images"] = 2*batch["images"]-1
    batch["target_imgs"] = 2*batch["target_imgs"]-1
    # parse batch
    # collapse K dimension into batch dimension (no concatenation happening)
    batch["prompt"] = [cap for cap in batch["prompt"]]
    prompt = collapse_prompt_to_batch_dim(batch["prompt"],3)
   
    
    _, pose = collapse_tensor_to_batch_dim(batch["pose"])
    _, K = collapse_tensor_to_batch_dim(batch["K"])
    _, intensity_stats = collapse_tensor_to_batch_dim(batch["intensity_stats"])
    bbox = batch["bbox"]

    _, known_images = collapse_tensor_to_batch_dim(batch["images"])
    known_images = known_images.to(pipeline.device)
    known_images = known_images.squeeze(1)
    print(known_images.shape)

    K = K.squeeze(1)[..., :3, :3]
    pose = pose.squeeze(1)
    intensity_stats = intensity_stats.squeeze(1)

    # build cross_attention_kwargs
    cross_attention_kwargs = build_cross_attention_kwargs(
        model_config=runfig.model,
        cfa_config=runfig.cross_frame_attention,
        pose=pose,
        K=K,
        intensity_stats=intensity_stats,
        bbox=bbox,
        orig_hw=orig_hw,
    )
    if "pose_cond" in cross_attention_kwargs:
            cross_attention_kwargs["pose_cond"] = torch.cat([cross_attention_kwargs["pose_cond"]] * 3)
    if "unproj_reproj_kwargs" in cross_attention_kwargs:
        proj_kwargs = cross_attention_kwargs["unproj_reproj_kwargs"]
        proj_kwargs["pose"] = torch.cat([proj_kwargs["pose"]] * 3)
        proj_kwargs["K"] = torch.cat([proj_kwargs["K"]] * 3)
        proj_kwargs["bbox"] = torch.cat([proj_kwargs["bbox"]] * 3)

    outputs = []
    all_psnrs = []
    all_lpipses = []
    all_ssims = []
    for _ in range(n_repeat_generation):
        output = pipeline(
            prompt=prompt,
            height=orig_hw[0],
            width=orig_hw[1],
            known_images=known_images,
            output_type="pt",  # return tensor normalized to [0, 1]
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            decode_all_timesteps=True,
            num_inference_steps=num_inference_steps,
            n_images_per_batch=model_config.n_input_images,
        )

        # re-create K dimension from batch dimension
        output.images = output.images.unsqueeze(1)
        expand_output_to_k(output, batch_size, model_config.n_input_images)

        outputs.append(output)
   



    

    return outputs



# %%
# batch = val_data[0]
# outputs = process_batch(pipeline, batch)


# %%
from torchvision.utils import save_image
import matplotlib.pyplot as plt
# save_image(outputs[0].images[0], "output.png")
# batch = val_data[0]

# save_image(batch["images"], "input.png")
# save_image(batch["target_imgs"], "target.png")


# %%
from pandas import DataFrame
print(val_data.metas[0])
df = DataFrame(val_data.metas,columns=["scan","ref_view","src_views","target_light"])    
df.head()                                       

# %%
print(df["scan"].unique())


# %%
from math import sqrt
import sys
sys.path.append('/root/autodl-tmp/project/dp_simple/')
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

import pytorch_lightning as pl
import sys


import matplotlib.pyplot as plt

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

from CasMVSNet_pl.datasets.utils import save_pfm, read_pfm
import cv2
import torch
import os, shutil
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

# for depth prediction
from CasMVSNet_pl.models.mvsnet import CascadeMVSNet
from CasMVSNet_pl.utils import load_ckpt
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
                        default='/root/autodl-tmp/eth3d_high_res_test',
                        help='root directory of dtu dataset')
    parser.add_argument('--dataset_name', type=str, default='dtu2',
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
    parser.add_argument('--num_groups', type=int, default=8, choices=[1, 2, 4, 8],
                        help='number of groups in groupwise correlation, must be a divisor of 8')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[640,512],
                        help='resolution (img_w, img_h) of the image, must be multiples of 32')
    parser.add_argument('--ckpt_path', type=str, default='/root/autodl-tmp/project/dp_simple/CasMVSNet_pl/ckpts/_ckpt_epoch_10.ckpt',
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
  
    

# %%
def decode_batch(batch):
    imgs = batch['images']
    proj_mats = batch['proj_mats']
    init_depth_min = batch['init_depth_min'].item()
    depth_interval = batch['depth_interval'].item()
    scan, vid = batch['scan_vid']
    return imgs, proj_mats, init_depth_min, depth_interval, \
           scan, vid


# define read_image and read_proj_mat for each dataset

def read_image(dataset_name, root_dir, scan, vid,light):
    if dataset_name == 'dtu':
        return cv2.imread(os.path.join(root_dir,
                    f'Rectified/{scan}_train/rect_{vid+1:03d}_{light}_r5000.png'))
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


# %%
args = get_opts()

# %%
model = CascadeMVSNet(n_depths=args.n_depths,
                        interval_ratios=args.interval_ratios,
                        num_groups=1,
                        norm_act=ABN)
device = 'cpu' if args.cpu else 'cuda:0'
model.to(device)
load_ckpt(model, args.ckpt_path)
model.eval()


# %%


# %%
transform = T.Compose([
   
    T.Normalize(mean=[0.485, 0.456, 0.406], 
std=[0.229, 0.224, 0.225])])

# %%
refine = True
read_gt = False

# %%
args.save_visual=True   

# %%
val_data.metas

# %%

print('Creating depth and confidence predictions...')


for scan in ["Lighthouse"]:
    dataset_path = os.path.join("/root/autodl-tmp/eth3d_high_res_test",scan,"image_refine")
    os.makedirs(dataset_path, exist_ok=True)
    depth_dir = f'./results/{args.dataset_name}/depth'
    depth_dir = os.path.join(depth_dir, scan)

    img_dir = f'./results/{args.dataset_name}/image_modified'
    img_dir = os.path.join(img_dir, scan)

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)


    data_range = [i for i, x in enumerate(val_data.metas) if x[0] == scan and (25>= x[2] >=15) ]
    print(f'Processing {scan} with {len(data_range)} views')
    for i in tqdm(data_range):
        batch =  val_data[i]
    
        imgs, proj_mats, init_depth_min, depth_interval, \
            scan, vid = decode_batch(batch)
        src1 = vid[1]
        src2 = vid[2]
        vid = vid[0]
        
        proj_mats = proj_mats.unsqueeze(0).to("cuda")
        imgs = imgs.unsqueeze(0).to(device)
        target_imgs = batch["target_imgs"].unsqueeze(0).to("cuda")

        
        
        
    
        
        
        
        os.makedirs(os.path.join(depth_dir, scan), exist_ok=True)

        with torch.no_grad():
        
            
            
            if refine == True:
                # whether image exist or not
                ref_path = os.path.join(img_dir, f'{vid:04d}.png')
                src1_path = os.path.join(img_dir, f'{src1:04d}.png')
                src2_path = os.path.join(img_dir, f'{src2:04d}.png')
                if False and os.path.exists(os.path.join(dataset_path, f'{vid:08d}.png')):
                    
                    img1 = cv2.imread(ref_path)
                    img2 = cv2.imread(src1_path)
                    img3 = cv2.imread(src2_path)

                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

                    

                    convert_totensor = T.Compose([
                        T.ToTensor(),
                        
                    ])
                    modified_imgs = torch.stack([convert_totensor(img1), convert_totensor(img2), convert_totensor(img3)]).unsqueeze(0).to("cuda")

                    
                    save_image(torch.cat([imgs[0], modified_imgs[0],target_imgs[0]], dim=0), os.path.join(img_dir,f"{vid}_{src1}_{src2}.png"),nrow=3)
                    
                    print("load modified:", modified_imgs.shape)

                    
                
                else:
                    for _ in range(1):
                        modified_imgs = process_batch(pipeline,batch,
                                                    10,3)[0].images
                        batch =  val_data[i]
    
                        imgs, proj_mats, init_depth_min, depth_interval, \
                            scan, vid = decode_batch(batch)
                        src1 = vid[1]
                        src2 = vid[2]
                        vid = vid[0]
                        
                        proj_mats = proj_mats.unsqueeze(0).to("cuda")
                        imgs = imgs.unsqueeze(0).to(device)
                        target_imgs = batch["target_imgs"].unsqueeze(0).to("cuda")
                        print(batch["images"].shape, modified_imgs.shape)   
                        batch["images"] = modified_imgs[0]

                    
                    #save image
                    print(modified_imgs.shape,"modified_images")
                    image1, image2, image3 = modified_imgs[0]

                    ref_img_path = os.path.join(img_dir, f'{vid:04d}.png')

                    refine_img_path = os.path.join(dataset_path, f'{vid:08d}.png')
                    print("save to ",refine_img_path)
                    save_image(image1, refine_img_path)

                    src1_img_path = os.path.join(img_dir, f'{src1:04d}.png')
                    src2_img_path = os.path.join(img_dir, f'{src2:04d}.png')
                    save_image(image1, ref_img_path)
                    save_image(image2, src1_img_path)
                    save_image(image3, src2_img_path)


                    
                

                    
                
            
                
                print("save modified:", modified_imgs.shape)
                results_modified = model(transform(modified_imgs), proj_mats, init_depth_min, depth_interval)
                
            
                
                imgs_original = imgs[0]
                pred_imgs = modified_imgs[0]
                target_image = target_imgs[0]
            
                save_image(torch.cat([imgs_original, pred_imgs,target_image], dim=0), 
                        os.path.join(img_dir, f'{vid:04d}_together.png'),nrow =3)
                
                

            
                results_ori = model(transform(imgs), proj_mats, init_depth_min, depth_interval)
                results_target = model(transform(target_imgs), proj_mats, init_depth_min, depth_interval)
                

            


                



                
            else:
                results_target = model(transform(target_imgs), proj_mats, init_depth_min, depth_interval)
                results_ori = model(transform(imgs), proj_mats, init_depth_min, depth_interval)


            
        if refine == True:
            depth = results_modified['depth_0'][0].cpu().numpy()
            depth = np.nan_to_num(depth)
            proba = results_modified['confidence_2'][0].cpu().numpy()
            proba = np.nan_to_num(proba)
            save_pfm(os.path.join(depth_dir, f'{scan}/depth_refined_{vid:04d}.pfm'), depth)
            save_pfm(os.path.join(depth_dir, f'{scan}/proba_refined_{vid:04d}.pfm'), proba)

            if args.save_visual:
                mi = np.min(depth[depth>0])
                ma = np.max(depth)
                depth = (depth-mi)/(ma-mi+1e-8)
                depth = (255*depth).astype(np.uint8)
                depth_img1 = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(depth_dir, f'{scan}/depth_visual_refined_{vid:04d}.jpg'),
                            depth_img1)
                cv2.imwrite(os.path.join(depth_dir, f'{scan}/proba_visual_refined_{vid:04d}.jpg'),
                            (255*(proba>args.conf)).astype(np.uint8))
        
        depth = results_ori['depth_0'][0].cpu().numpy()
        depth = np.nan_to_num(depth) # change nan to 0
        proba = results_ori['confidence_2'][0].cpu().numpy() # NOTE: this is 1/4 scale!
        proba = np.nan_to_num(proba) # change nan to 0
        save_pfm(os.path.join(depth_dir, f'{scan}/depth_{vid:04d}.pfm'), depth)
        save_pfm(os.path.join(depth_dir, f'{scan}/proba_{vid:04d}.pfm'), proba)

        depth_target = results_target['depth_0'][0].cpu().numpy()
        depth_target = np.nan_to_num(depth_target) # change nan to 0
        proba_target = results_target['confidence_2'][0].cpu().numpy() # NOTE: this is 1/4 scale!
        proba_target = np.nan_to_num(proba_target) # change nan to 0
        save_pfm(os.path.join(depth_dir, f'{scan}/depth_target_{vid:04d}.pfm'), depth_target)
        save_pfm(os.path.join(depth_dir, f'{scan}/proba_target_{vid:04d}.pfm'), proba_target)

        if args.save_visual:
            mi = np.min(depth[depth>0])
            ma = np.max(depth)
            depth = (depth-mi)/(ma-mi+1e-8)
            depth = (255*depth).astype(np.uint8)
            depth_img2 = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
            
            cv2.imwrite(os.path.join(depth_dir, f'{scan}/depth_visual_{vid:04d}.jpg'),
                        depth_img2)
            cv2.imwrite(os.path.join(depth_dir, f'{scan}/proba_visual_{vid:04d}.jpg'),
                        (255*(proba>args.conf)).astype(np.uint8))

            mi = np.min(depth_target[depth_target>0])
            ma = np.max(depth_target)
            depth_target = (depth_target-mi)/(ma-mi+1e-8)
            depth_target = (255*depth_target).astype(np.uint8)
            depth_img3 = cv2.applyColorMap(depth_target, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(depth_dir, f'{scan}/depth_visual_target_{vid:04d}.jpg'),
                        depth_img3)
            cv2.imwrite(os.path.join(depth_dir, f'{scan}/proba_visual_target_{vid:04d}.jpg'),
                        (255*(proba_target>args.conf)).astype(np.uint8))
            if refine == True:
                # concat the 3 depth images
                depth_img = np.concatenate([depth_img1, depth_img2, depth_img3], axis=1)
                cv2.imwrite(os.path.join(depth_dir, f'{scan}/depth_visual_{vid:04d}_together.jpg'),
                            depth_img)
            

    
        del imgs, proj_mats, results_ori
    
    torch.cuda.empty_cache()

# %%
refine=True

# %%
# Step 2. Perform depth filtering and fusion
point_dir = f'results/{args.dataset_name}/points'
os.makedirs(point_dir, exist_ok=True)
print('Fusing point clouds...')

for scan in ["scan106"]:
    print(f'Processing {scan} ...')
    
    # buffers for the final vertices of this scan
    vs = []
    v_colors = []
    # buffers storing the refined data of each ref view
    os.makedirs(f'results/{args.dataset_name}/image_refined/{scan}', exist_ok=True)
    image_refined = set()
    depth_refined = {}
    for meta in tqdm(list(filter(lambda x: x[0]==scan and x[2]==0, val_data.metas))[:args.max_ref_views]):
       
        try:
            ref_vid = meta[1]
            if ref_vid in image_refined: # not yet refined actually
                image_ref = read_refined_image(args.dataset_name, scan, ref_vid)
                depth_ref = depth_refined[ref_vid]
            else:
                if refine:
                    img_dir = f'./results/{args.dataset_name}/image_modified/{scan}'
                    image_ref = np.load(os.path.join(img_dir, f'{ref_vid:04d}_class6.npy'))[0]
                    print(image_ref.shape)
                    image_ref *= 255
                    image_ref = image_ref.transpose(1,2,0)
                    image_ref = image_ref.astype(np.uint8)
                    image_ref = cv2.resize(image_ref, tuple(args.img_wh))
                    plt.imshow(image_ref)
                    plt.show()
                    print(image_ref.shape)

                else:
                    image_ref = read_image(args.dataset_name, args.root_dir, scan, ref_vid,0)
                    image_ref = cv2.resize(image_ref, tuple(args.img_wh),
                                            interpolation=cv2.INTER_LINEAR)[:,:,::-1] # to RGB
                
                
                if refine:
                    depth_ref = read_pfm(f'results/{args.dataset_name}/depth/' \
                                            f'{scan}/{scan}/depth_refined_{ref_vid:04d}.pfm')[0]
                    print(depth_ref.shape)
                else:
                    depth_ref = read_pfm(f'results/{args.dataset_name}/depth/' \
                                        f'{scan}/{scan}/depth_{ref_vid:04d}.pfm')[0]
            
            if refine:
                proba_ref = read_pfm(f'results/{args.dataset_name}/depth/' \
                                    f'{scan}/{scan}/proba_refined_{ref_vid:04d}.pfm')[0]
                proba_ref = cv2.resize(proba_ref, None, fx=4, fy=4,
                                        interpolation=cv2.INTER_LINEAR)
            else:
                proba_ref = read_pfm(f'results/{args.dataset_name}/depth/' \
                                        f'{scan}/{scan}/proba_{ref_vid:04d}.pfm')[0]
                proba_ref = cv2.resize(proba_ref, None, fx=4, fy=4,
                                        interpolation=cv2.INTER_LINEAR)
            mask_conf = proba_ref > args.conf # confidence mask
            P_world2ref = read_proj_mat(args.dataset_name, val_data, scan, ref_vid)
            
            src_vids = meta[3]
            mask_geos = []
            depth_ref_reprojs = [depth_ref]
            image_src2refs = [image_ref]
            # for each src view, check the consistency and refine depth
            for src_vid in src_vids:
                if src_vid in image_refined: # use refined data of previous runs
                    image_src = read_refined_image(args.dataset_name, scan, src_vid)
                    depth_src = depth_refined[src_vid]
                else:
                    if refine:
                        img_dir = f'./results/{args.dataset_name}/image_modified/{scan}'
                        image_src = np.load(os.path.join(img_dir, f'{src_vid:04d}_class6.npy'))[0]
                        print(image_src.shape)
                        image_src *= 255
                        image_src = image_src.transpose(1,2,0)
                        image_src= image_src.astype(np.uint8)
                        image_src = cv2.resize(image_src, tuple(args.img_wh))
                    else: 
                        image_src = read_image(args.dataset_name, args.root_dir, scan, src_vid,6)
                        image_src = cv2.resize(image_src, tuple(args.img_wh),
                                                interpolation=cv2.INTER_LINEAR)[:,:,::-1] # to RGB




                    if read_gt:
                        depth_src = read_pfm(f'{args.root_dir}/Depths/{scan}/depth_map_{src_vid:04d}.pfm')[0]
                        depth_src = cv2.resize(depth_src, tuple(args.img_wh),
                                                interpolation=cv2.INTER_LINEAR)
                    else:
                        if refine:
                            depth_src = read_pfm(f'results/{args.dataset_name}/depth/' \
                                                f'{scan}/{scan}/depth_refined_{src_vid:04d}.pfm')[0]
                        else:
                            depth_src = read_pfm(f'results/{args.dataset_name}/depth/' \
                                                f'{scan}/{scan}/depth_{src_vid:04d}.pfm')[0]
                        

                   
                    depth_refined[src_vid] = depth_src
                P_world2src = read_proj_mat(args.dataset_name, val_data, scan, src_vid)
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

        except Exception as e:
            # some scenes might not have depth prediction due to too few valid src views
            
            print(f'Error: {e}')
    # clear refined buffer
    image_refined.clear()
    depth_refined.clear()
    shutil.rmtree(f'results/{args.dataset_name}/image_refined/{scan}')

    # process all points in the buffers
    vs = np.ascontiguousarray(np.vstack(vs).astype(np.float32))
    v_colors = np.vstack(v_colors).astype(np.uint8)
    print(f'{scan} contains {len(vs)/1e6:.2f} M points')
    vs.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    vertex_all = np.empty(len(vs), vs.dtype.descr+v_colors.dtype.descr)
    for prop in vs.dtype.names:
        vertex_all[prop] = vs[prop][:, 0]
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]
    if read_gt:
        el = PlyElement.describe(vertex_all, 'vertex')
        PlyData([el]).write(f'{point_dir}/{scan}_gt.ply')
    elif refine:
        el = PlyElement.describe(vertex_all, 'vertex')
        PlyData([el]).write(f'{point_dir}/{scan}_refine.ply')
    
    else:
        el = PlyElement.describe(vertex_all, 'vertex')
        PlyData([el]).write(f'{point_dir}/{scan}.ply')
    del vertex_all, vs, v_colors
shutil.rmtree(f'results/{args.dataset_name}/image_refined')

print('Done!')


