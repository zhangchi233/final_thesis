from torch.utils.data import Dataset
import sys
sys.path.append('/workspace/project/dp_simple/')
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

@dataclass
class DatasetArgsConfig:
    """Arguments for JsonIndexDataset. See here for a full list: pytorch3d/implicitron/dataset/json_index_dataset.py"""

    remove_empty_masks: bool = False
    """Removes the frames with no active foreground pixels
            in the segmentation mask after thresholding (see box_crop_mask_thr)."""

    load_point_clouds: bool = False
    """If pointclouds should be loaded from the dataset"""

    load_depths: bool = False
    """If depth_maps should be loaded from the dataset"""

    load_depth_masks: bool = False
    """If depth_masks should be loaded from the dataset"""

    load_masks: bool = False
    """If foreground masks should be loaded from the dataset"""

    box_crop: bool = False
    """Enable cropping of the image around the bounding box inferred
            from the foreground region of the loaded segmentation mask; masks
            and depth maps are cropped accordingly; cameras are corrected."""

    image_width: Optional[int] = 640
    """The width of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing."""

    image_height: Optional[int] = 512
    """The height of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing."""

    pick_sequence: Tuple[str, ...] = ()
    """A list of sequence names to restrict the dataset to."""

    exclude_sequence: Tuple[str, ...] = ()
    """a list of sequences to exclude"""

    n_frames_per_sequence: int = -1
    """If > 0, randomly samples #n_frames_per_sequence
        frames in each sequences uniformly without replacement if it has
        more frames than that; applied before other frame-level filters."""


@dataclass
class BatchConfig:
    """Arguments for how batches are constructed."""

    n_parallel_images: int = 3
    """How many images of the same sequence are selected in one batch (used for multi-view supervision)."""

    image_width: int = 512
    """The desired image width after applying all augmentations (e.g. crop) and resizing operations."""

    image_height: int = 512
    """The desired image height after applying all augmentations (e.g. crop) and resizing operations."""

    other_selection: Literal["random", "sequence", "mix", "fixed-frames"] = "random"
    """How to select the other frames for each batch.
        The mode 'random' selects the other frames at random from all remaining images in the dataset.
        The mode 'sequence' selects the other frames in the order as they appear after the first frame (==idx) in the dataset. Selects i-th other image as (idx + i * sequence_offset).
        The mode 'mix' decides at random which of the other two modes to choose. It also randomly samples sequence_offset when choosing the mode 'sequence'.
        The mode 'fixed-frames' gets as frame indices as input and directly uses them."""

    other_selection_frame_indices: Tuple[int, ...] = ()
    """The frame indices to use when --other_selection=fixed-frames. Must be as many indices as --n_parallel_images."""

    sequence_offset: int = 1
    """If other_selection='sequence', uses this offset to determine how many images to skip for each next frame.
    Allows to do short-range and long-range consistency tests by setting to a small or large number."""

    crop: Literal["random", "foreground", "resize", "center"] = "random"
    """Performs a crop on the original image such that the desired (image_height, image_width) is achieved. 
       The mode 'random' crops randomly in the image.
       The mode 'foreground' crops centered around the foreground (==object) mask.
       The mode 'resize' performs brute-force resizing which ignores the aspect ratio.
       The mode 'center' crops centered around the middle pixel (similar to DFM baseline)."""

    mask_foreground: bool = False
    """If true, will mask out the background and only keep the foreground."""

    prompt: str = "Editorial Style Photo, ${category}, 4k --ar 16:9"
    """The text prompt for generation. The string ${category} will be replaced with the actual category."""

    use_blip_prompt: bool = False
    """If True, will use blip2 generated prompts for the sequence instead of the prompt specified in --prompt."""

    load_recentered: bool = False
    """If True, will load the recentered poses/bbox from the dataset. Will skip all sequences for which this was not pre-computed."""

    replace_pose_with_spherical_start_phi: float = -400.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this many degrees. Default: -400, meaning do not replace."""

    replace_pose_with_spherical_end_phi: float = 360.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this many degrees. Default: -1, meaning do not replace."""

    replace_pose_with_spherical_phi_endpoint: bool = False
    """If True, will set endpoint=True for np.linspace, else False."""

    replace_pose_with_spherical_radius: float = 4.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this radius. Default: 3.0."""

    replace_pose_with_spherical_theta: float = 45.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this elevation. Default: 45.0."""



    
@dataclass
class DTUConfig:
    """Arguments for setup of the CO3Dv2_Dataset."""
    dataset_args: DatasetArgsConfig
    batch: BatchConfig

   
    

    category: Optional[str] = None
    """If specified, only selects this category from the dataset. Can be a comma-separated list of categories as well."""

    subset: Optional[str] = None
    """If specified, only selects images corresponding to this subset. See https://github.com/facebookresearch/co3d for available options."""

    split: Optional[str] = None
    """Must be specified if --subset is specified. Tells which split to use from the subset."""

    max_sequences: int = -1
    """If >-1, randomly select max_sequence sequences per category. Only sequences _with pointclouds_ are selected. Mutually exclusive with --sequence."""

    seed: Optional[int] = 42
    """Random seed for all rng objects"""
 
    split: Optional[str] = None
    """Must be specified if --subset is specified. Tells which split to use from the subset."""

    root_dir: str = "/workspace/mvs_training/dtu/"

    n_views:int=3 
    levels:int=3 
    depth_interval:int =2.65
    img_wh:int=None
    abs_error:Optional[str] ="abs"
    output_total:Optional[bool]=False
    threshold: Optional[int] = 4.7
    prompt_dir: Optional[str] = "/workspace/mvs_training/dtu/co3d_blip2_captions_final.json"

class DTUDataset(Dataset):
    def __init__(self, config: DTUConfig):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """

        self.root_dir = config.root_dir
        self.split = config.split
        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train", "val" or "test"!'
        self.img_wh = (config.batch.image_height,config.batch.image_width)
        if config.img_wh is not None:
            if type(config.img_wh) is int:
                self.img_wh = (config.img_wh, config.img_wh)
            assert self.img_wh[0]%32==0 and self.img_wh[1]%32==0, \
                'img_wh must both be multiples of 32!'
        self.threshold = config.threshold
        self.build_metas()
        self.n_views = config.n_views
        self.levels = config.levels # FPN levels
        self.depth_interval = config.depth_interval
        self.build_proj_mats()
        self.define_transforms()
        self.output_total = config.output_total
        prompt_dir = config.prompt_dir
        if prompt_dir != None:
            import json
            captions = json.load(open(prompt_dir))
        self.prompt_dir =captions
        
      
        
        
        
    def build_metas(self):
        self.metas = []
        with open(f'/workspace/project/dp_simple/CasMVSNet_pl/datasets/lists/dtu/{self.split}.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()]
        output_pkl = f'/workspace/project/dp_simple/CasMVSNet_pl/datasets/lists/dtu/{self.split}_abs.pkl'
        import pickle
        with open(output_pkl, 'rb') as f:
            self.output_pkl = pickle.load(f)
        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        outputs_total = {}
        for scan in self.output_pkl.keys():
            scan_index = scan.split('_')[0]
            if scan_index not in outputs_total:
                outputs_total[scan_index] = []
            outputs_total[scan_index].append(self.output_pkl[scan])
        for scan in outputs_total.keys():
            outputs_total[scan] = np.mean(np.array(outputs_total[scan]), axis=0)
            print(f"scan {scan} mean output: {outputs_total[scan]}")
        self.total_pkl = outputs_total


        light_idxs = list(range(7))

        pair_file = "Cameras/pair.txt"
        for scan in self.scans:
            with open(os.path.join(self.root_dir, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    

                    for light_idx in light_idxs:
                        output_key = f"{scan}_{ref_view}_{src_views[0]}_{src_views[1]}"
                        losses = self.output_pkl[output_key]
                        if abs(np.argmin(losses)-light_idx)>4 :
                            
                            self.metas += [(scan, ref_view,light_idx, src_views,int(np.argmin(losses)))]
                         
    def build_proj_mats(self):
        proj_mats = []
        for vid in range(49): # total 49 view ids
            if self.img_wh is None:
                proj_mat_filename = os.path.join(self.root_dir,
                                                 f'Cameras/train/{vid:08d}_cam.txt')
            else:
                proj_mat_filename = os.path.join(self.root_dir,
                                                 f'Cameras/{vid:08d}_cam.txt')
            intrinsics, extrinsics, depth_min = \
                self.read_cam_file(proj_mat_filename)
            if self.img_wh is not None: # resize the intrinsics to the coarsest level
                intrinsics[0] *= self.img_wh[0]/1600/4
                intrinsics[1] *= self.img_wh[1]/1200/4
            K = intrinsics
            R = extrinsics
            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_ls = []
            for l in reversed(range(self.levels)):
                proj_mat_l = np.eye(4)
                proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                intrinsics[:2] *= 2 # 1/4->1/2->1
                proj_mat_ls += [torch.FloatTensor(proj_mat_l)]
            # (self.levels, 4, 4) from fine to coarse
            proj_mat_ls = torch.stack(proj_mat_ls[::-1])
           
            proj_mats += [(proj_mat_ls, depth_min,K,R)]

        self.proj_mats = proj_mats

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

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32) # (1200, 1600)
        if self.img_wh is None:
            depth = cv2.resize(depth, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            depth_0 = depth[44:556, 80:720] # (512, 640)
        else:
            depth_0 = cv2.resize(depth, self.img_wh,
                                 interpolation=cv2.INTER_NEAREST)
        depth_1 = cv2.resize(depth_0, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)
        depth_2 = cv2.resize(depth_1, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)

        depths = {"level_0": torch.FloatTensor(depth_0),
                  "level_1": torch.FloatTensor(depth_1),
                  "level_2": torch.FloatTensor(depth_2)}
        
        return depths

    def read_mask(self, filename):
        mask = cv2.imread(filename, 0) # (1200, 1600)
       
        if self.img_wh is None:
            mask = cv2.resize(mask, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            mask_0 = mask[44:556, 80:720] # (512, 640)
        else:
            mask_0 = cv2.resize(mask, self.img_wh,
                                interpolation=cv2.INTER_NEAREST)
        mask_1 = cv2.resize(mask_0, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)
        mask_2 = cv2.resize(mask_1, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)

        masks = {"level_0": torch.BoolTensor(mask_0),
                 "level_1": torch.BoolTensor(mask_1),
                 "level_2": torch.BoolTensor(mask_2)}

        return masks

    def define_transforms(self):
        if self.split == 'train': # you can add augmentation here
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
        else:
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
        self.unpreprocess = T.Compose([
            T.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        ])
    
    def decode_batch(self, batch):
        imgs = batch['imgs']
        proj_mats = batch['proj_mats']
        depths = batch['depths']
        masks = batch['masks']
        init_depth_min = batch['init_depth_min']
        depth_interval = batch['depth_interval']
        return imgs, proj_mats, depths, masks, init_depth_min, depth_interval

    def __len__(self):
        return len(self.metas)


    


    def  __getitem__(self, idx):
       
        scan, ref_view,light_idx, src_views,target_light = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        # output_key = f"{scan}_{ref_view}_{src_views[0]}_{src_views[1]}"
        # if self.total_pkl:
        #     target_light = self.total_pkl[scan]
        #     target_light = np.argmin(target_light)
        # else:
        #     target_light = self.output_pkl[output_key]
        #     target_light = np.argmin(target_light)

        

        sample = {}
        imgs = []
        cams = []
        proj_mats = []
        target_imgs = []
        Ks = []
        Rs = []
        intensity_stats =[]
        prompt = str(np.random.choice(self.prompt_dir[scan][str(ref_view)],1)[0])
        change_light = target_light - light_idx
        sample['prompt'] = [f"modify the lightness of image to light_class_{target_light} style"]
        for i, vid in enumerate(view_ids):
        # NOTE that the id in image file names is from 1 to 49 (not 0~48)
        
            img_filename = os.path.join(self.root_dir,
                            f'Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png')
            target_filename = os.path.join(self.root_dir,
                            f'Rectified/{scan}_train/rect_{vid+1:03d}_{target_light}_r5000.png')
            mask_filename = os.path.join(self.root_dir,
                            f'Depths/{scan}/depth_visual_{vid:04d}.png')
            depth_filename = os.path.join(self.root_dir,
                            f'Depths/{scan}/depth_map_{vid:04d}.pfm')
    

            img = Image.open(img_filename)
            target_img = Image.open(target_filename)
            if self.img_wh is not None:
                img = img.resize(self.img_wh, Image.BILINEAR)
                target_img = target_img.resize(self.img_wh, Image.BILINEAR)

            img = self.transform(img)
            target_img = self.transform(target_img)
            imgs += [img]
            target_imgs += [target_img]

            proj_mat_ls, depth_min,K,R = self.proj_mats[vid]
            Ks += [K]
            Rs += [R]
        



            if i == 0:  # reference view
                
                sample['init_depth_min'] = torch.FloatTensor([depth_min])
                
                sample['masks'] = self.read_mask(mask_filename)
                for key in sample['masks']:
                    sample['masks'][key] = sample['masks'][key]
                sample['depths'] = self.read_depth(depth_filename)
                for key in sample['depths']:
                    sample['depths'][key] = sample['depths'][key]
                sample["depth"] = sample["depths"]["level_0"]
                ref_proj_inv = torch.inverse(proj_mat_ls)
            else:
                
                proj_mats += [proj_mat_ls @ ref_proj_inv]
            var, mean = torch.var_mean(img)
            intensity_stat = torch.stack([mean, var], dim=0)
            intensity_stats.append(intensity_stat)
    
    
        imgs = torch.stack(imgs) # (V, 3, H, W)
        target_imgs = torch.stack(target_imgs)
        proj_mats = torch.stack(proj_mats)[:,:,:3] # (V-1, self.levels, 3, 4) from fine to coarse
        
        imgs = self.unpreprocess(imgs)
        target_imgs = self.unpreprocess(target_imgs)
       
        Ks = np.stack(Ks)
        Rs = np.stack(Rs)
        sample['pose'] = Rs
        sample['K'] = Ks
        sample['images'] = imgs
        sample["intensity_stats"] = torch.stack(intensity_stats)
        sample['proj_mats'] = proj_mats
        sample['depth_interval'] = torch.FloatTensor([self.depth_interval])
        sample['scan_vid'] = (scan, ref_view)
        

        sample['target_imgs'] = target_imgs
        sample["bbox"] =torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)



        return sample





