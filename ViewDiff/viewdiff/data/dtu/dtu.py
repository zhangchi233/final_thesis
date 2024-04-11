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

import cv2
import numpy as np
import random
def filter_flare(image_path):
    # Load the lens flare image
    flare_image_path = image_path
    flare_image = cv2.imread(flare_image_path)

    # Since the flare is brighter than the surroundings, we can filter it out using thresholding
    # Convert the image to grayscale

    gray_flare = cv2.cvtColor(flare_image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image to reduce noise
    
    
    # Use a threshold to create a mask where the bright regions (presumably the flare) are white
    _, flare_mask = cv2.threshold(gray_flare, 20, 255, cv2.THRESH_BINARY)

    # Apply the mask to get only the flare
    flare_only = cv2.bitwise_and(flare_image, flare_image, mask=flare_mask)

    return flare_only
   
# Function to add lens flare at random position with random intensity and size
def add_lens_flare(image_path, flare_path):
    # Read the base image and the flare image
    image = cv2.imread(image_path)

    flare = filter_flare(flare_path)
    
    # resize the flare image to the size of the base image
    intensity =1
    flare = cv2.resize(flare, (image.shape[1], image.shape[0]))
    # calculate the bounding box of the flare
    
    

    # Choose random position for the flare

   
    resized_flare =flare
    
    x_pos = random.randint(0, image.shape[1] - resized_flare.shape[1])
    y_pos = random.randint(0, image.shape[0] - resized_flare.shape[0])

    # Resize flare to a random scale
    

    # Choose random intensity for the flare
    
   
    resized_flare = (resized_flare).astype(np.uint8)
    

    # Create an overlay with the same size as the image
    overlay = np.zeros_like(image, dtype=np.uint8)
   
    # Place the resized flare on the overlay
    overlay[:,:] = resized_flare

    

    
    # Blend the overlay with the image using the flare alpha channel as mask
    resized_flare = cv2.cvtColor(resized_flare, cv2.COLOR_BGR2BGRA)
    # make the the dark regions of the flare transparent
    # the channel 3 of the flare image is the alpha channel, should be proportional to the brightness of the flare
    resized_flare[:,:,3] = cv2.cvtColor(resized_flare, cv2.COLOR_BGR2GRAY)

    
    

    # the transparency of the flare should be ratio between the flare and the image
    

    alpha_mask = resized_flare[:,:,3] / 255.0
    alpha_mask = alpha_mask
    alpha_mask = np.clip(alpha_mask, 0, 1)
    
    # the black regions of the flare should be transparent
    # and 

    # the transparency of the flare should be higher in the dark regions
    


    for c in range(0, 3):
        image[y_pos:y_pos+resized_flare.shape[0], x_pos:x_pos+resized_flare.shape[1], c] =  (alpha_mask * overlay[:,:, c] +
                                                                                               (1-alpha_mask) * image[:,:, c])


    # Save the result
    
    # resize the mask to the target size
    
    mask = alpha_mask>0
    mask = mask.astype(np.uint8)
    
    return image,mask







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

    image_width: Optional[int] = 512
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

    root_dir: str =f"/{ROOTDIR}/mvs_training/dtu/"
    target_light = 6
    n_views:int=3 
    levels:int=3 
    depth_interval:int =2.65
    img_wh:int=None
    abs_error:Optional[str] ="abs"
    output_total:Optional[bool]=False
    threshold: Optional[int] = 0.8
    prompt_dir: Optional[str] = f"/{ROOTDIR}/mvs_training/dtu/co3d_blip2_captions_final.json"
    debug: Optional[int] = 0
    light_strength: int =  200
    light_gamma: int =  1.2
    dataset_id: str = "eth3d"
   

class DTUDataset(Dataset):
    def __init__(self, config: DTUConfig):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """

        self.root_dir = config.root_dir
        self.split = config.split
        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train", "val" or "test"!'
        
        self.dataset_id = config.dataset_id
        
        self.light_class = config.target_light
        self.img_wh = (config.batch.image_width, config.batch.image_height)
        if config.img_wh is not None:
            if type(config.img_wh) is int:
                self.img_wh = (config.img_wh, config.img_wh)
            assert self.img_wh[0]%32==0 and self.img_wh[1]%32==0, \
                'img_wh must both be multiples of 32!'
        self.debug = config.debug
        self.read_bbox()
        self.threshold = config.threshold
        self.build_metas()
        self.n_views = config.n_views
        self.levels = config.levels # FPN levels
        self.depth_interval = config.depth_interval
        self.build_proj_mats()
        self.define_transforms()
        self.output_total = config.output_total
        prompt_dir = None
        if prompt_dir != None:
            import json
            captions = json.load(open(prompt_dir))
            self.prompt_dir =captions
        
      
        
        
        
    def build_metas(self):
        self.metas = []
        import pickle
        if self.dataset_id == "dtu":
            if self.debug==1:
                self.split = "train"
            with open(f'/{ROOTDIR}/project/dp_simple/CasMVSNet_pl/datasets/lists/dtu/{self.split}.txt') as f:
                self.scans = [line.rstrip() for line in f.readlines()]
            output_pkl = f'/{ROOTDIR}/project/dp_simple/CasMVSNet_pl/datasets/lists/dtu/{self.split}_abs.pkl'
        
            
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
                            if self.split=="train":
                                if self.debug ==1:
                                    if scan == "scan105":
                                        self.metas += [(scan, ref_view,light_idx, src_views,int(np.argmin(losses)))]
                                else:                               
                                    self.metas += [(scan, ref_view,light_idx, src_views,int(np.argmin(losses)))]
                            elif self.split!="train":
                                if light_idx!=0 or scan !="scan106":
                                    continue
                                else:
                                    self.metas += [(scan, ref_view,light_idx, src_views,int(np.argmin(losses)))]
                               
        
        elif self.dataset_id == "eth3d":
            split_path = os.path.join(self.root_dir, f'{self.split}.txt')
            bbox_path = os.path.join(self.root_dir, 'bbox.pkl')

            self.metas = []
            import pickle as pkl
            with open(split_path) as f:
                self.scans = [line.rstrip() for line in f.readlines()]
            with open(bbox_path, 'rb') as f:
                self.total_pkl = pickle.load(f)
                
            for scan in self.scans:
                pair_file = "pair.txt"
                print(scan,self.root_dir, pair_file)
                pair_file = os.path.join(self.root_dir, scan, pair_file)
                with open(pair_file,"r") as f:
                    num_viewpoint = int(f.readline())
                    print(f"num_viewpoint: {num_viewpoint}")
                    for _ in range(num_viewpoint):
                        ref_view = int(f.readline().rstrip())
                        src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                        
                        self.metas += [(scan, ref_view,None, src_views,None)]
    


            

            
                         
    def build_proj_mats(self):
        proj_mats = []
        if self.dataset_id == "dtu":
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
        elif self.dataset_id == "eth3d":
            proj_mats = []

            img_scan_whs = {}
        
            for scan in self.scans:
                target_filename = os.path.join(self.root_dir,
                                    f'{scan}/images/{0:08d}.jpg')
                target_img = Image.open(target_filename)
                img_scan_whs[scan] = target_img.size[::-1]
            self.img_scan_whs = img_scan_whs    
            
       
            for meta in self.metas:
                scan, ref_view, _, src_views, _ = meta
                print("img_scan_whs; ",img_scan_whs)
                camera_file = os.path.join(self.root_dir, scan, f'cams_1/{ref_view:08d}_cam.txt')
                intrinsics, extrinsics, depth_min,depth_max = self.read_cam_file(camera_file)
                


                if self.img_wh is not None: # resize the intrinsics to the coarsest level
                    width = img_scan_whs[scan][1]
                    height = img_scan_whs[scan][0]
                    print(width,height)
                    intrinsics[0] *= self.img_wh[0]/width/4
                    intrinsics[1] *= self.img_wh[1]/height/4

                    # intrinsics[0] *= self.img_wh[0]/6223
                    # intrinsics[1] *= self.img_wh[1]/4146


                K = intrinsics
                R = extrinsics
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
        return intrinsics, extrinsics, depth_min,float(lines[11].split()[1])

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
                            interpolation=cv2.INTER_NEAREST)   # 
        mask_2 = cv2.resize(mask_1, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)

        mask_3 = cv2.resize(mask_2, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)


        masks = {"level_0": torch.BoolTensor(mask_0),
                 "level_1": torch.BoolTensor(mask_1),
                 "level_2": torch.BoolTensor(mask_2),
                 "level_3": torch.BoolTensor(mask_3)}

        return masks
    def read_bbox(self):
        import pickle as pkl
        bbox_path = os.path.join(self.root_dir,"bbox.pkl")
        with open(bbox_path, 'rb') as f:
            bbox = pkl.load(f)
        self.bbox = bbox


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
        if self.dataset_id == "eth3d":
            scan, ref_view,light_idx, src_views,target_light = self.metas[idx]
            task = np.random.choice([#"different albedo",
                                     #"different dark",
                                     "flare"
                                    #"overexposed","shadow"
                                     ],1)
            
            view_ids = [ref_view] + src_views[:self.n_views-1]
            sample = {}
            imgs = []
            cams = []
            proj_mats = []
            target_imgs = []
            Ks = []
            Rs = []
            intensity_stats =[]
            index = np.random.permutation(np.array([0,1,1]))

            x_min = self.bbox[f"{scan}"]["x_min"]
            x_max = self.bbox[f"{scan}"]["x_max"]
            y_min = self.bbox[f"{scan}"]["y_min"]
            y_max = self.bbox[f"{scan}"]["y_max"]
            z_min = self.bbox[f"{scan}"]["z_min"]
            z_max = self.bbox[f"{scan}"]["z_max"]

            sample['prompt'] = [f"modify the images for task {task}"]
            sample["small_mask"]=[]
            for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
                if index[i]:
                    img_filename = os.path.join(self.root_dir,
                                    f'{scan}/images/{vid:08d}.jpg')
                else:
                    if task == "different albedo":
                        img_filename = os.path.join(self.root_dir,
                                    f'{scan}/images_albedo/different_albedo_{vid:08d}.jpg')
                    elif task == "flare":
                        
                        img_filename = os.path.join(self.root_dir,
                                    f'{scan}/images/{vid:08d}.jpg')
                        output_path = os.path.join(self.root_dir,
                                    f'{scan}/images_dark/different_albedo_{vid:08d}.jpg')
                        flare_file_path = "/root/autodl-tmp/lens_flare"
                        files = os.listdir(flare_file_path)
                        flare_file = random.choice(files)
                        flare_file_path = os.path.join(flare_file_path,flare_file)


                    elif task == "overexposed":
                        img_filename = os.path.join(self.root_dir,
                                    f'{scan}/images/{vid:08d}.jpg')
                        # img_filename = os.path.join(self.root_dir,
                        #             f'{scan}/images_shadow/different_shadow_{vid:08d}.jpg')
                    else:
                        img_filename = os.path.join(self.root_dir,
                                    f'{scan}/images/{vid:08d}.jpg')
                        # img_filename = os.path.join(self.root_dir,
                        #             f'{scan}/images_shadow/different_dark_{vid:08d}.jpg')


                target_filename = os.path.join(self.root_dir,
                                    f'{scan}/images/{vid:08d}.jpg')
                # mask_filename = os.path.join(self.root_dir,
                #                 f'Depths/{scan}/depth_visual_{vid:04d}.png')
                # depth_filename = os.path.join(self.root_dir,
                #                 f'Depths/{scan}/depth_map_{vid:04d}.pfm')
                sample["small_mask"].append(torch.zeros(self.img_wh[0]//8,self.img_wh[1]//8))
                if task == "flare" and not index[i]:
                    img,small_mask = add_lens_flare(img_filename,flare_file_path)
                    
                    # resize the mask to the target size
                    
                    small_mask = cv2.resize(small_mask, (self.img_wh[0]//8,self.img_wh[1]//8), interpolation=cv2.INTER_NEAREST)
                    

                    #small_mask = cv2.resize(small_mask, (self.img_wh[0]//8,self.img_wh[1]//8), interpolation=cv2.INTER_NEAREST)
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    sample["small_mask"][-1] = torch.tensor(small_mask).float()

                    
                    

                else:
                    img = Image.open(img_filename)
                target_img = Image.open(target_filename)

                
                if self.img_wh is not None:
                    
                    img = img.resize(self.img_wh, Image.BILINEAR)
                    target_img = target_img.resize(self.img_wh, Image.BILINEAR)
               
                # if not index[i]:
                #     if task =="overexposed":
                #         # increase contrast and the image looks
                #         img = np.array(img)
                #         alpha = np.random.uniform(1.5, 2)
                #         beta = np.random.uniform(10, 15)
                #         img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                #     elif task =="shadow":
                #         img = np.array(img)
                #         alpha = np.random.uniform(0.5, 0.75)
                #         beta = np.random.uniform(-15, -10)
                #         img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                img = self.transform(img)
                target_img = self.transform(target_img)
                imgs += [img]
                target_imgs += [target_img]

                proj_mat_ls, depth_min,K,R = self.proj_mats[idx]
                Ks += [K]
                Rs += [R]
            



                if i == 0:  # reference view
                    
                    
                    sample['init_depth_min'] = torch.FloatTensor([depth_min])
                    
                    
                    ref_proj_inv = torch.inverse(proj_mat_ls)
                else:
                    # small_mask = self.read_mask(mask_filename)["level_0"]
                    # small_wh = (self.img_wh[0]/8,self.img_wh[1]/8)
                    # sample["small_mask"] = interpolate(small_mask[None,None].float(), small_wh, mode='nearest')[0,0].byte()
                    
                    proj_mats += [proj_mat_ls @ ref_proj_inv]
                var, mean = torch.var_mean(img)
                intensity_stat = torch.stack([mean, var], dim=0)
                intensity_stats.append(intensity_stat)
        
            sample["small_mask"] = torch.stack(sample["small_mask"])
            sample["small_mask"] = sample["small_mask"].unsqueeze(1)
            sample["small_mask"] = sample["small_mask"].bool()
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

            sample["bbox"] =torch.tensor([[x_min,y_min,z_min], 
                                        [x_max,y_max,z_max]], dtype=torch.float32)




            return sample




        else:
            scan, ref_view,light_idx, src_views,target_light = self.metas[idx]
            # use only the reference view and first nviews-1 source views
            # shuffle the source views

            view_ids = [ref_view] + src_views[:self.n_views-1]
            light_input = np.random.choice(7,3)
            
            input_lights=light_input
            target_light = target_light



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
           

            x_min = self.bbox[f"{scan}_train"]["x"]["min"]
            x_max = self.bbox[f"{scan}_train"]["x"]["max"]
            y_min = self.bbox[f"{scan}_train"]["y"]["min"]
            y_max = self.bbox[f"{scan}_train"]["y"]["max"]
            z_min = self.bbox[f"{scan}_train"]["z"]["min"]
            z_max = self.bbox[f"{scan}_train"]["z"]["max"]

            sample['prompt'] = [f"modify the lightness of image to light_class_{light_idx} style"]
            for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            
                img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}_train/rect_{vid+1:03d}_{input_lights[i]}_r5000.png')
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
                # if input_lights[i] != target_light:
                #     # make image brighter or darker
                #     img = np.array(img)
                #     if input_lights[i] > target_light:
                #         # increase contrast and the image looks 
                #         alpha = np.random.uniform(1.5, 2)
                #         beta = np.random.uniform(10, 15)
                #         img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                #     else:
                #         alpha = np.random.uniform(0.5, 0.75)
                #         beta = np.random.uniform(-15, -10)
                #         img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                        
                    

                img = self.transform(img)
                target_img = self.transform(target_img)
                imgs += [img]
                target_imgs += [target_img]

                proj_mat_ls, depth_min,K,R = self.proj_mats[vid]
                Ks += [K]
                Rs += [R]
            



                if i == 0:  # reference view
                    sample["small_mask"]=[]
                    
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
                    # small_mask = self.read_mask(mask_filename)["level_0"]
                    # small_wh = (self.img_wh[0]/8,self.img_wh[1]/8)
                    # sample["small_mask"] = interpolate(small_mask[None,None].float(), small_wh, mode='nearest')[0,0].byte()
                    
                    proj_mats += [proj_mat_ls @ ref_proj_inv]
                var, mean = torch.var_mean(img)
                intensity_stat = torch.stack([mean, var], dim=0)
                intensity_stats.append(intensity_stat)
        
        
            imgs = torch.stack(imgs) # (V, 3, H, W)
            
            target_imgs = torch.stack(target_imgs)
            proj_mats = torch.stack(proj_mats)[:,:,:3] # (V-1, self.levels, 3, 4) from fine to coarse
            
            imgs = self.unpreprocess(imgs)
            target_imgs = self.unpreprocess(target_imgs)
            
            
            img_mask = (imgs-target_imgs).abs().mean(1,keepdim=True).repeat(1,3,1,1)



        
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
            small_mask = sample["masks"]["level_0"]
            small_wh = (80,64)
            sample["class"] = torch.tensor([light_idx])
            sample["small_mask"] = sample["masks"]["level_3"]

            sample["bbox"] =torch.tensor([[x_min,y_min,z_min], 
                                        [x_max,y_max,z_max]], dtype=torch.float32)




            return sample





