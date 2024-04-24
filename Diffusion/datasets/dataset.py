import os
import PIL.Image
import torch
import torch.utils.data
import PIL
from PIL import Image
import re
import numpy as np
from datasets.data_augment import PairCompose, PairRandomCrop, PairToTensor,adjust_gamma

from torch.utils.data import Dataset

from .utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
import sys

from torchvision import transforms as T
class TrainTestDtuDataset(Dataset):    
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        if not self.config.data.single_frame:
            train_dataset = DTUDataset(self.config.data.root_dir,split='train',n_views=3,levels=3,depth_interval=2.65,
                                       split_path=self.config.data.split_path,
                                        img_wh=None
                                                )
            val_dataset = DTUDataset(self.config.data.root_dir,split='val',n_views=3,levels=3,depth_interval=2.65,
             split_path=self.config.data.split_path,
                                            )
        else:
            train_dataset = Dtu(self.config.data.root_dir,patch_size=self.config.data.patch_size,train=True)
            val_dataset = DTUDataset("/root/autodl-tmp/mvs_training/dtu",split='val',n_views=3,levels=3,depth_interval=2.65,
             split_path=self.config.data.split_path,
                                            )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True,)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True,
                                                 )

        return train_loader, val_loader
class DTUDataset(Dataset):
    def __init__(self, root_dir, split,split_path=None, n_views=3, levels=3, depth_interval=2.65,
                 img_wh=None):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train", "val" or "test"!'
        self.img_wh = img_wh
        if img_wh is not None:
            assert img_wh[0]%32==0 and img_wh[1]%32==0, \
                'img_wh must both be multiples of 32!'
        self.build_metas()
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.depth_interval = depth_interval
        self.build_proj_mats()
        self.define_transforms()

    def build_metas(self):
        self.metas = []
        with open(f'/root/autodl-tmp/project/dp_simple/CasMVSNet_pl/datasets/lists/dtu/{self.split}.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
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
                        self.metas += [(scan, light_idx, ref_view, src_views)]

    def build_proj_mats(self):
        proj_mats = []
        for vid in range(49): # total 49 view ids
            if self.img_wh is None:
                proj_mat_filename = os.path.join(self.root_dir,
                                                 f'Cameras/train/{vid:08d}_cam.txt')
            else:
                proj_mat_filename = os.path.join(self.root_dir,
                                                 f'Cameras/train/{vid:08d}_cam.txt')
            intrinsics, extrinsics, depth_min = \
                self.read_cam_file(proj_mat_filename)
            if self.img_wh is not None: # resize the intrinsics to the coarsest level
                intrinsics[0] *= self.img_wh[0]/640
                intrinsics[1] *= self.img_wh[1]/512

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_ls = []
            for l in reversed(range(self.levels)):
                proj_mat_l = np.eye(4)
                proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                intrinsics[:2] *= 2 # 1/4->1/2->1
                proj_mat_ls += [torch.FloatTensor(proj_mat_l)]
            # (self.levels, 4, 4) from fine to coarse
            proj_mat_ls = torch.stack(proj_mat_ls[::-1])
            cam = self.load_cam(open(proj_mat_filename, "r"))
            proj_mats += [(proj_mat_ls, depth_min,cam)]

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
        if self.img_wh is not None:
            depth = cv2.resize(depth, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            depth_0 = depth[44:556, 80:720] # (512, 640)
            
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
        if self.img_wh is not None:
            mask = cv2.resize(mask, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            mask_0 = mask[44:556, 80:720] # (512, 640)
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
        
        self.transform = T.Compose([T.ToTensor(),
                                        # T.Normalize(mean=[0.485, 0.456, 0.406], 
                                        #             std=[0.229, 0.224, 0.225]),
                                       ])
        self.transform2 = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                     std=[0.229, 0.224, 0.225])])
       
            
        self.unpreprocess = T.Compose([
            T.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        ])
    def load_cam(self, file, interval_scale=1):
        """ read camera txt file """
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = 256
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0
        if self.img_wh is not None:
            cam[1][0] *= self.img_wh[0]/1600
            cam[1][1] *= self.img_wh[1]/1200
        return cam
        

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        sample = {}
        scan, light_idx, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        cams = []
        proj_mats = [] # record proj mats between views
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            
            img_filename = os.path.join(self.root_dir,
                            f'Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png')
            mask_filename = os.path.join(self.root_dir,
                            f'Depths/{scan}/depth_visual_{vid:04d}.png')
            depth_filename = os.path.join(self.root_dir,
                            f'Depths/{scan}/depth_map_{vid:04d}.pfm')
        

            img = Image.open(img_filename)
            if self.img_wh is not None:
            
                img = img.resize(self.img_wh, Image.BILINEAR)
                
            img = self.transform(img)
            
            imgs += [img]

            proj_mat_ls, depth_min,cam = self.proj_mats[vid]
            cams += [cam]



            if i == 0:  # reference view
                sample['init_depth_min'] = torch.FloatTensor([depth_min])
                
                sample['masks'] = self.read_mask(mask_filename)
                sample['depths'] = self.read_depth(depth_filename)
                sample["depth"] = sample["depths"]["level_0"]
                ref_proj_inv = torch.inverse(proj_mat_ls)
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]
        cams = np.stack(cams)
        cams = torch.tensor(cams)
        
        imgs = torch.stack(imgs) # (V, 3, H, W)
        proj_mats = torch.stack(proj_mats)[:,:,:3] # (V-1, self.levels, 3, 4) from fine to coarse
        

        
        

     
        
        
        sample['cams'] = cams
        sample['imgs_gt'] = imgs
        sample["imgs"] = imgs
        sample['proj_mats'] = proj_mats
        sample['depth_interval'] = torch.FloatTensor([self.depth_interval])
        sample['scan_vid'] = (scan, ref_view)

        return sample

    

class Dtu(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size,train=True):
        super().__init__()
        
        self.dir = dir
        self.train = train
        
   
        image_list = []
        scans = os.listdir(dir)
        
        
        scans = [scan for scan in scans if os.path.isdir(os.path.join(dir,scan))]
        for scan in scans:
            images = os.listdir(os.path.join(dir,scan))
            for image in images:
                image_list.append(os.path.join(scan,image))
       

        self.input_names = image_list
       
        self.patch_size = patch_size
        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')
        
        # imagename = os.path.join(self.dir, input_name), the image name format is rect_{index}_{class}_r5000.png
        # imgid = index+class
       
        imag_name = input_name.split("/")[-1]
        
        img_id = imag_name.split('_')[1]+input_name.split('_')[2]
        input_img = Image.open(os.path.join(self.dir, input_name)) 
        sample = {}


        gt_img = input_img
        input_img, gt_img = self.transforms(input_img, gt_img)
        import numpy as np
        gamma = np.random.uniform(2, 5)
        input_img = T.functional.adjust_gamma(input_img, gamma)
        sigmoidB = np.random.uniform(0, 1)
        sigmoid = np.sqrt(sigmoidB*(25/255)**2)
        
        gaussian_noise = torch.normal(input_img, sigmoid,)
        imgs_noisy = input_img + gaussian_noise
        imgs_noisy = torch.clamp(imgs_noisy, 0, 1)
        input_img = imgs_noisy
        
        sample["imgs"] = input_img
        sample["imgs_gt"] = gt_img
        sample["scan_vid"] = input_name.split("/")[-2]+img_id

        return sample
    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)




class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):

        train_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'train'),
                                          patch_size=self.config.data.patch_size,
                                          filelist='{}_train.txt'.format(self.config.data.train_dataset))
        val_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'val'),
                                        patch_size=self.config.data.patch_size,
                                        filelist='{}_val.txt'.format(self.config.data.val_dataset), train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()

        self.dir = dir
        self.train = train
        self.file_list = filelist
        self.train_list = os.path.join(dir, self.file_list)
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('low', 'high') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')
        gt_name = self.gt_names[index].replace('\n', '')
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        gt_img = Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)

        input_img, gt_img = self.transforms(input_img, gt_img)

        return torch.cat([input_img, gt_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
