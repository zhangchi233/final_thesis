# %%


# %%
from models.mvsnet import CascadeMVSNet
from utils import load_ckpt
from inplace_abn import ABN
from losses import SL1Loss
model = CascadeMVSNet(n_depths=[8,32,48],
                      interval_ratios=[1.0,2.0,4.0],
                      num_groups=1,
                      norm_act=ABN).cuda()
load_ckpt(model, 'ckpts/_ckpt_epoch_10.ckpt')
model.eval()

# %%
from torch.utils.data import Dataset
from datasets.utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T

class DTUDataset(Dataset):
    def __init__(self, root_dir, split,model, n_views=3, levels=3, depth_interval=2.65,
                 img_wh=(512,640)):
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
        self.model = model.cuda()
        self.model.device = "cuda"
        self.loss = SL1Loss()
        
    def build_metas(self):
        self.metas = []
        with open(f'/root/autodl-tmp/project/dp_simple/CasMVSNet_pl/datasets/lists/dtu/{self.split}.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()]
        self.output_pkl = f'/root/autodl-tmp/project/dp_simple/CasMVSNet_pl/datasets/lists/dtu/{self.split}.pkl'

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
                    self.metas += [(scan, ref_view, src_views)]

                    # for light_idx in light_idxs:
                    #     
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

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_ls = []
            for l in reversed(range(self.levels)):
                proj_mat_l = np.eye(4)
                proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                intrinsics[:2] *= 2 # 1/4->1/2->1
                proj_mat_ls += [torch.FloatTensor(proj_mat_l)]
            # (self.levels, 4, 4) from fine to coarse
            proj_mat_ls = torch.stack(proj_mat_ls[::-1])
           
            proj_mats += [(proj_mat_ls, depth_min)]

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


    def runs(self):
        import pickle as pkl
        dictionary = {}
        from tqdm import tqdm
        print("length is",len(self.metas))

        for idx in tqdm(range(len(self.metas))):
            index = self.runonce(idx,dictionary)
            print(index)

        with open(self.output_pkl,'wb') as f:
            pkl.dump(dictionary,f)


    def runonce(self, idx,dictionary):
       
        scan, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        output_key = f"{scan}_{ref_view}_{src_views[0]}_{src_views[1]}"
        
        losses =[]
        for light_idx in range(7):
            sample = {}
            imgs = []
            cams = []
            proj_mats = []
            
            for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            
                img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png')
                mask_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}_train/depth_visual_{vid:04d}.png')
                depth_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}_train/depth_map_{vid:04d}.pfm')
        

                img = Image.open(img_filename)
                if self.img_wh is not None:
                    img = img.resize(self.img_wh, Image.BILINEAR)
                img = self.transform(img)
                imgs += [img]

                proj_mat_ls, depth_min = self.proj_mats[vid]
            



                if i == 0:  # reference view
                    
                    sample['init_depth_min'] = torch.FloatTensor([depth_min]).unsqueeze(0).to(self.model.device)
                    
                    sample['masks'] = self.read_mask(mask_filename)
                    for key in sample['masks']:
                        sample['masks'][key] = sample['masks'][key].unsqueeze(0).to(self.model.device)
                    sample['depths'] = self.read_depth(depth_filename)
                    for key in sample['depths']:
                        sample['depths'][key] = sample['depths'][key].unsqueeze(0).to(self.model.device)
                    sample["depth"] = sample["depths"]["level_0"]
                    ref_proj_inv = torch.inverse(proj_mat_ls)
                else:
                    
                    proj_mats += [proj_mat_ls @ ref_proj_inv]
       
        
            imgs = torch.stack(imgs) # (V, 3, H, W)
            proj_mats = torch.stack(proj_mats)[:,:,:3] # (V-1, self.levels, 3, 4) from fine to coarse
            
            
        
            sample['imgs'] = imgs.unsqueeze(0).to(self.model.device)
            sample['proj_mats'] = proj_mats.unsqueeze(0).to(self.model.device)
            sample['depth_interval'] = torch.FloatTensor([self.depth_interval]).unsqueeze(0).to(self.model.device)
            sample['scan_vid'] = (scan, ref_view)

            imgs, proj_mats, depths, masks, init_depth_min, depth_interval = self.decode_batch(sample)
            with torch.no_grad():
                results = self.model(imgs, proj_mats, init_depth_min, depth_interval)
                loss = self.loss(results, depths, masks)
                losses.append(loss.item())
        # save the minimum loss from list losses
        print(losses)
        index = np.argmin(np.array(losses))
        dictionary[output_key] =  index




            

        return index

# %%
dtu_data = DTUDataset(root_dir="/root/autodl-tmp/mvs_training/dtu",split="val",model=model)
dtu_data.runs()

# %%
dtu_data = DTUDataset(root_dir="/root/autodl-tmp/mvs_training/dtu",split="test",model=model)
dtu_data.runs()

# %%
dtu_data = DTUDataset(root_dir="/root/autodl-tmp/mvs_training/dtu",split="train",model=model)
dtu_data.runs()


