import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import math
# import cv2

def random_crop(HR, LR, patch_size_lr, scale_factor,masks=None,depths=None): # HR: N*H*W
    _, _,_, h_hr, w_hr = HR.shape
    h_lr = h_hr // scale_factor
    w_lr = w_hr // scale_factor

    # obtain h and w from mask
    mask = masks['level_0'].cpu().numpy()

    
    for i in range(len(mask)):
        mask_i = mask[i]
        y_indices, x_indices = np.where(mask_i)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    if x_max - x_min < patch_size_lr:
        h_start_lr = x_min
        h_end_lr = x_max
    else:
        h_start_lr = random.randint(x_min, x_max - patch_size_lr)
        h_end_lr = h_start_lr + patch_size_lr
    if y_max - y_min < patch_size_lr:
        w_start_lr = y_min
        w_end_lr = y_max
    

    # h_start_lr = random.randint(5, h_lr - patch_size_lr - 5)
    # w_start_lr = random.randint(5, w_lr - patch_size_lr - 5)
    else:
       
        w_start_lr = random.randint(y_min, y_max - patch_size_lr)
        w_end_lr = w_start_lr + patch_size_lr

    h_start = h_start_lr * scale_factor
    h_end = h_end_lr * scale_factor
    w_start = w_start_lr * scale_factor
    w_end = w_end_lr * scale_factor

    HR = HR[:, :,:, h_start:h_end, w_start:w_end]
    LR = LR[:, :,:, h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    
    if depths is not None:
       for i in range(len(depths)):
            level_key = 'level_{}'.format(i)
            mask_level = masks[level_key]
            depth_level = depths[level_key]

            
            depths[level_key] = depth_level[:, h_start:h_end, w_start:w_end]
            masks[level_key] = mask_level[:, h_start:h_end, w_start:w_end]
            
            h_start =h_start //  2
            h_end = h_end// 2 
            w_start = w_start // 2
            w_end = w_end// 2
            
            

    return HR, LR, masks, depths

def add_noise(img, n_std):
    return img + np.random.normal(0, n_std, img.shape)

def add_light(img, light, *paras, mode):
    if mode == 'point':
        x0, y0, radius = paras
        light_res = np.zeros(3, radius, radius)
        for i in range(radius):
            for j in range(radius):
                light_res[0, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)
                light_res[1, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)
                light_res[2, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)

        light_res = np.clip(light_res + img[:, x0-radius//2:x0+1+radius//2, y0-radius//2:y0+1+radius//2, :], 0, 255)
        img[:, x0-radius//2:x0+1+radius//2, y0-radius//2:y0+1+radius//2, :] = light_res
    return img

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''

    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def ycbcr2rgb(ycbcr_img):
    ycbcr_img = ycbcr_img.numpy()
    in_img_type = ycbcr_img.dtype
    if in_img_type != np.uint8:
        ycbcr_img *= 255.
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.array([16, 128, 128])
    rgb_img = np.zeros(ycbcr_img.shape)
    for x in range(ycbcr_img.shape[0]):
        for y in range(ycbcr_img.shape[1]):
            rgb_img[x, y, :] = np.maximum(0, np.minimum(255,np.round(np.dot(mat_inv, ycbcr_img[x, y, :] - offset) * 255.0)))
    return torch.from_numpy(np.ascontiguousarray(rgb_img.astype(np.float32)/255))
