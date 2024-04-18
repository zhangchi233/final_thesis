import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    try:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
    except:
        mi = 0
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_
def normalize_depth(depth):
    """
    depth: (B,H, W)
    """
    # fill nan in torch tensor
    x = torch.nan_to_num(depth)
    mi = torch.min(x[x>0],dim =1) # get minimum positive depth (ignore background)
    ma = torch.max(x,dim = 1)
    
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    return x

def visualize_prob(prob, cmap=cv2.COLORMAP_BONE):
    """
    prob: (H, W) 0~1
    """
    x = (255*prob).cpu().numpy().astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_