import cv2
import numpy as np
import os

# Define the codec and create VideoWriter object

from argparse import ArgumentParser
def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--modified_dir', type=str,
                        default='/openbayes/input/input0/ViewDiff/viewdiff/results/dtu/image_modified/scan35',
                        help='root directory of dtu dataset')
    parser.add_argument('--original_file', type=str,
                        default='/openbayes/input/input0/mvs_training/dtu/Rectified/scan35_train',
                        help='root directory of dtu dataset')
    
    parser.add_argument('--output_dir', type=str,
                        default='/openbayes/home',
                        help='root directory of dtu dataset')
    parser.add_argument('--interval', type=int, default=3,
                        
                        help='which dataset to train/val')
    parser.add_argument('--gt_light', type=int, default=6,
                        help='which split to evaluate')
    parser.add_argument('--mix_light', type=int, default=0,
                        help='specify scan to evaluate (must be in the split)')
    parser.add_argument('--scan', type=str, default="scan35",
                        help='specify scan to evaluate (must be in the split)')
    
    args, _ = parser.parse_known_args()
    return args

args = get_opts()
scan = args.scan
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'comple_modified_{scan}.mp4', fourcc, 20.0, (640,512))
for vid in range(49):
    # Create a frame with random colors
    modiefied_img = f'{vid:04d}_class6.npy'
    modified_path = os.path.join(args.modified_dir,modiefied_img)

    img = np.load(modified_path)[0]
    print(img.shape)
    img = img.transpose(1,2,0)
    img = img*225
    img = np.clip(img, 0, 255)
    

# Convert the image data to 8-bit
    img.astype(np.uint8)

    frame = img.astype(np.uint8)
    
    # Write the frame into the file 'output.avi'
    out.write(frame)

# Release everything when the job is finished
out.release()


out = cv2.VideoWriter(f'output_mix_{scan}.mp4', fourcc, 20.0, (640,512))
for vid in range(49):
    # Create a frame with random colors
    if vid%args.interval==0:
        img_original = f'rect_{vid+1:03d}_{args.mix_light}_r5000.png'
    else:
        img_original = f'rect_{vid+1:03d}_{args.gt_light}_r5000.png'
    img_path = os.path.join(args.original_file,img_original)

    img = cv2.imread(img_path)


   
    frame = img
    
    # Write the frame into the file 'output.avi'
    out.write(frame)

# Release everything when the job is finished
out.release()



out = cv2.VideoWriter(f'output_refine__{scan}.mp4', fourcc, 20.0, (640,512))
for vid in range(49):
    # Create a frame with random colors
    if vid%args.interval==0:
        modiefied_img = f'{vid:04d}_class6.npy'
        modified_path = os.path.join(args.modified_dir,modiefied_img)

        img = np.load(modified_path)[0]

        img = img.transpose(1,2,0)
        
        img = img*225
        img = np.clip(img, 0, 255)
        img =img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
    else:
        img_original = f'rect_{vid+1:03d}_{args.gt_light}_r5000.png'
        img_path = os.path.join(args.original_file,img_original)

        img = cv2.imread(img_path)


   
    frame = img

    
    # Write the frame into the file 'output.avi'
    out.write(frame)

# Release everything when the job is finished
out.release()


