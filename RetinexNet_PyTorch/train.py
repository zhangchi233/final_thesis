import os
import argparse
from glob import glob
import numpy as np
from model import RetinexNet

parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_id', dest='gpu_id', default="0",
                    help='GPU ID (-1 for CPU)')
parser.add_argument('--epochs', dest='epochs', type=int, default=200,
                    help='number of total epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8,
                    help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=96,
                    help='patch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--data_dir', dest='data_dir',
                    default='/root/autodl-tmp/mvs_training/dtu/Rectified',
                    help='directory storing the training data')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='/root/autodl-tmp/RetinexNet_PyTorch/ckpts/test2',
                    help='directory for checkpoints')

args = parser.parse_args()

def train(model):

    lr = args.lr * np.ones([args.epochs])
    lr[20:] = lr[0] / 10.0

    # train_low_data_names = glob(args.data_dir + '/data/our485/low/*.png') + \
    #                        glob(args.data_dir + '/data/syn/low/*.png')
    train_low_data_names = []
    train_high_data_names= []
    train_ref_data_names = []
    
    files = os.listdir(args.data_dir)
    for file in files:
        images = os.listdir(os.path.join(args.data_dir, file))
        path = os.path.join(args.data_dir, file)
        count = 100
        for image in images:
            if image.endswith('.png'):
                # randomly select on image from each set as ref
                ref_image = np.random.choice(images)
                rer_splits = ref_image.split('_')
                rer_splits[2] = str(6)
                ref_image = '_'.join(rer_splits)
                #index = int(ref_image.split('_')[2])
                image_split = image.split('_')
                image_split[2] = str(6)
                high_image = '_'.join(image_split)
                train_low_data_names.append(os.path.join(path, image))
                train_high_data_names.append(os.path.join(path, high_image))
                train_ref_data_names.append(os.path.join(path, ref_image))
            count -= 1
            if count == 0:
                break
            
    eval_low_data_names  = glob("/root/autodl-tmp/frames/case_0/*.jpg")
    eval_ref_data_names  = glob("/root/autodl-tmp/frames/test/*.jpg")
    eval_ref_data_names *= len(eval_low_data_names) // len(eval_ref_data_names)
    eval_low_data_names.sort()



    assert len(train_low_data_names) == len(train_high_data_names)
    assert len(train_low_data_names) == len(train_ref_data_names)
    assert len(eval_low_data_names) == len(eval_ref_data_names)
    print('Number of training data: %d' % len(train_low_data_names))


    model.train(train_low_data_names,
                train_high_data_names,
                train_ref_data_names,
                eval_low_data_names,
                eval_ref_data_names,

                batch_size=args.batch_size,
                patch_size=args.patch_size,
                epoch=10,
                lr=lr,
                vis_dir=args.vis_dir,
                ckpt_dir=args.ckpt_dir,
                eval_every_epoch=1,
                train_phase="Decom")

    model.train(train_low_data_names,
                train_high_data_names,
                train_ref_data_names,
                eval_low_data_names,
                eval_ref_data_names,

                batch_size=args.batch_size,
                patch_size=args.patch_size,
                epoch=10,
                lr=lr,
                vis_dir=args.vis_dir,
                ckpt_dir=args.ckpt_dir,
                eval_every_epoch=1,
                train_phase="Relight")


if __name__ == '__main__':
    if args.gpu_id != "-1":
        # Create directories for saving the checkpoints and visuals
        args.vis_dir = args.ckpt_dir + '/visuals/'
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.vis_dir):
            os.makedirs(args.vis_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        # Create the model

        model = RetinexNet().cuda()
        # Train the model
        model.device = "cuda"
        train(model)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError
