DTU_DIR="/openbayes/input/input0/mvs_training/dtu"

python train.py \
   --dataset_name dtu \
   --root_dir $DTU_DIR \
   --num_epochs 16 --batch_size 4 \
   --depth_interval 2.65 --n_depths 8 32 48 --interval_ratios 1.0 2.0 4.0 \
   --optimizer adam --lr 1e-3 --lr_scheduler  steplr\
   --exp_name exp_dark \
