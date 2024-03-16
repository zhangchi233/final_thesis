export CO3DV2_DATASET_ROOT="/root/autodl-tmp/mvs_training/dtu"
python -m viewdiff.data.dtu.generate_blip2_captions --dataset-config.root_dir $CO3DV2_DATASET_ROOT --output_file  $CO3DV2_DATASET_ROOT/co3d_blip2_captions.json \
