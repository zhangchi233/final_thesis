#!/bin/bash

# Train pix2pix script

python -m accelerate.commands.launch \
  /workspace/ViewDiff/viewdiff/train_pix2pix_debug.py \
  --finetune-config.io.pretrained_model_name_or_path timbrooks/instruct-pix2pix \
  --finetune-config.io.output_dir /workspace/ViewDiff/output_var_debug \
  --finetune-config.io.experiment_name class6 \
  --finetune-config.training.mixed_precision bf16 \
  --finetune-config.training.dataloader_num_workers 4 \
  --finetune-config.training.num_train_epochs 600 \
  --finetune-config.training.train_batch_size 3 \
  --finetune-config.training.dreambooth_prior_preservation_loss_weight -1 \
  --finetune_config.training.noise_prediction_type epsilon \
  --finetune_config.training.prob_images_not_noisy 0.25 \
  --finetune_config.training.max_num_images_not_noisy 2 \
  --finetune_config.training.validation_epochs 1 \
  --finetune_config.training.dreambooth_prior_preservation_every_nth -1 \
  --finetune-config.optimizer.learning_rate 1e-5 \
  --finetune-config.optimizer.vol_rend_learning_rate 1e-3 \
  --finetune-config.optimizer.vol_rend_adam_weight_decay 0.0 \
  --finetune-config.optimizer.gradient_accumulation_steps 1 \
  --finetune-config.optimizer.max_grad_norm 5e-3 \
  --finetune-config.cross_frame_attention.to_k_other_frames 2 \
  --finetune-config.cross_frame_attention.random_others \
  --finetune-config.cross_frame_attention.with_self_attention \
  --finetune-config.cross_frame_attention.use_temb_cond \
  --finetune-config.cross_frame_attention.mode pretrained \
  --finetune-config.cross_frame_attention.n_cfa_down_blocks 1 \
  --finetune-config.cross_frame_attention.n_cfa_up_blocks 1 \
  --finetune-config.cross_frame_attention.unproj_reproj_mode with_cfa \
  --finetune-config.cross_frame_attention.num_3d_layers 1 \
  --finetune-config.cross_frame_attention.dim_3d_latent 16 \
  --finetune-config.cross_frame_attention.dim_3d_grid 64 \
  --finetune-config.cross_frame_attention.n_novel_images 1 \
  --finetune-config.cross_frame_attention.vol_rend_proj_in_mode multiple \
  --finetune-config.cross_frame_attention.vol_rend_proj_out_mode multiple \
  --finetune-config.cross_frame_attention.vol_rend_aggregator_mode ibrnet \
  --finetune-config.cross_frame_attention.last_layer_mode zero-conv \
  --finetune_config.cross_frame_attention.vol_rend_model_background \
  --finetune_config.cross_frame_attention.vol_rend_background_grid_percentage 0.5 \
  --finetune-config.model.pose_cond_mode sa-ca \
  --finetune-config.model.pose_cond_coord_space absolute \
  --finetune-config.model.pose_cond_lora_rank 64 \
  --finetune-config.model.n_input_images 3 \
  --dataset-config.root-dir /workspace/mvs_training/dtu \
  --dataset-config.threshold 0.8 \
  --dataset-config.split train \
  --dataset-config.img_wh 512\
  --dataset-config.debug 0 \
  --validation-dataset-config.debug 0\
  --validation-dataset-config.root-dir /workspace/mvs_training/dtu \
  --validation-dataset-config.split val \
  --validation-dataset-config.threshold 0.8\

