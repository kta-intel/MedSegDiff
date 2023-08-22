#!/bin/bash

# python scripts/segmentation_sample.py --data_name ISIC --data_dir data/ISIC/Part1 --out_dir outputs --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 100 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5

# python scripts/segmentation_sample.py --data_name BRATS --data_dir data/MICCAI_BraTS2020_TrainingData --out_dir outputs --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 100 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5

python scripts/segmentation_inference.py --data_name BRATS --data_dir data/MICCAI_BraTS2020_TrainingData --out_dir outputs --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 100 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5