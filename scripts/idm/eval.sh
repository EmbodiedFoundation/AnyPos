#!/bin/bash
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")

accelerate launch \\
 --main_process_port=29344 eval_qpos.py \\
 --load_from <checkpoint> \\
 --model_name dino_with_split \\
 --learning_rate 5e-4 \\
 --use_normalization \\
 --use_transform \\
 --batch_size 8 \\
 --num_iterations 72000 \\
 --eval_interval 4000 \\
 --num_workers 32 \\
 --prefetch_factor 4 \\
 --dataset_path <test data dir> \\
 --test_dataset_path <test data dir> \\
 --save_dir <save dir> \\
 --learning_dim 0,1,2,3,4,5,7,8,9,10,11,12 \\
 --dinov2_name facebook/dinov2-with-register-base \\
 --wandb_mode offline \\
 --run_name $timestamp  > eval-$timestamp.log 2>&1