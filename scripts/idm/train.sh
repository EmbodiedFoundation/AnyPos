#!/bin/bash
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
accelerate launch \\
 --main_process_port=29344 train_idm.py \\
 --model_name direction-aware \\
 --learning_rate 5e-4 \\
 --use_normalization \\
 --use_transform \\
 --batch_size 32 \\
 --num_iterations 72000 \\
 --eval_interval 4000 \\
 --num_workers 32 \\
 --prefetch_factor 4 \\
 --dataset_path <train data dir> \\
 --test_dataset_path <test data dir> \\
 --save_dir <save data dir> \\
 --dinov2_name facebook/dinov2-with-register-base \\
 --wandb_mode offline \\
 --run_name $timestamp > train-$timestamp.log 2>&1 &