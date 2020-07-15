#!/bin/bash
#SBATCH --job-name=DF40h
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/imagenet-log/googlenet_bn-dragonfly-40h-2gpu.log
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --nodelist=chpc-gpu002
#SBATCH --cpus-per-task=6

singularity exec ../../cyliu/tensor_with_datasets python train-dragonfly.py  --iter_size 1 --lr_sched exp  googlenet_bn-2gpu-model-final.h5