#!/bin/bash
#SBATCH --job-name=DF10h
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/imagenet-log/googlenet_bn-dragonfly-10h-1gpu-official.log
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --mem-per-cpu=10G
#SBATCH --nodelist=chpc-gpu002
#SBATCH --cpus-per-task=4

singularity exec ../../cyliu/tensor_with_datasets python train-dragonfly.py  --iter_size 1 --lr_sched exp  googlenet_bn-model-final.h5