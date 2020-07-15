#!/bin/bash
#SBATCH --job-name=BOHB40h
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/imagenet-log/googlenet_bn-bohb-40h-2gpu.log
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --nodelist=chpc-gpu002
#SBATCH --cpus-per-task=4

# singularity exec ../../cyliu/tensor_with_datasets python train.py  --iter_size 1 --lr_sched exp  googlenet_bn-2gpu-model-final.h5
singularity exec ../../cyliu/tensor_with_datasets nnictl create --config config-bohb.yml -p 8080
sleep 50h