#!/bin/bash
#SBATCH --job-name=BOHB90h
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/imagenet-log/googlenet_bn-2gpu-500classes-90h-bohb.log
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --nodelist=chpc-gpu002
#SBATCH --cpus-per-task=6

# singularity exec ../../cyliu/tensor_with_datasets python train.py  --iter_size 1 --lr_sched exp  googlenet_bn-2gpu-model-final.h5
singularity exec ../../cyliu/tensor_with_datasets nnictl create --config /lustre/project/EricLo/chen.yu/keras_imagenet/config-bohb.yml -p 8083
sleep 4d