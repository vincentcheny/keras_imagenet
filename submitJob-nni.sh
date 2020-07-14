#!/bin/bash
#SBATCH --job-name=BOHB20h
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/imagenet-log/googlenet_bn-bohb-20h-1gpu-official.log
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --mem-per-cpu=10G
#SBATCH --nodelist=chpc-gpu003
#SBATCH --cpus-per-task=4

nnictl create --config config-bohb.yml -p 8083
sleep 26h