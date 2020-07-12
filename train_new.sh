#!/bin/bash
#SBATCH --job-name=imagenet
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/imagenet-log/googlenet_bn-dragonfly-test.log
#SBATCH --gres=gpu:1
#SBATCH --mem=10000

set -e

usage()
{
    echo
    echo "Usage: ./trash_new.sh <model_name>"
    echo
    echo "where <model_name> could be one of the following:"
    echo "    mobilenet_v2, resnet50, googlenet_bn, inception_v2,"
    echo "    efficientnet_b0, efficientnet_b1, efficientnet_b4"
    echo
}

if [ $# -ne 1 ]; then
    usage
    exit
fi

case $1 in
    mobilenet_v2 )
        python3 train.py --dropout_rate 0.2 --weight_decay 3e-5 \
                         --optimizer adam --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 1 mobilenet_v2
        ;;
    resnet50 )
        python3 train.py --dropout_rate 0.5 --weight_decay 2e-4 \
                         --optimizer adam --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 1 resnet50
        ;;
    googlenet_bn )
        python3 train-dragonfly.py --dropout_rate 0.4 --weight_decay 2e-4 \
                         --optimizer adam --batch_size 8 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 1 googlenet_bn
        ;;
    inception_v2 )
        python3 train.py --dropout_rate 0.4 --weight_decay 2e-4 \
                         --optimizer adam --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 1 inception_v2
        ;;
    inception_v2x )
        python3 train.py --dropout_rate 0.4 --weight_decay 2e-4 \
                         --optimizer adam --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 80 inception_v2x
        ;;
    inception_mobilenet )
        python3 train.py --dropout_rate 0.4 --weight_decay 2e-4 \
                         --optimizer adam --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 80 inception_mobilenet
        ;;
    efficientnet_b0 )
        python3 train.py --dropout_rate 0.2 --weight_decay 3e-5 \
                         --optimizer sgd --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-4 \
                         --epochs 1 efficientnet_b0
        ;;
    osnet )
        python3 train.py --dropout_rate 0.2 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 60 osnet
        ;;
    * )
        usage
        exit
esac
