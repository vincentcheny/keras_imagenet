"""train.sh

This script is used to train the ImageNet models.
"""

import numpy as np
from utils.utils import fix_randomness
fix_randomness()
from tensorflow.python.client import device_lib
NUM_GPU = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(list(range(NUM_GPU)))[1:-1].replace(" ", "") # 3 => "0,1,2"
import time
import argparse

import tensorflow as tf

from config import config
from utils.utils import config_keras_backend, clear_keras_session
from utils.dataset import get_dataset
from models.models import get_batch_size
from models.models import get_iter_size
from models.models import get_lr_func
from models.models import get_initial_lr
from models.models import get_final_lr
from models.models import get_weight_decay
from models.models import get_optimizer
from models.models import get_training_model
from models.models import get_customize_lr_callback

import nni


DESCRIPTION = """For example:
$ python3 train.py --dataset_dir  ${HOME}/data/ILSVRC2012/tfrecords \
                   --dropout_rate 0.4 \
                   --optimizer    adam \
                   --epsilon      1e-1 \
                   --label_smoothing \
                   --batch_size   32 \
                   --iter_size    1 \
                   --lr_sched     exp \
                   --initial_lr   1e-2 \
                   --final_lr     1e-5 \
                   --weight_decay 2e-4 \
                   --epochs       60 \
                   googlenet_bn
"""
SUPPORTED_MODELS = (
    '"mobilenet_v2", "resnet50", "googlenet_bn", "inception_v2", '
    '"inception_v2x", "inception_mobilenet", "efficientnet_b0", '
    '"efficientnet_b1", "efficientnet_b4", "osnet" or just specify '
    'a saved Keras model (.h5) file')


def train(model_name, dropout_rate, optim_name, epsilon,
          label_smoothing, use_lookahead, batch_size, iter_size,
          lr_sched, initial_lr, final_lr,
          weight_decay, epochs, dataset_dir):
    """Prepare data and train the model."""
    batch_size   = get_batch_size(model_name, batch_size)
    iter_size    = get_iter_size(model_name, iter_size)
    initial_lr   = get_initial_lr(model_name, initial_lr)
    final_lr     = get_final_lr(model_name, final_lr)
    optimizer    = get_optimizer(model_name, optim_name, initial_lr, epsilon)
    weight_decay = get_weight_decay(model_name, weight_decay)

    # get training and validation data
    ds_train = get_dataset(dataset_dir, 'train', batch_size)
    ds_valid = get_dataset(dataset_dir, 'validation', batch_size)

    # build model and do training
    model = get_training_model(
        model_name=model_name,
        dropout_rate=dropout_rate,
        optimizer=optimizer,
        label_smoothing=label_smoothing,
        use_lookahead=use_lookahead,
        iter_size=iter_size,
        weight_decay=weight_decay,
        gpus=NUM_GPU)
    

    class PrintAccOnEpochEnd(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            nni.report_intermediate_result(logs.get('acc'))
            print(f"Epoch{epoch} ends with acc:{logs.get('acc')}")


    NUM_DISTRIBUTE = NUM_GPU if NUM_GPU > 0 else 1
    steps = 1280000 // batch_size // NUM_DISTRIBUTE
    print(f"[INFO] Epochs:{epochs} Steps:{steps} Workers:{NUM_DISTRIBUTE} Batch size:{batch_size}")
    his = model.fit(
        x=ds_train,
        steps_per_epoch=steps,
        callbacks=[get_lr_func(epochs, lr_sched, initial_lr, final_lr), PrintAccOnEpochEnd()],
        # The following doesn't seem to help in terms of speed.
        # use_multiprocessing=True, workers=4,
        epochs=epochs,
        verbose=1)
    
    final_acc = 0. if len(his.history['acc']) < 1 else his.history['acc'][-1]
    print(f"Final acc:{final_acc}")
    nni.report_final_result(final_acc)
    # training finished
    # model.save('googlenet_bn-2gpu-model-final.h5')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dataset_dir', type=str,
                        default=config.DEFAULT_DATASET_DIR)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--epsilon', type=float, default=1e-1)
    parser.add_argument('--label_smoothing', action='store_false')
    parser.add_argument('--use_lookahead', action='store_true')
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--iter_size', type=int, default=-1)
    parser.add_argument('--lr_sched', type=str, default='linear',
                        choices=['linear', 'exp'])
    parser.add_argument('--initial_lr', type=float, default=-1.)
    parser.add_argument('--final_lr', type=float, default=-1.)
    parser.add_argument('--weight_decay', type=float, default=-1.)
    parser.add_argument('--epochs', type=int, default=1,
                        help='total number of epochs for training [1]')
    parser.add_argument('model', type=str,
                        help=SUPPORTED_MODELS)
    args = parser.parse_args()

    if args.use_lookahead and args.iter_size > 1:
        raise ValueError('cannot set both use_lookahead and iter_size')

    # os.makedirs(config.SAVE_DIR, exist_ok=True)
    # os.makedirs(config.LOG_DIR, exist_ok=True)
    config_keras_backend()
    # check if running hyperband
    epochs_to_run = params["NUM_EPOCH"] if 'TRIAL_BUDGET' not in params.keys() else params["TRIAL_BUDGET"]%4 # valid NUM_EPOCH:{1,2,3}
    train(args.model,
        0, #drop out rate
        'adam', #optimizer
        params["EPSILON"],
        args.label_smoothing, 
        args.use_lookahead,
        params["BATCH_SIZE"],
        args.iter_size,
        args.lr_sched, 
        params["INIT_LR"],
        params["FINAL_LR"],
        params["WEIGHT_DECAY"],
        epochs_to_run,
        args.dataset_dir)
    clear_keras_session()


def get_default_params():
    return {
        "EPSILON":0.3,
        "BATCH_SIZE":64,
        "INIT_LR":0.5,
        "FINAL_LR":1e-06,
        'WEIGHT_DECAY':7e-05,
        'NUM_EPOCH':1
    }

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    params = get_default_params()
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)
    main()
