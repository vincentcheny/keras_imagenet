"""train.sh

This script is used to train the ImageNet models.
"""


import os
import time
import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras import backend as K

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
          weight_decay, steps, dataset_dir):
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
        weight_decay=weight_decay)
    
    class CustomizedLearningRateScheduler(Callback):
        def __init__(self, model, total_step, initial_lr, final_lr, update_interval=100, decay_type="linear"):
            self.model = model
            self.total_step = total_step
            self.initial_lr = initial_lr
            self.final_lr = final_lr
            self.update_interval = update_interval
            self.decay_type = decay_type

        def on_train_batch_end(self, batch, logs=None):
            if batch % self.update_interval == 0:
                if not hasattr(self.model.optimizer, 'lr'):
                    raise ValueError('Optimizer must have a "lr" attribute.')
                try:  # new API
                    lr = float(K.get_value(self.model.optimizer.lr))
                    if batch < 100:
                        lr = self.initial_lr
                    else:
                        if self.decay_type == "exp":
                            lr_decay = (self.final_lr / self.initial_lr) ** (1. / (self.total_step - 1))
                            lr = self.initial_lr * (lr_decay ** batch)
                        else:
                            ratio = max((self.total_step - batch - 1.) / (self.total_step - 1.), 0.)
                            lr = self.final_lr + (self.initial_lr - self.final_lr) * ratio
                        print(f'\n[UPDATE] Step {batch+1}, lr = {lr}')
                except TypeError:  # Support for old API for backward compatibility
                    lr = self.initial_lr
                    print(f"There is a TypeError: {TypeError}")
                K.set_value(self.model.optimizer.lr, lr)


    his = model.fit(
        x=ds_train,
        steps_per_epoch=steps,
        callbacks=[CustomizedLearningRateScheduler(model,steps,initial_lr,final_lr)],
        # The following doesn't seem to help in terms of speed.
        # use_multiprocessing=True, workers=4,
        epochs=1)
    final_acc = his.history['acc'][0]
    print(f"Final acc:{final_acc}")
    nni.report_final_result(final_acc)
    # training finished
    # model.save('{}/{}-model-final.h5'.format(config.SAVE_DIR, save_name))


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

    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    hardware_para = [
        params["inter_op_parallelism_threads"],
        params["intra_op_parallelism_threads"],
        params["max_folded_constant"],
        params["build_cost_model"],
        params["do_common_subexpression_elimination"],
        params["do_function_inlining"],
        params["global_jit_level"],
        params["infer_shapes"],
        params["place_pruned_graph"],
        params["enable_bfloat16_sendrecv"],
    ]
    config_keras_backend(hardware_para)
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
        params["NUM_STEPS"],
        args.dataset_dir)
    clear_keras_session()


def get_default_params():
    return {
        "EPSILON":0.5,
        "BATCH_SIZE":16,
        "INIT_LR":1,
        "FINAL_LR":5e-4,
        'WEIGHT_DECAY':2e-4,
        'NUM_STEPS':500,
		"inter_op_parallelism_threads":1,
        "intra_op_parallelism_threads":2,
        "max_folded_constant":6,
        "build_cost_model":4,
        "do_common_subexpression_elimination":1,
        "do_function_inlining":1,
        "global_jit_level":1,
        "infer_shapes":1,
        "place_pruned_graph":1,
        "enable_bfloat16_sendrecv":1
    }

if __name__ == '__main__':
    params = get_default_params()
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)
    main()
