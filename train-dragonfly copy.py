"""train.sh

This script is used to train the ImageNet models.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable redundant message

import numpy as np
from utils.utils import fix_randomness
fix_randomness()

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.python.client import device_lib
NUM_GPU = 2 # len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
os.environ["CUDA_VISIBLE_DEVICES"]=str(list(range(NUM_GPU)))[1:-1].replace(" ", "") # 3 => "0,1,2"

import time
import argparse

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

from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
from dragonfly.utils.option_handler import get_option_specs, load_options
# import warnings
# warnings.simplefilter("ignore", category=DeprecationWarning)
import config as config1
config1.num_trial = 0

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


def train(model_name, 
          dropout_rate, 
          optim_name, 
          epsilon,
          label_smoothing, 
          use_lookahead, 
          batch_size, 
          iter_size,
          lr_sched, 
          initial_lr, 
          final_lr,
          weight_decay, 
          epochs, 
          dataset_dir,
          cross_device_ops,
          num_packs,
          tf_gpu_thread_mode):
    start = time.time()
    """Prepare data and train the model."""
    if tf_gpu_thread_mode in ["global", "gpu_private", "gpu_shared"]:
        os.environ['TF_GPU_THREAD_MODE'] = tf_gpu_thread_mode
    batch_size   = get_batch_size(model_name, batch_size)
    iter_size    = get_iter_size(model_name, iter_size)
    initial_lr   = get_initial_lr(model_name, initial_lr)
    final_lr     = get_final_lr(model_name, final_lr)
    optimizer    = get_optimizer(model_name, optim_name, initial_lr, epsilon)
    weight_decay = get_weight_decay(model_name, weight_decay)
    # get training and validation data
    ds_train = get_dataset(dataset_dir, 'train', batch_size) # 300 modification
    ds_valid = get_dataset(dataset_dir, 'validation', batch_size) # 300 modification
    # ds_train = get_dataset("/lustre/project/EricLo/cx/imagenet/imagenet_1000classes_train/", 'train', batch_size) # 1000 modification
    # ds_valid = get_dataset("/lustre/project/EricLo/cx/imagenet/imagenet_1000classes_val/", 'validation', batch_size) # 1000 modification
    if cross_device_ops == "HierarchicalCopyAllReduce":
        mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=num_packs))
    elif cross_device_ops == "NcclAllReduce":
        mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce(num_packs=num_packs))
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = get_training_model(
            model_name=model_name,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            label_smoothing=label_smoothing,
            use_lookahead=use_lookahead,
            iter_size=iter_size,
            weight_decay=weight_decay,
            gpus=NUM_GPU)

    class PrintAcc(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch{epoch+1} {logs}")

            
    
    NUM_DISTRIBUTE = NUM_GPU if NUM_GPU > 0 else 1
    # train_steps = int(1281167 / batch_size) # 1000 classes
    # val_steps = int(50000 / batch_size) # 1000 classes
    # train_steps = int(383690 / batch_size) # 300 modification
    # val_steps = int(15000 / batch_size) # 300 modification
    train_steps = int(642289 / batch_size) # 500 modification
    val_steps = int(25000 / batch_size) # 500 modification
    print(f"[INFO] Total Epochs:{epochs} Train Steps:{train_steps} Validate Steps: {val_steps} Workers:{NUM_DISTRIBUTE} Batch size:{batch_size}")
    his = model.fit(
        x=ds_train,
        steps_per_epoch=train_steps,
        validation_data=ds_valid,
        validation_steps=val_steps,
        callbacks=[get_lr_func(epochs, lr_sched, initial_lr, final_lr, NUM_GPU)],
        # The following doesn't seem to help in terms of speed.
        # use_multiprocessing=True, workers=4,
        epochs=epochs,
        verbose=2)

    end = time.time()
    fit_time = (end - start) / 3600.0
    acc = 0. if len(his.history['val_top_k_categorical_accuracy']) < 1 else his.history['val_top_k_categorical_accuracy'][-1]
    print(f"[TRIAL END] time: {fit_time} {his.history}")
    return acc, fit_time

    # training finished
    # model.save('{}/{}-model-final.h5'.format(config.SAVE_DIR, save_name))


def runtime_eval(x):
    print(f"Trial config:{x}")
    config_keras_backend(x[6:15])

    global final_acc
    config1.num_trial += 1
    print(config1.num_trial, final_acc)
    
    if config1.num_trial == 0:
        acc = 0.4
        final_acc = acc
        return -float(0.2)
    elif config1.num_trial == 1:
        acc = 0.5
        final_acc = acc
        return -float(0.6)
    acc, fit_time = train(model_name, 
                            0, #drop out rate
                            x[15], #optimizer
                            x[0], #epsilon
                            label_smoothing, 
                            use_lookahead,
                            x[1], #batch size
                            iter_size,
                            lr_sched, 
                            x[2], #init LR
                            x[3], #final LR
                            x[4], #weight decay
                            x[5], #num_epoch
                            datadir,
                            x[16],
                            x[17],
                            x[18])
    print(f"NUM_TRIAL in runtime2: {config1.num_trial}, {acc} {fit_time}")
    clear_keras_session()
    # global final_acc
    final_acc = acc
    return -float(fit_time)

def acc_eval(x):
    print(f"NUM_TRIAL in eval: {config1.num_trial}")
    if config1.num_trial == 0:
        config1.num_trial += 1
        print(f"NUM_TRIAL in eval2: {config1.num_trial}")
        return 7.745438137319353
    elif config1.num_trial == 1:
        config1.num_trial += 1
        print(f"NUM_TRIAL in eval2: {config1.num_trial}")
        return 11.477242516875267      
    config1.num_trial += 1
    global final_acc
    return float(final_acc)


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

model_name = args.model
label_smoothing = args.label_smoothing
use_lookahead = args.use_lookahead
iter_size = args.iter_size
lr_sched = args.lr_sched
datadir = args.dataset_dir



if args.use_lookahead and args.iter_size > 1:
    raise ValueError('cannot set both use_lookahead and iter_size')

# os.makedirs(config.SAVE_DIR, exist_ok=True)
# os.makedirs(config.LOG_DIR, exist_ok=True)



#dragonfly part
final_acc = 0.0

#model para
epsilon_list = [0.1,0.3,0.5,0.7,1.0]
batch_list = [8,16,32,48,64]
batch_list = [i*NUM_GPU for i in batch_list]
init_LR_list = [1,5e-1,3e-1,1e-1,7e-2,5e-2,3e-2,1e-2]
final_LR_list = [5e-4,1e-4,5e-5,1e-5,5e-6,1e-6]
weight_decay_list = [2e-3,7e-4,2e-4,7e-5,2e-5]
epoch_list = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
optimizer_list = ['sgd', 'adam', 'rmsprop']


#hardware para
inter_list = [2,3,4]
intra_list = [2,4,6]
infer_shapes_list = [True,False]
place_pruned_graph_list = [True,False]
enable_bfloat16_sendrecv_list = [True,False]
do_common_subexpression_elimination_list = [True,False]
max_folded_constant_list = [2,4,6,8,10]
do_function_inlining_list = [True,False]
global_jit_level_list = [0,1,2]

cross_device_ops_list = ["HierarchicalCopyAllReduce", "NcclAllReduce"]
num_packs_list = [0,1,2,3,4,5]
tf_gpu_thread_mode_list = ["gpu_private", "global", "gpu_shared"]


domain_vars = [{'type': 'discrete_numeric', 'items': epsilon_list},
                {'type': 'discrete_numeric', 'items': batch_list},
                {'type': 'discrete_numeric', 'items': init_LR_list},
                {'type': 'discrete_numeric', 'items': final_LR_list},
                {'type': 'discrete_numeric', 'items': weight_decay_list},
                {'type': 'discrete_numeric', 'items': epoch_list},
                {'type': 'discrete_numeric', 'items': inter_list},
                {'type': 'discrete_numeric', 'items': intra_list},
                {'type': 'discrete', 'items': infer_shapes_list},
                {'type': 'discrete', 'items': place_pruned_graph_list},
                {'type': 'discrete', 'items': enable_bfloat16_sendrecv_list},
                {'type': 'discrete', 'items': do_common_subexpression_elimination_list},
                {'type': 'discrete_numeric', 'items': max_folded_constant_list},
                {'type': 'discrete', 'items': do_function_inlining_list},
                {'type': 'discrete_numeric', 'items': global_jit_level_list},
                {'type': 'discrete', 'items': optimizer_list},
                {'type': 'discrete', 'items': cross_device_ops_list},
                {'type': 'discrete_numeric', 'items': num_packs_list},
                {'type': 'discrete', 'items': tf_gpu_thread_mode_list}
                ]

dragonfly_args = [ 
  get_option_specs('report_results_every', False, 2, 'Path to the json or pb config file. '),
  get_option_specs('init_capital', False, None, 'Path to the json or pb config file. '),
  get_option_specs('init_capital_frac', False, 0.05, 'Path to the json or pb config file. '),
  get_option_specs('num_init_evals', False, 2, 'Path to the json or pb config file. ')]

options = load_options(dragonfly_args)
config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 60 * 60 * 90
moo_objectives = [runtime_eval, acc_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='realtime',config=config,options=options)
f = open("./googlenet_bn-dragonfly-90h-2gpu-output.log","w+")
print(pareto_opt_pts,file=f)
print("\n",file=f)
print(pareto_opt_vals,file=f)
print("\n",file=f)
print(history,file=f)