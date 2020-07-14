"""train.sh

This script is used to train the ImageNet models.
"""
import numpy as np
from utils.utils import fix_randomness
fix_randomness()
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras import backend as K

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
          total_img, 
          dataset_dir):
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

    start = time.time()
    steps = total_img // batch_size
    his = model.fit(
        x=ds_train,
        steps_per_epoch=steps,
        callbacks=[get_customize_lr_callback(model,steps,initial_lr,final_lr)],
        # The following doesn't seem to help in terms of speed.
        # use_multiprocessing=True, workers=4,
        epochs=1,
        verbose=2)
    end = time.time()
    spent = (end - start) / 3600.0
    print(spent)
    print(his.history['acc'][0])
    return his.history['acc'][0],spent


    # training finished
    # model.save('{}/{}-model-final.h5'.format(config.SAVE_DIR, save_name))


def runtime_eval(x):
    print(x)
    config_keras_backend(x[6:])
    # config_keras_backend()
    acc,spent_time = train(model_name, 
                            0, #drop out rate
                            'adam', #optimizer
                            x[0], #epsilon
                            label_smoothing, 
                            use_lookahead,
                            x[1], #batch size
                            iter_size,
                            lr_sched, 
                            x[2], #init LR
                            x[3], #final LR
                            x[4], #weight decay
                            x[5], #total_img
                            datadir)
    clear_keras_session()
    global final_acc
    final_acc = acc
    return -float(spent_time)

def acc_eval(x):
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
init_LR_list = [1,5e-1,3e-1,1e-1,7e-2,5e-2,3e-2,1e-2]
final_LR_list = [5e-4,1e-4,5e-5,1e-5,5e-6,1e-6]
weight_decay_list = [2e-3,7e-4,2e-4,7e-5,2e-5]
total_img_list = [563200,640000,768000]



#hardware para
inter_list = [1,2,3,4]
intra_list = [2,4,6,8,10,12]
infer_shapes_list = [True,False]
place_pruned_graph_list = [True,False]
enable_bfloat16_sendrecv_list = [True,False]
do_common_subexpression_elimination_list = [True,False]
max_folded_constant_list = [2,4,6,8,10]
do_function_inlining_list = [True,False]
global_jit_level_list = [0,1,2]


domain_vars = [{'type': 'discrete_numeric', 'items': epsilon_list},
                {'type': 'discrete_numeric', 'items': batch_list},
                {'type': 'discrete_numeric', 'items': init_LR_list},
                {'type': 'discrete_numeric', 'items': final_LR_list},
                {'type': 'discrete_numeric', 'items': weight_decay_list},
                {'type': 'discrete_numeric', 'items': total_img_list},
                {'type': 'discrete_numeric', 'items': inter_list},
                {'type': 'discrete_numeric', 'items': intra_list},
                {'type': 'discrete', 'items': infer_shapes_list},
                {'type': 'discrete', 'items': place_pruned_graph_list},
                {'type': 'discrete', 'items': enable_bfloat16_sendrecv_list},
                {'type': 'discrete', 'items': do_common_subexpression_elimination_list},
                {'type': 'discrete_numeric', 'items': max_folded_constant_list},
                {'type': 'discrete', 'items': do_function_inlining_list},
                {'type': 'discrete_numeric', 'items': global_jit_level_list}
                ]

dragonfly_args = [ 
  get_option_specs('report_results_every', False, 2, 'Path to the json or pb config file. '),
  get_option_specs('init_capital', False, None, 'Path to the json or pb config file. '),
  get_option_specs('init_capital_frac', False, 0.05, 'Path to the json or pb config file. '),
  get_option_specs('num_init_evals', False, 2, 'Path to the json or pb config file. ')]

options = load_options(dragonfly_args)
config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 60 * 60 * 10
moo_objectives = [runtime_eval, acc_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='realtime',config=config,options=options)
f = open("./googlenet_bn-1gpu-dragonfly-10h-output.log","w+")
print(pareto_opt_pts,file=f)
print("\n",file=f)
print(pareto_opt_vals,file=f)
print("\n",file=f)
print(history,file=f)







