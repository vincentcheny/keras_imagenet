"""utils.py

Commond utils functions for TensorFlow & Keras.
"""

import os
import random
import numpy as np
import tensorflow as tf

def fix_randomness():
    seed_value=0
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)


def config_keras_backend(params=None):
    print(f"\nHardware params:{params}")
    """Config tensorflow backend to use less GPU memory."""
    if not params or len(params) < 9:
        print("\nUse default hardware config.")
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=int(params[0]),
            intra_op_parallelism_threads=int(params[1]),
            graph_options=tf.compat.v1.GraphOptions(
                infer_shapes=params[2],
                place_pruned_graph=params[3],
                enable_bfloat16_sendrecv=params[4],
                optimizer_options=tf.compat.v1.OptimizerOptions(
                    do_common_subexpression_elimination=params[5],
                    max_folded_constant_in_bytes=params[6],
                    do_function_inlining=params[7],
                    global_jit_level=params[8])))
        print("\nUse customized hardware config.")
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.keras.backend.set_session(session)


def clear_keras_session():
    """Clear keras session.

    This is for avoiding the problem of: 'Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object ...'
    """
    tf.keras.backend.clear_session()
