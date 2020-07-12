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
    print(np.random.rand())
    np.random.seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)


def config_keras_backend(params=None):
    print(f"params:{params}")
    """Config tensorflow backend to use less GPU memory."""
    if not params or len(params) < 10:
        print("\nUse default hardware config.")
        config = tf.ConfigProto()
    else:
        # pass
        config = tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=int(params[0]),
            intra_op_parallelism_threads=int(params[1]))
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)


def clear_keras_session():
    """Clear keras session.

    This is for avoiding the problem of: 'Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object ...'
    """
    tf.keras.backend.clear_session()
