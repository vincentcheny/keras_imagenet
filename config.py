"""config.py
"""

import os
import types


config = types.SimpleNamespace()

# Subdirectory name for saving trained weights and models
config.SAVE_DIR = 'saves'

# Subdirectory name for saving TensorBoard log files
config.LOG_DIR = 'logs'


# Default path to the ImageNet TFRecords dataset files
# config.DEFAULT_DATASET_DIR = os.path.join(
#     os.environ['HOME'], 'data/ILSVRC2012/tfrecords')
config.DEFAULT_DATASET_DIR = '/lustre/project/EricLo/cx/imagenet/tf_records'

# Number of parallel works for generating training/validation data
config.NUM_DATA_WORKERS = 8

# Do image data augmentation or not
config.DATA_AUGMENTATION = False
