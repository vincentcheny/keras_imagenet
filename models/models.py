"""models.py

Implemented models:
    1. MobileNetV2 ('mobilenet_v2')
    2. ResNet50 ('resnet50')
    3. GoogLeNetBN ('googlenet_bn')
    4. InceptionV2 ('inception_v2')
    5. EfficientNetB0_224x224 ('efficientnet_b0')
    6. EfficientNetB1_224x224 ('efficientnet_b1')
    7. EfficientNetB4_224x224 ('efficientnet_b4')
    8. OSNet ('osnet')
"""


import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras import backend as K

from config import config
from .googlenet import GoogLeNetBN
from .inception_v2 import InceptionV2, InceptionV2X
from .inception_mobilenet import InceptionMobileNet
from .efficientnet import EfficientNetB0_224x224
from .efficientnet import EfficientNetB1_224x224
from .efficientnet import EfficientNetB4_224x224
from .osnet import OSNet
from .adamw import AdamW
from .optimizer import convert_to_accum_optimizer
from .optimizer import convert_to_lookahead_optimizer

from tensorflow.keras.utils import multi_gpu_model


IN_SHAPE = (224, 224, 3)  # shape of input image tensor
# NUM_CLASSES = 300        # number of output classes (1000 for ImageNet) # 300 modification
# NUM_CLASSES = 1000
NUM_CLASSES = 500 #500 modification

def _set_l2(model, weight_decay):
    """Add L2 regularization into layers with weights

    Reference: https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.add_loss(
                tf.keras.regularizers.l2(weight_decay)(layer.depthwise_kernel))
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.add_loss(
                tf.keras.regularizers.l2(weight_decay)(layer.kernel))
        elif isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(
                tf.keras.regularizers.l2(weight_decay)(layer.kernel))
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
           if layer.gamma is not None:
               layer.add_loss(
                   tf.keras.regularizers.l2(weight_decay)(layer.gamma))


def get_batch_size(model_name, value):
    """get_batch_size

    These default batch_size values were chosen based on available
    GPU RAM (11GB) on GeForce GTX-2080Ti.
    """
    if value > 0:
        return value
    elif 'mobilenet_v2' in model_name:
        return 64
    elif 'resnet50' in model_name:
        return 16
    elif 'googlenet_bn' in model_name:
        return 64
    else:
        raise ValueError


def get_iter_size(model_name, value):
    """get_iter_size

    These default iter_size values were chosen to make 'effective'
    batch_size to be 256.
    """
    if value > 0:
        return value
    elif 'mobilenet_v2' in model_name:
        return 4
    elif 'resnet50' in model_name:
        return 16
    elif 'googlenet_bn' in model_name:
        return 4
    else:
        raise ValueError


def get_initial_lr(model_name, value):
    return value if value > 0. else 3e-4


def get_final_lr(model_name, value):
    return value if value > 0. else 3e-4


def get_lr_func(total_epochs, lr_sched='linear',
                initial_lr=1e-2, final_lr=1e-5, NUM_GPU=1):
    """Returns a learning decay function for training.

    2 types of lr_sched are supported: 'linear' or 'exp' (exponential).
    """
    def linear_decay(epoch):
        """Decay LR linearly for each epoch."""
        if total_epochs == 1:
            return initial_lr
        else:
            ratio = max((total_epochs - epoch - 1.) / (total_epochs - 1.), 0.)
            lr = final_lr + (initial_lr - final_lr) * ratio
            print('Epoch %d, lr = %f' % (epoch+1, lr))
            return lr

    def exp_decay(epoch):
        """Decay LR exponentially for each epoch."""
        if total_epochs == 1:
            return initial_lr
        else:
            lr_decay = (final_lr / initial_lr) ** (1. / (total_epochs - 1))
            lr = initial_lr * (lr_decay ** epoch)
            print('Epoch %d, lr = %f' % (epoch+1, lr))
            return lr

    if total_epochs < 1:
        raise ValueError('bad total_epochs (%d)' % total_epochs)
    if lr_sched == 'linear':
        return tf.keras.callbacks.LearningRateScheduler(linear_decay)
    elif lr_sched == 'exp':
        return tf.keras.callbacks.LearningRateScheduler(exp_decay)
    else:
        raise ValueError('bad lr_sched')


def get_customize_lr_callback(*args, **kwargs):
    class CustomizedLearningRateScheduler(Callback):
        def __init__(self, model, total_step, initial_lr, final_lr, update_interval=1000, decay_type="linear"):
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
                    lr = float(K.get_value(self.model.optimizer.learning_rate))
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
                K.set_value(self.model.optimizer.learning_rate, lr)

    return CustomizedLearningRateScheduler(*args, **kwargs)

def get_weight_decay(model_name, value):
    return value if value >= 0. else 1e-5


def get_optimizer(model_name, optim_name, initial_lr, epsilon=1e-2):
    """get_optimizer

    Note:
    1. Learning rate decay is implemented as a callback in model.fit(),
       so I do not specify 'decay' in the optimizers here.
    2. Refer to the following for information about 'epsilon' in Adam:
       https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/keras/optimizer_v2/adam.py#L93
    """
    if optim_name == 'sgd':
        return tf.keras.optimizers.SGD(lr=initial_lr, momentum=0.9)
    elif optim_name == 'adam':
        return tf.keras.optimizers.Adam(lr=initial_lr, epsilon=epsilon)
    elif optim_name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(lr=initial_lr, epsilon=epsilon,
                                           rho=0.9)
    else:
        # implementation of 'AdamW' is removed temporarily
        raise ValueError


def get_training_model(model_name, dropout_rate, optimizer, label_smoothing,
                       use_lookahead, iter_size, weight_decay,gpus):
    """Build the model to be trained."""
    if model_name.endswith('.h5'):
        # load a saved model
        # model = tf.keras.models.load_model(
        #     model_name,
        #     compile=False,
        #     custom_objects={'AdamW': AdamW})
        # model.load_weights(model_name)
        model = tf.keras.models.load_model(
             "./saves/keras_save",
             compile=False)
    else:
        # initialize the model from scratch
        model_class = {
            'mobilenet_v2': tf.keras.applications.mobilenet_v2.MobileNetV2,
            'resnet50': tf.keras.applications.resnet50.ResNet50,
            'googlenet_bn': GoogLeNetBN,
            'inception_v2': InceptionV2,
            'inception_v2x': InceptionV2X,
            'inception_mobilenet': InceptionMobileNet,
            'efficientnet_b0': EfficientNetB0_224x224,
            'efficientnet_b1': EfficientNetB1_224x224,
            'efficientnet_b4': EfficientNetB4_224x224,
            'osnet': OSNet,
        }[model_name]
        backbone = model_class(
            input_shape=IN_SHAPE, include_top=False, weights=None, classes=NUM_CLASSES)

        # Add a Dropout layer before the final Dense output
        x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
        if dropout_rate and dropout_rate > 0.0 and dropout_rate < 1.0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        bias_initializer = tf.constant_initializer(value=0.0)
        x = tf.keras.layers.Dense(
            NUM_CLASSES, activation='softmax', name='Logits',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)(x)
        model = tf.keras.models.Model(inputs=backbone.input, outputs=x)

    if weight_decay > 0.:
        _set_l2(model, weight_decay)

    if iter_size > 1:
        optimizer = convert_to_accum_optimizer(optimizer, iter_size)
    if use_lookahead:
        optimizer = convert_to_lookahead_optimizer(optimizer)

    # make sure all layers are set to be trainable
    for layer in model.layers:
        layer.trainable = True

    if tf.__version__ >= '1.14':
        smooth = 0.1 if label_smoothing else 0.
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=smooth)
    else:
        tf.logging.warning('"--label_smoothing" not working for '
                           'tensorflow-%s' % tf.__version__)
        loss = 'categorical_crossentropy'
    
    try:
        # model = multi_gpu_model(model, gpus=gpus, cpu_merge=False)
        # model.load_weights(model_name)
        print(f"Training using {gpus} GPUs...")
    except Exception as e:
        print(f"[Error] Catch an exception:\n{e}")
        print("Training using single GPU or CPU...")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', 'top_k_categorical_accuracy'])
    # print(model.summary())

    return model
