"""This is a TensorFlow implementation of Unet.

The model comes from https://github.com/zhixuhao/unet

Some useful implementations that are worth considering in the future:

https://github.com/qubvel/segmentation_models: 
This package and another package for different models implement many deep learning models
using TensorFlow 1.

https://github.com/FelixGruen/tensorflow-u-net:
Implementation of Unet using TensorFlow 1.

https://github.com/FelixGruen/tensorflow-u-net:
Another implementation of Unet using TensorFlow 1.

https://github.com/atch841/one-shot-texture-segmentation:
A TensorFlow implementation of segmentation model, but not Unet.

https://www.tensorflow.org/tutorials/images/segmentation:
TensorFlow also has an example of Unet.
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import re
import copy
import dill
from constants import *

class UnetModel(Model):
    def __init__(self, keep_prob, num_classes, plot_model=False):
        super(UnetModel, self).__init__(name='UnetModel')

        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.DROP_PROB = 1.0-self.KEEP_PROB

        # Encoder
        # 256
        self.conv1_1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv1_1')
        self.conv1_2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv1_2')
        self.pool1 = layers.MaxPool2D(pool_size=2, padding='VALID', data_format='channels_last', name='pool1')
        
        # 128
        self.conv2_1 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv2_1')
        self.conv2_2 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv2_2')
        self.pool2 = layers.MaxPool2D(pool_size=2, padding='VALID', data_format='channels_last', name='pool2')
        
        # 64
        self.conv3_1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv3_1')
        self.conv3_2 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv3_2')
        self.pool3 = layers.MaxPool2D(pool_size=2, padding='VALID', data_format='channels_last', name='pool3')
        
        # 32
        self.conv4_1 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv4_1')
        self.conv4_2 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv4_2')
        self.drop4 = layers.Dropout(rate=self.DROP_PROB, name='drop4')
        self.pool4 = layers.MaxPool2D(pool_size=2, padding='VALID', data_format='channels_last', name='pool4')

        # Bottle neck
        # 16
        self.conv5_1 = layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv5_1')
        self.conv5_2 = layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv5_2')
        self.drop5 = layers.Dropout(rate=self.DROP_PROB, name='drop5')

        # Decoder
        self.up6 = layers.UpSampling2D(size=2, data_format='channels_last', interpolation='nearest', name='up6')
        # 32
        self.conv6_1 = layers.Conv2D(filters=512, kernel_size=2, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv6_1')
        self.merge6 = layers.Concatenate(axis=-1, name='merge6')
        self.conv6_2 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv6_2') 
        self.conv6_3 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv6_3')
        
        self.up7 = layers.UpSampling2D(size=2, data_format='channels_last', interpolation='nearest', name='up7')
        # 64
        self.conv7_1 = layers.Conv2D(filters=256, kernel_size=2, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv7_1')
        self.merge7 = layers.Concatenate(axis=-1, name='merge7')
        self.conv7_2 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv7_2') 
        self.conv7_3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv7_3')

        self.up8 = layers.UpSampling2D(size=2, data_format='channels_last', interpolation='nearest', name='up8')
        # 128
        self.conv8_1 = layers.Conv2D(filters=128, kernel_size=2, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv8_1')
        self.merge8 = layers.Concatenate(axis=-1, name='merge8')
        self.conv8_2 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv8_2') 
        self.conv8_3 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv8_3')

        self.up9 = layers.UpSampling2D(size=2, data_format='channels_last', interpolation='nearest', name='up9')
        # 256
        self.conv9_1 = layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv9_1')
        self.merge9 = layers.Concatenate(axis=-1, name='merge9')
        self.conv9_2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv9_2') 
        self.conv9_3 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME', 
                                     data_format='channels_last', activation=tf.nn.relu, 
                                     use_bias=True, kernel_initializer='he_normal', name='conv9_3')
        
        # Final output layer
        # 256
        self.conv10 = layers.Conv2D(filters=self.NUM_CLASSES, kernel_size=1, strides=1, padding='SAME', 
                                    data_format='channels_last', activation=None, 
                                    use_bias=True, kernel_initializer='he_normal', name='conv10')
        
        # Functional API for plotting network
        if plot_model:
            inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
            outputs = self._sequece(inputs)
            self.graph = tf.keras.Model(inputs=inputs, outputs=outputs)

    def _sequece(self, inputs, training=False):
        # Encoder
        conv1_1 = self.conv1_1(inputs)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)
        
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)
        
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)
        
        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        drop4 = self.drop4(conv4_2, training=training)
        pool4 = self.pool4(drop4)
        
        # Bottleneck
        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        drop5 = self.drop5(conv5_2, training=training)
        
        # Decoder        
        up6 = self.up6(drop5)
        conv6_1 = self.conv6_1(up6)
        merge6 = self.merge6([drop4, conv6_1])
        conv6_2 = self.conv6_2(merge6)
        conv6_3 = self.conv6_3(conv6_2)

        up7 = self.up7(conv6_3)
        conv7_1 = self.conv7_1(up7)
        merge7 = self.merge7([conv3_2, conv7_1])
        conv7_2 = self.conv7_2(merge7)
        conv7_3 = self.conv7_3(conv7_2)

        up8 = self.up8(conv7_3)
        conv8_1 = self.conv8_1(up8)
        merge8 = self.merge8([conv2_2, conv8_1])
        conv8_2 = self.conv8_2(merge8)
        conv8_3 = self.conv8_3(conv8_2)

        up9 = self.up9(conv8_3)
        conv9_1 = self.conv9_1(up9)
        merge9 = self.merge9([conv1_2, conv9_1])
        conv9_2 = self.conv9_2(merge9)
        conv9_3 = self.conv9_3(conv9_2)

        # Final layer
        conv10 = self.conv10(conv9_3)

        return conv10

    def call(self, input_tensor, training=False):
        inputs = tf.reshape(input_tensor, [-1,256,256,3])
        outputs = self._sequece(inputs, training=training)

        if not training:
            outputs = tf.nn.softmax(outputs, axis=-1)
        return outputs

    def get_config(self):
        config = super(UnetModel, self).get_config()
        config.update({'keep_prob': self.KEEP_PROB, 'num_classes': self.NUM_CLASSES})
        return config


class Unet(Model):
    """Implementation of a Unet."""
    def __init__(self, keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT', rnd_seed=123, plot_model=False):
        """Create the graph of the Unet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        super(Unet, self).__init__()
        # Parse input arguments into class variables
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.NUM_CLASSES = num_classes
        self.RND_SEED = rnd_seed

        # if weights_path == 'DEFAULT':
        #     self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        # else:
        #     self.WEIGHTS_PATH = weights_path

        # Construct and build the Unet model
        self.model = UnetModel(self.KEEP_PROB, self.NUM_CLASSES, plot_model=plot_model)
        self.model(np.zeros((1, 256, 256, 3), dtype=np.float32))        

    # Set forward pass
    # Examples of using @tf.function: https://towardsdatascience.com/tensorflow-2-0-tf-function-and-autograph-af2b974cf4f7
    @tf.function
    def call(self, input_tensor, training=False):
        return self.model(input_tensor, training)

    def get_config(self):
        config = super(Unet, self).get_config()
        config.update({'keep_prob': self.KEEP_PROB, 
                       'num_classes': self.NUM_CLASSES,
                       'rnd_seed': self.RND_SEED})
        return config

    def load_layer_weights_expand_last_layer(self, old_model):
        # old_model = dill.load(open(old_model_path, 'rb')).model
        for old_ly, ly in zip(old_model.layers, self.model.layers):
            for old_var, var in zip(old_ly.trainable_variables, ly.trainable_variables):
                if old_var.shape == var.shape:
                    var.assign(old_var.numpy())
                else:
                    new_wei_shape = tuple([dim if old_dim==dim else 1 for old_dim, dim in zip(old_var.shape, var.shape)])
                    if re.search(r'kernel', var.name):
                        # Use the same initialization method
                        init = tf.keras.initializers.GlorotUniform(self.RND_SEED)
                        var.assign(np.concatenate((old_var.numpy(), init(new_wei_shape).numpy()), axis=-1))
                        # Initialize as zero
                        # var.assign(np.concatenate((old_var.numpy(), np.zeros(new_wei_shape)), axis=-1))
                        # Initialize as minimum
                        # var.assign(np.concatenate((old_var.numpy(), np.min(old_var.numpy(), axis=-1, keepdims=True)), axis=-1))
                    else:
                        var.assign(np.concatenate((old_var.numpy(), np.zeros(new_wei_shape)), axis=-1))
                        # # Initialize as minimum
                        # var.assign(np.concatenate((old_var.numpy(), np.min(old_var.numpy(), axis=-1, keepdims=True)), axis=-1))




