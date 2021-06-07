"""This is a TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import re
import copy

# Customized Layer for local response normalization layer
class LRN(layers.Layer):
    """ Implementation of Local Response Normalization Layer.
       
        See Krizhevsky et al., ImageNet classification with deep convolutional neural networks (NIPS 2012).

        Reference: 
            Customized Layers: https://www.tensorflow.org/guide/keras/custom_layers_and_models
            How to inherit from Layer: https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/nn_ops.py
            Some nn. operations: /home/ghhgkz/.local/lib/python3.7/site-packages/tensorflow_core/python/ops/gen_nn_ops.py
    """
    def __init__(self, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None, plot_model=False, **kwargs):
        super(LRN, self).__init__(name=name, **kwargs)
        self.LRN_kwargs = {}
        self.LRN_kwargs['depth_radius']=depth_radius
        self.LRN_kwargs['bias']=bias
        self.LRN_kwargs['alpha']=alpha
        self.LRN_kwargs['beta']=beta
        
        # For plotting model graph
        if plot_model:
            inputs = tf.keras.Input(shape=(55,55,96))
            outputs = tf.nn.local_response_normalization(inputs, **self.LRN_kwargs)
            self.graph = tf.keras.Model(inputs=inputs, outputs=outputs)

    # def build(self, input_shape):
    #     self.depth_radius_v = tf.Variable(initial_value=self.depth_radius, trainable=False)
    #     self.bias_v=tf.Variable(initial_value=self.bias, trainable=False)
    #     self.alpha_v=tf.Variable(initial_value=self.alpha, trainable=False)
    #     self.beta_v=tf.Variable(initial_value=self.beta, trainable=False)

    def call(self, inputs):
        return tf.nn.local_response_normalization(inputs, **self.LRN_kwargs)
    
    def get_config(self):
        config = super(LRN, self).get_config()
        config.update(self.LRN_kwargs)
        return config


# Customized Layer for group convolutions
class GroupConv2D(layers.Layer):
    """ Implementation of group convolutions. 

        Reference:
            https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
            http://slazebni.cs.illinois.edu/spring17/lec12_vae.pdf
            https://blog.yani.io/filter-group-tutorial/
            https://towardsdatascience.com/grouped-convolutions-convolutions-in-parallel-3b8cc847e851

    """
    def __init__(self, 
                filters,
                kernel_size,
                groups=1,
                strides=(1, 1),
                padding='valid',
                data_format=None,
                dilation_rate=(1, 1),
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                name=None,
                plot_model=False,
                **kwargs):
        super(GroupConv2D, self).__init__(name=name, **kwargs)
        self.groups=groups
        self.Conv2D_kwargs = {}
        self.Conv2D_kwargs['filters']=filters
        self.Conv2D_kwargs['kernel_size']=kernel_size
        self.Conv2D_kwargs['strides']=strides
        self.Conv2D_kwargs['padding']=padding
        self.Conv2D_kwargs['data_format']=data_format
        self.Conv2D_kwargs['dilation_rate']=dilation_rate
        self.Conv2D_kwargs['activation']=activation
        self.Conv2D_kwargs['use_bias']=use_bias
        self.Conv2D_kwargs['kernel_initializer']=kernel_initializer
        self.Conv2D_kwargs['bias_initializer']=bias_initializer
        self.Conv2D_kwargs['kernel_regularizer']=kernel_regularizer
        self.Conv2D_kwargs['bias_regularizer']=bias_regularizer
        self.Conv2D_kwargs['activity_regularizer']=activity_regularizer
        self.Conv2D_kwargs['kernel_constraint']=kernel_constraint
        self.Conv2D_kwargs['bias_constraint']=bias_constraint

        # For plotting model graph
        if plot_model:
            inputs = tf.keras.Input(shape=(27,27,96))
            outputs = self._sequence(inputs)
            self.graph = tf.keras.Model(inputs=inputs, outputs=outputs)
                
    def _sequence(self, inputs):
        # In order to have trainable variables when we follow the tf.keras.layers api (without providing the input dimension),
        # we need to put all layers in the self.build(). The input dimension will be inferred in self.call().
        if self.groups == 1:
            return layers.Conv2D(**self.Conv2D_kwargs)(inputs)
        else:
            assert not self.Conv2D_kwargs['filters'] % self.groups
            group_filters = self.Conv2D_kwargs['filters'] // self.groups
            
            group_Conv2D_kwargs = copy.deepcopy(self.Conv2D_kwargs)
            group_Conv2D_kwargs['filters'] = group_filters

            output_groups = [layers.Conv2D(name='group{}'.format(i), **group_Conv2D_kwargs) for i in range(self.groups)]

            # In a grouped convolution layer, input and output channels are divided into `cardinality` groups,
            # and convolutions are separately performed within each group
            
            # Split input and weights and convolve them separately
            # The input has dimension as [batch, height, width, filter]
            input_groups = tf.split(value=inputs, num_or_size_splits=self.groups, axis=-1)
            output_groups = [ly(ip) for ly, ip in zip(output_groups, input_groups)]
            
            # Concat the convolved output together again
            return layers.concatenate(inputs=output_groups, axis=-1) 
        
    def build(self, input_shape):
        # In order to have trainable variables when we follow the tf.keras.layers api (without providing the input dimension),
        # we need to put all layers in the self.build(). The input dimension will be inferred in self.call().
        if self.groups == 1:
            self.output_layer = layers.Conv2D(**self.Conv2D_kwargs)
        else:
            assert not self.Conv2D_kwargs['filters'] % self.groups
            self.group_filters = self.Conv2D_kwargs['filters'] // self.groups
            
            group_Conv2D_kwargs = copy.deepcopy(self.Conv2D_kwargs)
            group_Conv2D_kwargs['filters'] = self.group_filters

            self.output_groups = [layers.Conv2D(name='group{}'.format(i), **group_Conv2D_kwargs) for i in range(self.groups)]

    def call(self, inputs):
        if self.groups==1:
            self.out = self.output_layer(inputs)
        else:
            # In a grouped convolution layer, input and output channels are divided into `cardinality` groups,
            # and convolutions are separately performed within each group
            
            # Split input and weights and convolve them separately
            # The input has dimension as [batch, height, width, filter]
            input_groups = tf.split(value=inputs, num_or_size_splits=self.groups, axis=-1)
            output_groups = [ly(ip) for ly, ip in zip(self.output_groups, input_groups)]
            
            # Concat the convolved output together again
            self.out = layers.concatenate(inputs=output_groups, axis=-1)
        
        return self.out

    def get_config(self):
        # This is for best practise and we can call the model as a Functional API.
        # https://www.tensorflow.org/guide/keras/custom_layers_and_models#you_can_optionally_enable_serialization_on_your_layers
        config = super(GroupConv2D, self).get_config()
        config.update(self.Conv2D_kwargs)
        return config


class AlexNetModel(Model):
    """
        Reference:
            Explanation on different paddings (VALID or SAME): https://stackoverflow.com/a/37675359/4307919
    """
    def __init__(self, keep_prob, num_classes, plot_model=False):
        super(AlexNetModel, self).__init__(name='AlexNetModel')

        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.DROP_PROB = 1.0-self.KEEP_PROB
        
        # Define layers
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        # Notice that the default dtype is tf.float32 (https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/engine/base_layer.py#L328)
        self.conv1 = layers.Conv2D(filters=96, kernel_size=11, strides=4, padding='VALID', 
                                   data_format='channels_last', activation=tf.nn.relu, use_bias=True, name='conv1')
        self.norm1 = LRN(depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75, name='norm1', plot_model=plot_model)
        self.pool1 = layers.MaxPool2D(pool_size=3, strides=2, padding='VALID', data_format='channels_last', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        self.conv2 = GroupConv2D(filters=256, kernel_size=5, groups=2, strides=1, padding='SAME', 
                                 data_format='channels_last', activation=tf.nn.relu, use_bias=True, name='conv2', plot_model=plot_model)
        self.norm2 = LRN(depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75, name='norm2')
        self.pool2 = layers.MaxPool2D(pool_size=3, strides=2, padding='VALID', data_format='channels_last', name='pool2')

        # 3nd Layer: Conv (w ReLu)
        self.conv3 = layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='SAME', 
                                   data_format='channels_last', activation=tf.nn.relu, use_bias=True, name='conv3')
        
        # 4nd Layer: Conv (w ReLu) splitted into 2 groups
        self.conv4 = GroupConv2D(filters=384, kernel_size=3, groups=2, strides=1, padding='SAME', 
                                 data_format='channels_last', activation=tf.nn.relu, use_bias=True, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into 2 groups
        self.conv5 = GroupConv2D(filters=256, kernel_size=3, groups=2, strides=1, padding='SAME', 
                                 data_format='channels_last', activation=tf.nn.relu, use_bias=True, name='conv5')
        self.pool5 = layers.MaxPool2D(pool_size=3, strides=2, padding='VALID', data_format='channels_last', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        self.flatten = layers.Flatten(data_format='channels_last', name='flatten')
        self.fc6 = layers.Dense(units=4096, activation=tf.nn.relu, use_bias=True, name='fc6')
        self.dropout6 = layers.Dropout(rate=self.DROP_PROB, name='dropout6')

        # 7th Layer: FC (w ReLu) -> Dropout
        self.fc7 = layers.Dense(units=4096, activation=tf.nn.relu, use_bias=True, name='fc7')
        self.dropout7 = layers.Dropout(rate=self.DROP_PROB, name='dropout7')

        # 8th Layer: FC and return unscaled activations (w/o ReLu)
        self.fc8 = layers.Dense(units=self.NUM_CLASSES, activation=None, use_bias=True, name='fc8')

        # Functional API for plotting network
        if plot_model:
            inputs = tf.keras.Input(shape=(227, 227, 3))
            outputs = self._sequence(inputs)
            self.graph = tf.keras.Model(inputs=inputs, outputs=outputs)

    def _sequence(self, inputs, is_training=False):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        
        x = self.conv4(x)
        
        x = self.conv5(x)
        x = self.pool5(x)
        
        x = self.flatten(x)
        x = self.fc6(x)
        x = self.dropout6(x, training=is_training)
        
        x = self.fc7(x)
        x = self.dropout7(x, training=is_training)

        return self.fc8(x)

    def call(self, input_tensor, is_training=False):
        # The reason that the input shape is different from the original AlexNet paper is because that
        # the pre-trained model is from bvlc(http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/).
        inputs = tf.reshape(input_tensor, [-1, 227, 227, 3])
        outputs = self._sequence(inputs, is_training=is_training)

        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            outputs = tf.nn.softmax(outputs)
        return outputs

    def get_config(self):
        config = super(AlexNetModel, self).get_config()
        config.update({'keep_prob': self.KEEP_PROB, 'num_classes': self.NUM_CLASSES})
        return config
    

class AlexNet(Model):
    """Implementation of the AlexNet."""
    # Set layers
    def __init__(self, keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT', plot_model=False):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        super(AlexNet, self).__init__()
        # Parse input arguments into class variables
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Construct and build the AlexNet model
        self.model = AlexNetModel(self.KEEP_PROB, self.NUM_CLASSES, plot_model=plot_model)
        self.model(np.zeros((1, 227, 227, 3), dtype=np.float32))

        if weights_path is not None and len(weights_path)>0:
            self.load_initial_weights()        

    # Set forward pass
    @tf.function
    def call(self, input_tensor, is_training=False):
        return self.model(input_tensor, is_training)

    def get_config(self):
        config = super(AlexNet, self).get_config()
        config.update({'keep_prob': self.KEEP_PROB, 'num_classes': self.NUM_CLASSES})
        return config
        
    def find_tf_var(self, op_layer, ls_pats):
        for var in op_layer.variables:
            print(var.name)
            match_flags = np.array([re.search(pat, var.name) is not None for pat in ls_pats])
            if np.all(match_flags):
                return var

    def check_assign(self, var, dat):
        # This utility can help skip layers with inconsistent shapes.
        if var.shape == dat.shape:
            var.assign(dat)
        else:
            print("The layer {}({}) does not have the same shape as the pre-trained weights({}).".format(var.name, var.shape, dat.shape))

    def load_layer_weights(self, op_layer, ls_data):
        for data in ls_data:
            # GroupConv2D Layer
            if hasattr(op_layer, 'groups') and op_layer.groups>1:
                data_groups = np.split(ary=data, indices_or_sections=op_layer.groups, axis=-1) # The axis has to be -1, otherwise for bias it would report error.
                for i, dat in enumerate(data_groups):
                    # Biases
                    if len(dat.shape) == 1:
                        var = self.find_tf_var(op_layer, ['group{}'.format(i), 'bias'])
                        self.check_assign(var, dat)
                    # Weights
                    else:
                        var = self.find_tf_var(op_layer, ['group{}'.format(i), 'kernel'])
                        print(dat.shape)
                        self.check_assign(var, dat)
                        
            # Normal Conv2D Layer
            else:
                # Biases
                if len(data.shape) == 1:
                    var = self.find_tf_var(op_layer, ['bias'])
                    self.check_assign(var, data)
                # Weights
                else:
                    var = self.find_tf_var(op_layer, ['kernel'])
                    self.check_assign(var, data)

    def load_initial_weights(self):
        """ Load weights from file into network.

            As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
            come as a dict of lists (e.g. weights['conv1'] is a list) and not as
            dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
            'biases') we need a special load function.
        """
        # Load the weights into memory
        # Add `allow_pickle=True`. OW, it raises error.
        weights_dict = np.load(self.WEIGHTS_PATH, allow_pickle=True, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            print("The name of variable is {}.".format(op_name))

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:
                op_layer = self.model.get_layer(name=op_name)
                if hasattr(op_layer, 'groups'):
                    print(op_layer.groups)
                self.load_layer_weights(op_layer, weights_dict[op_name])

