from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import re
import copy

class DenseNet(Model):
    def __init__(self, keep_prob, num_classes, num_nodes, plot_model=False):
        super(DenseNet, self).__init__(name='DenseNet')

        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.DROP_PROB = 1.0-self.KEEP_PROB
        self.NUM_NODES = num_nodes
        
        # Define layers
        # 1th Layer: Flatten -> FC (w ReLu) -> Dropout
        self.flatten = layers.Flatten(data_format='channels_last', name='flatten')
        self.fc1 = layers.Dense(units=self.NUM_NODES, activation=tf.nn.relu, use_bias=True, name='fc1')
        self.dropout1 = layers.Dropout(rate=self.DROP_PROB, name='dropout1')

        # 2th Layer: FC (w ReLu) -> Dropout
        self.fc2 = layers.Dense(units=self.NUM_NODES, activation=tf.nn.relu, use_bias=True, name='fc2')
        self.dropout2 = layers.Dropout(rate=self.DROP_PROB, name='dropout2')

        # 3th Layer: FC and return unscaled activations (w/o ReLu)
        self.fc3 = layers.Dense(units=self.NUM_CLASSES, activation=None, use_bias=True, name='fc3')

        # Functional API for plotting network
        if plot_model:
            inputs = tf.keras.Input(shape=(227,227,3))
            outputs = self._sequence(inputs)
            self.graph = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Initialize the weights of all layers by passing a dummy data row
        self._sequence(np.zeros((1, 227, 227, 3), dtype=np.float32))

    def _sequence(self, inputs, is_training=False):
        x = self.flatten(inputs)
        x = self.fc1(x)
        x = self.dropout1(x, training=is_training)
        x = self.fc2(x)
        x = self.dropout2(x, training=is_training)
        return self.fc3(x)

    def call(self, input_tensor, is_training=False):
        
        inputs = tf.reshape(input_tensor, [-1, 227, 227, 3])
        outputs = self._sequence(inputs, is_training=is_training)

        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            outputs = tf.nn.softmax(outputs)
        return outputs

    def get_config(self):
        config = super(AlexNetModel, self).get_config()
        config.update({'keep_prob': self.KEEP_PROB, 'num_classes': self.NUM_CLASSES, 'num_nodes': self.NUM_NODES})
        return config
