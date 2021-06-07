# %% [markdown]
# Understand how to change variables using tensorflow 2 functions.
import tensorflow as tf
import numpy as np
import re


# %%
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1), name='conv2a')
        self.bn2a = tf.keras.layers.BatchNormalization(name='bn2a')

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', name='conv2b')
        self.bn2b = tf.keras.layers.BatchNormalization(name='bn2b')

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1), name='conv2c')
        self.bn2c = tf.keras.layers.BatchNormalization(name='bn2c')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])

# %% [markdown]
# Before given any data to the model
block.layers

# %% [markdown]
## Before passing in any data, the variable is empty
block.variables

# %% [markdown]
## We can get layers by name
conv2a = block.get_layer(name='conv2a')
print(conv2a, conv2a.name, conv2a.trainable, conv2a.trainable_weights, conv2a.variables, conv2a.get_weights())

# %% [markdown]
# We build the model
block.build((None, 2, 3, 3))

# %% [markdown]
block.layers

# %% [markdown]
## The variables show up and intialized
block.variables

# %% [markdown]
## Get the layer and print variables
conv2a = block.get_layer(name='conv2a')
### conv2a is tf.keras.layers.Layer
print(type(conv2a), conv2a)
print(conv2a.name)
print(conv2a.trainable)
### conv2a.trainable_weights is a list of tf.Variable
print(type(conv2a.trainable_weights), type(conv2a.trainable_weights[0]), conv2a.trainable_weights)
### conv2a.variables is a list of tf.Variable
print(type(conv2a.variables), type(conv2a.variables[0]), conv2a.variables)
### conv2a.get_weights() is as list of np.ndarray
print(type(conv2a.get_weights()), type(conv2a.get_weights()[0]), conv2a.get_weights())


# %% [markdown]
## Show shape of variables
print(conv2a.variables[0].shape, conv2a.variables[1].shape)

# %% [markdown]
## Assign value to a layer
k_v, b_v = np.ones(conv2a.variables[0].shape)*2, np.ones(conv2a.variables[1].shape)*2
print("The old variable values of conv2d are: {}.".format(conv2a.variables))
conv2a.variables[0].assign(k_v)
conv2a.variables[1].assign(b_v)
print("The new variable values of conv2a are: {}.".format(conv2a.variables))

# %% [markdown]
## Get the layer and print variables for 'conv2c'
conv2c = block.get_layer(name='conv2c')
### conv2c is tf.keras.layers.Layer
print(type(conv2c), conv2c)
print(conv2c.name)
print(conv2c.trainable)
### conv2c.trainable_weights is a list of tf.Variable
print(type(conv2c.trainable_weights), type(conv2c.trainable_weights[0]), conv2c.trainable_weights)
### conv2c.variables is a list of tf.Variable
print(type(conv2c.variables), type(conv2c.variables[0]), conv2c.variables)
### conv2c.get_weights() is as list of np.ndarray
print(type(conv2c.get_weights()), type(conv2c.get_weights()[0]), conv2c.get_weights())


# %% [markdown]
## Show shape of variables
print(conv2c.variables[0].shape, conv2c.variables[1].shape)


# %% [markdown]
## Assign value to a layer
k_v, b_v = np.ones(conv2c.variables[0].shape)*2, np.ones(conv2c.variables[1].shape)*2
print("The old variable values of conv2d are: {}.".format(conv2c.variables))
conv2c.variables[0].assign(k_v)
conv2c.variables[1].assign(b_v)
print("The new variable values of conv2c are: {}.".format(conv2c.variables))

# %% [markdown]
# Finally print the variables after assignment
for var in block.variables:
    print(var.name, var.trainable, var)


# %% [markdown]
for var in block.variables:
    if var.trainable:
        print(var.name, var.trainable, var)

# %% [markdown]
# Explore pre-trained AlexNet weights
weights_dict = np.load('../bvlc_alexnet.npy', allow_pickle=True, encoding='bytes').item()

# %%
for op_name in weights_dict:
    print(op_name)
    for data in weights_dict[op_name]:
        print(data.shape)

# %%
block.variables

# %%
conv2a.trainable_variables

# %%
import tensorflow as tf
# tf.enable_eager_execution()

features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor 
labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor 
dataset = tf.data.Dataset.from_tensor_slices((features, labels)) 
# Both the features and the labels tensors can be converted 
# to a tf.data.Dataset object separately and combined after. 
features_dataset = tf.data.Dataset.from_tensor_slices(features) 
labels_dataset = tf.data.Dataset.from_tensor_slices(labels) 
dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

# for element in dataset.as_numpy_iterator(): 
#     print(element)

for ele in dataset: 
    print(ele[0].numpy(), ele[1].numpy().decode("utf-8"))

def my_func(feat, lab):
    print(tf.executing_eagerly())
    return tf.reduce_sum(feat), tf.cast(lab, dtype=tf.int32)

dataset = dataset.map(my_func)

for element in dataset.as_numpy_iterator(): 
    print(element)

# # A batched feature and label set can be converted to a tf.data.Dataset 
# # in similar fashion. 
# batched_features = tf.constant([[[1, 3], [2, 3]], 
#                                 [[2, 1], [1, 2]], 
#                                 [[3, 3], [3, 2]]], shape=(3, 2, 2)) 
# batched_labels = tf.constant([['A', 'A'], 
#                               ['B', 'B'], 
#                               ['A', 'B']], shape=(3, 2, 1)) 
# dataset = tf.data.Dataset.from_tensor_slices((batched_features, batched_labels)) 
# for element in dataset.as_numpy_iterator(): 
#     print(element) 


