#!/usr/bin/env python
# coding: utf-8

# ## Random initialization and train all layers

# In[1]:


"""Script to finetune Unet using Tensorflow."""

# %%
import os
import re
import dill
import numpy as np
import tensorflow as tf
import time
import pickle
from unet import Unet
from datagenerator import *
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from generate_collages import *

# The random seed for np and tf are independent. In order to reproduce results, I need to set both seeds.
np.random.seed(123)
tf.random.set_seed(123)

"""
Configuration Part.

The model training parts follow examples: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb
and examples: https://www.tensorflow.org/tensorboard/get_started.

Also use the TensorBoard examples in the tensorflow page above.

"""

# Path to the textfiles for the trainings and validation set
# tr_file = '/path/to/train.txt'
# val_file = '/path/to/val.txt'
# fd_name = '5_texture_images_5c'
fd_name = '5_texture_images_5v2'
post_fix = '_1_2'
root_dir = '/projects/p30309/'
# root_dir = '~/scratch'
base_dir = os.path.join(os.path.expanduser(os.path.join(root_dir, 'Data/texture/Brodatz/')), fd_name)
tr_file = os.path.join(base_dir, 'train.txt')
val_file = os.path.join(base_dir, 'valid.txt')
plt_dir = os.path.join(os.path.expanduser(os.path.join(root_dir, '20200211_Unet_seg_res/figures/')), fd_name+post_fix)

if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)

# # Generate datasets
# train_size, valid_size, test_size = 20000, 2000, 5000
ls_fnames = list(os.listdir(base_dir)) # Must past in the list of filenames to keep the order and label of textures.
# gen_save_train_valid_test_dataset(base_dir, train_size, valid_size, test_size, ls_fnames=ls_fnames)


# Learning params
learning_rate = 0.0001
num_epochs = 1
batch_size = 32
weight_decay= 0.0005 # Caffe style regularization parameter
keep_prob = 0.5
num_classes = 5
segmentation_regions = num_classes

# Network params
dropout_rate = 0.5
# # train_layers = ['fc8', 'fc7', 'fc6'] # Only train last few layers.
train_layers = ['conv1_1', 'conv1_2',
                'conv2_1', 'conv2_2',
                'conv3_1', 'conv3_2',
                'conv4_1', 'conv4_2',
                'conv5_1', 'conv5_2',
                'conv6_1', 'conv6_2', 'conv6_3',
                'conv7_1', 'conv7_2', 'conv7_3',
                'conv8_1', 'conv8_2', 'conv8_3',
                'conv9_1', 'conv9_2', 'conv9_3',
                'conv10'] # Train all trainable layers.

# How often we want to write the tf.summary data to disk
display_step = 50

# Path for tf.summary.FileWriter and to store model checkpoints
log_dir = os.path.expanduser(os.path.join(root_dir, "logdir/Unet/"))
filewriter_path = os.path.join(log_dir, "finetune_unet/tensorboard")
checkpoint_path = os.path.join(log_dir, "finetune_unet/checkpoints")

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = UnetDataGenerator(tr_file,
                                mode='training',
                                batch_size=batch_size,
                                num_classes=num_classes)
    val_data = UnetDataGenerator(val_file,
                                mode='inference',
                                batch_size=batch_size,
                                num_classes=num_classes)

# Initialize model
# # Don't load the trainable layers
model = Unet(keep_prob, num_classes, [])
# # Still load the trainable layers, but the last layer.
# model = AlexNet(keep_prob, num_classes, ['fc8'], weights_path='../bvlc_alexnet.npy')

print(model.model.layers)

for var in model.model.variables:
    print(var.name, var.trainable)

# Load the pretrained weights into the model
# model.load_initial_weights()

# Note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    # Convert labels to int 32 for tf cross-entropy function.
    y = tf.cast(y, tf.int32)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=-1, output_type=tf.int32), 
                                  tf.cast(y_true, tf.int32))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Loss objects
train_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
test_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_regu = tf.keras.metrics.Mean('train_regularization', dtype=tf.float32)
train_obj_val = tf.keras.metrics.Mean('train_objective_value', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('valid_accuracy')

# List of trainable variables of the layers we want to train
var_list = []
for lname in train_layers:
    var_list.extend(model.model.get_layer(lname).trainable_variables)
for var in var_list:
    print("The variable name is {}.".format(var.name))

# Optimization process. 
def train_step(model, x, y, optimizer, var_list, weight_decay):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = model(x, is_training=True)
        # Compute loss.
        loss = train_loss_object(y, pred)
        # Be careful about the weight decay in caffe and L2 regularization.
        # https://bbabenko.github.io/weight-decay/
        # https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
        regu = weight_decay*tf.reduce_sum([tf.nn.l2_loss(var) for var in var_list if re.search(r'kernel', var.name)])
        obj_val = loss + regu

    # Compute gradients.
    gradients = g.gradient(obj_val, var_list)
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, var_list))

    # Log metrics
    train_loss(loss)
    train_regu(regu)
    train_obj_val(obj_val)
    train_accuracy(y, pred)

def valid_step(model, x, y):
    # Forward pass.
    pred = model(x)
    # Compute loss.
    loss = test_loss_object(y, pred)

    # Log metrics
    valid_loss(loss)
    valid_accuracy(y, pred)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
valid_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Create filer writer
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join(filewriter_path, current_time+'/train/')
valid_log_dir = os.path.join(filewriter_path, current_time+'/valid/')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)


# In[1]:

def plt_seg_res(image, lab, pred, fig, num_samples, i, rand_line_colors, num_classes, plt_dir, title_size=20):
    img_size = image.shape[0]
    acc = np.sum(lab==pred)/img_size**2
    uni_labs = np.unique(lab)

    ax = fig.add_subplot(num_samples, 3, (i-1)*3+1)
    image = image - np.min(image)
    image = image/np.max(image)*255
    ax.imshow(image.astype(np.uint8), cmap='gray')
    ax.set_title("Texture labels in this plot:\n{}".format(uni_labs), size=title_size)

    ax = fig.add_subplot(num_samples, 3, (i-1)*3+2)
    ax.imshow(lab, cmap='gray')
    ax.set_title("The true segmentation", size=title_size)

    ax = fig.add_subplot(num_samples, 3, (i-1)*3+3)
    ax.imshow(lab, cmap='gray')
    for j, l_idx in enumerate(list(range(num_classes))):
        coord_y, coord_x = np.where(pred==l_idx)
        ax.scatter(coord_x, coord_y, c=rand_line_colors[j%len(rand_line_colors)], marker='o', s=0.5, alpha=0.2)
    ax.set_title("The segementation with\nthe true: acc({:.4f})".format(acc), size=title_size)

# Train the model
start_time = time.time()

line_colors = ['blue', 'red', 'green', 'cyan', 'orange', 'magenta']

rand_line_colors = line_colors[:num_classes]
np.random.shuffle(rand_line_colors)

# Run training for the given number of steps.
print("{} Start training...".format(datetime.now()))
print("{} Open Tensorboard at --logdir {}".format(datetime.now(), os.path.join(filewriter_path, current_time)))
for epoch in range(num_epochs):

    print("{} Epoch number: {}".format(datetime.now(), epoch+1))

    # On cpu (crunch), each epoch takes about 640s. On gpu (colab), each epoch takes about 170s.
    ep_start_time = time.time()
    # Training
    for step, (batch_x, batch_y) in enumerate(tr_data.data, 1):
        # Run the optimization to update W and b values.
        train_step(model, batch_x, batch_y, optimizer, var_list, weight_decay)
        
        if step % display_step == 0:
            pred = model(batch_x)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accuracy(pred, batch_y)
            # print(pred[0].shape, batch_y[0].shape, batch_x[0].shape)
            # print(pred[0], batch_y[0], batch_x[0])
            fig = plt.figure(figsize=(15,6), facecolor='w')
            plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                        i=1, rand_line_colors=rand_line_colors, num_classes=num_classes, plt_dir=plt_dir)
            plt.savefig(os.path.join(plt_dir, 'train_{}.png'.format(step,)))
            plt.close()
            # np.savetxt(os.path.join(plt_dir, 'train_true_lab_{}.csv'.format(step,)), batch_y[0].numpy(), fmt='%d', delimiter=',')
            # np.savetxt(os.path.join(plt_dir, 'train_pred_lab_{}.csv'.format(step,)), pred[0].numpy().argmax(axis=-1), fmt='%d', delimiter=',')
            # # print("{} training step: %i, loss: %f, accuracy: %f" % (datetime.now(), step, loss, acc*100))
            print("{} training step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), step, loss, acc*100))

        if step >= train_batches_per_epoch:
            break
    
    # Log metrics for training
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('regularization', train_regu.result(), step=epoch)
        tf.summary.scalar('objective value', train_obj_val.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    # Validation
    for step, (batch_x, batch_y) in enumerate(val_data.data, 1):
        # Run validation.
        valid_step(model, batch_x, batch_y)
        
        if step % display_step == 0:
            pred = model(batch_x)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accuracy(pred, batch_y)
            # print(pred[0], batch_y[0], batch_x[0])
            fig = plt.figure(figsize=(15,6), facecolor='w')
            plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                        i=1, rand_line_colors=rand_line_colors, num_classes=num_classes, plt_dir=plt_dir)
            plt.savefig(os.path.join(plt_dir, 'valid_{}.png'.format(step,)))
            plt.close()
            np.savetxt(os.path.join(plt_dir, 'valid_true_lab_{}.csv'.format(step,)), batch_y[0].numpy(), fmt='%d', delimiter=',')
            np.savetxt(os.path.join(plt_dir, 'valid_pred_lab_{}.csv'.format(step,)), pred[0].numpy().argmax(axis=-1), fmt='%d', delimiter=',')
            # print("{} validating step: %i, loss: %f, accuracy: %f" % (datetime.now(), step, loss, acc*100))
            print("{} validating step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), step, loss, acc*100))

        if step >= valid_batches_per_epoch:
            break
    
    # Log metrics for validation
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', valid_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)

    # Reset metrics every epoch
    train_loss.reset_states()
    train_regu.reset_states()
    train_obj_val.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    
    print("The epoch %i takes %f s." % (epoch, time.time()-ep_start_time))

print("Total training for {} steps is: {}s.".format(num_epochs, time.time()-start_time))

# Save the model
# Using pickle would generate error: TypeError: can't pickle weakref objects
# pickle.dump(model, open(os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+current_time+'.h5'), 'wb'))
# model_name = 'rand_init_full_train_model_epoch_'+str(num_epochs)+current_time+'.h5'
model_name = '_'.join(['rand_init_full_train_model_epoch', fd_name, str(num_epochs), current_time])+'.h5'
dill.dump(model, open(os.path.join(checkpoint_path, model_name), 'wb'))
loaded_model = dill.load(open(os.path.join(checkpoint_path, model_name), 'rb'))
print(model_name)


# %% Validate on testing datasets
te_file = os.path.join(base_dir, 'test.txt')
te_data = UnetDataGenerator(te_file,
                            mode='inference',
                            batch_size=batch_size,
                            num_classes=num_classes)

for step, (batch_x, batch_y) in enumerate(te_data.data, 1):
    # Run validation.
    valid_step(loaded_model, batch_x, batch_y)
    
    if step % display_step == 0:
        pred = loaded_model(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("{} testing step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), step, loss, acc*100))

print("{} The testing loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), valid_loss.result(), valid_accuracy.result()))
valid_loss.reset_states()
valid_accuracy.reset_states()

# %%[markdown]
# Plot some validation segmentation results

all_textures = generate_texture(base_dir)

fig = plt.figure(figsize=(num_classes*5, 7), facecolor='w')

for i, texture in enumerate(all_textures):
    fig.add_subplot(1,num_classes, i+1)
    plt.imshow(texture.astype(np.uint8), cmap='gray')
    plt.title("Class label: {}.".format(i), size=20)
    plt.axis('off')
plt.savefig(os.path.join(plt_dir, 'all_textures_true_labs.png'))
plt.close()

# %%[markdown]
# Plot some validation segmentation results
num_samples = 10

fig = plt.figure(figsize=(15, 6*num_samples), facecolor='w')

line_colors = ['blue', 'red', 'green', 'cyan', 'orange', 'magenta']

rand_line_colors = line_colors[:num_classes]
np.random.shuffle(rand_line_colors)

for i, (image, lab) in enumerate(te_data.data.unbatch().take(num_samples),1):
    image, lab = image.numpy(), lab.numpy()

    pred = loaded_model(image).numpy()[0].argmax(axis=-1).astype(np.int32)

    plt_seg_res(image, lab, pred, fig, num_samples, i, rand_line_colors, num_classes, plt_dir)

plt.savefig(os.path.join(plt_dir, 'unet_seg_res_testing.png'))
plt.close()



