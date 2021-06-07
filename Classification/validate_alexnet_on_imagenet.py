#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this notebook we will test the implementation of the AlexNet class provided in the `alexnet.py` file. This is part of [this](https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html) blog article on how to finetune AlexNet with TensorFlow 1.0.
# 
# To run this notebook you have to download the `bvlc_alexnet.npy` file from [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/), which stores the pretrained weigts of AlexNet.
# 
# The idea to validate the implementation is to create an AlexNet graph with the provided script and load all pretrained weights into the variables (so no finetuneing!), to see if everything is wired up correctly.


# %% [markdown]
## Import libraries.

#some basic imports and setups
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# The random seed for np and tf are independent. In order to reproduce results, I need to set both seeds.
np.random.seed(123)
tf.random.set_seed(123)

#mean of imagenet dataset in BGR (notice it is BGR, not RGB)
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = os.getcwd()
image_dir = os.path.join(current_dir, 'images')

get_ipython().magic(u'matplotlib inline')


# %% [markdown]
## Plot testing images.

#get list of all images
img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg') or f.endswith('.png')]
img_true_labs = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.jpeg') or f.endswith('.png')]

#load all images
imgs = []
for f in img_files:
    imgs.append(cv2.imread(f))
    
#plot images
fig = plt.figure(figsize=(20,12))
for i, img in enumerate(imgs):
    fig.add_subplot(2,len(imgs)//2,i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')


# First we will create placeholder for the dropout rate and the inputs and create an AlexNet object. Then we will link the activations from the last layer to the variable `score` and define an op to calculate the softmax values.

# %% [markdown]
## Construct AlexNet.

from alexnet import AlexNet
from caffe_classes import class_names

keep_prob = 0.5
num_classes = 1000
skip_layer = []

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = AlexNet(keep_prob, num_classes, skip_layer, weights_path='../bvlc_alexnet.npy')

# Now we will start a TensorFlow session and load pretrained weights into the layer weights. Then we will loop over all images and calculate the class probability for each image and plot the image again, together with the predicted class and the corresponding class probability.


# %% [markdown]
## Don't load the pre-trained model. The model is randomly initialized.
## The predictions are completely off.

# Top k values
top_k = 5

# Create figure handle
fig = plt.figure(figsize=(20,12))

# Loop over all imageis
for i, image in enumerate(imgs):
    
    # Convert image to float32 and resize to (227x227)
    img = cv2.resize(image.astype(np.float32), (227,227))
    
    # Subtract the ImageNet mean
    img -= imagenet_mean
    
    # Reshape as needed to feed into model
    img = img.reshape((1,227,227,3))
    
    # Run the session and calculate the class probability
    probs = model(img).numpy().reshape((-1,))
    
    # Get the class name of the class with the highest probability
    class_name = class_names[np.argmax(probs)]

    # Top k probabilities and class names
    top_k_prob_idx = probs.argsort()[-top_k:][::-1]
    top_k_probs = [probs[j] for j in top_k_prob_idx]
    top_k_class_names = [class_names[j] for j in top_k_prob_idx]
    
    # Plot image with class name and prob in the title
    fig.add_subplot(2,len(imgs)//2,i+1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('True Class: {}\n'.format(img_true_labs[i]) + 
              '\n'.join(["Class: " + cn.split(',')[0] + ", probability: %.4f" %pr for cn, pr in zip(top_k_class_names, top_k_probs)]))
    plt.axis('off')


# %% [markdown]
## Load pre-trained AlexNet.
## The predictions are very close.

# Load the pretrained weights into the model
model.load_initial_weights()

# Create figure handle
fig = plt.figure(figsize=(20,12))

# Loop over all imageis
for i, image in enumerate(imgs):
    
    # Convert image to float32 and resize to (227x227)
    img = cv2.resize(image.astype(np.float32), (227,227))
    
    # Subtract the ImageNet mean
    img -= imagenet_mean
    
    # Reshape as needed to feed into model
    img = img.reshape((1,227,227,3))
    
    # Run the session and calculate the class probability
    probs = model(img).numpy().reshape((-1,))
    
    # Get the class name of the class with the highest probability
    class_name = class_names[np.argmax(probs)]

    # Top k probabilities and class names
    top_k_prob_idx = probs.argsort()[-top_k:][::-1]
    top_k_probs = [probs[j] for j in top_k_prob_idx]
    top_k_class_names = [class_names[j] for j in top_k_prob_idx]
    
    # Plot image with class name and prob in the title
    fig.add_subplot(2,len(imgs)//2,i+1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('True Class: {}\n'.format(img_true_labs[i]) + 
              '\n'.join(["Class: " + cn.split(',')[0] + ", probability: %.4f" %pr for cn, pr in zip(top_k_class_names, top_k_probs)]))
    plt.axis('off')


# %% [markdown]
## Notice that cv2 would read data in BGR order instead of RGB. If we mistakenly past in RGB to the AlexNet
## trained by Caffe, which uses BGR order, the inference results can be off quite a bit.
## The credibility of predictions decreases.

# Create figure handle
fig = plt.figure(figsize=(20,12))

# Loop over all imageis
for i, image in enumerate(imgs):
    
    # Convert image to float32 and resize to (227x227)
    img = cv2.resize(image.astype(np.float32), (227,227))
    
    # Subtract the ImageNet mean
    img -= imagenet_mean
    
    # Reshape as needed to feed into model
    img = img.reshape((1,227,227,3))
    
    # Run the session and calculate the class probability
    # probs = model(img).numpy().reshape((-1,))
    probs = model(img[...,::-1]).numpy().reshape((-1,))
    
    # Get the class name of the class with the highest probability
    class_name = class_names[np.argmax(probs)]

    # Top k probabilities and class names
    top_k_prob_idx = probs.argsort()[-top_k:][::-1]
    top_k_probs = [probs[j] for j in top_k_prob_idx]
    top_k_class_names = [class_names[j] for j in top_k_prob_idx]
    
    # Plot image with class name and prob in the title
    fig.add_subplot(2,len(imgs)//2,i+1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(image)
    plt.title('True Class: {}\n'.format(img_true_labs[i]) + 
              '\n'.join(["Class: " + cn.split(',')[0] + ", probability: %.4f" %pr for cn, pr in zip(top_k_class_names, top_k_probs)]))
    plt.axis('off')


# %%
for var in model.model.variables:
    if var.trainable:
        print(var.name)

# %%
for ly in model.model.layers:
    print(ly.name, len(ly.variables), hasattr(ly, 'groups'))
    if len(ly.variables):
        for var in ly.variables:
            print(var.name, var.shape)

# %%
