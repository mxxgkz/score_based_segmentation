# %%
import sys
import dill
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# %%
sys.path.append('Segmentation/Unet')
print(sys.path)
from unet import *

# %%
existed_model_path = 'logdir/Unet/finetune_unet/checkpoints/rand_init_full_train_model_epoch_120200202-114209.h5'
model = dill.load(open(existed_model_path,'rb'))

# # %%
# for ly in model.model.layers:
#     for var in ly.trainable_variables:
#         print(var.name, var.shape)
# 
# # %%
# conv10_k, conv10_b = model.model.get_layer('conv10').trainable_variables
# 
# # %%
# conv10_k.assign(np.ones(conv10_k.shape))
# print(conv10_k)
# 
# # %%
# model.model.get_layer('conv10').trainable_variables
# 
# # %%
# conv10_k.numpy()
# 
# # %%
# conv10_k.name, conv10_b.name
# 
# # %%
# init_exp = tf.keras.initializers.GlorotUniform()
# 
# # %%
# init_exp((10,1)).numpy()
# 
# # %%
# arr1 = np.array([[1,2],[3,4]])
# arr2 = np.array([[5],[6]])
# print(arr1, arr2)
# 
# # %%
# np.concatenate((arr1, arr2), axis=-1)
# 
# # %%
# tuple([1,2,3])

# %%
# Plot diagram of Unet:
temp_model = Unet(keep_prob=0.5, num_classes=6, skip_layer=[], plot_model=True)
# tf.keras.utils.plot_model(temp_model.model.graph, 'unet_plot.png', show_shapes=True, expand_nested=True)

# %%
temp_model.model.graph.summary()

fig = plt.figure(figsize=(15,12), facecolor='w')



# %%
print("The weights before loading: \n{}.".format(model.model.get_layer('conv10').trainable_variables))

# Load the pretrained weights into the model
temp_model.load_layer_weights_expand_last_layer(existed_model_path)

print("The weights after loading: \n{}.".format(temp_model.model.get_layer('conv10').trainable_variables))
