#!/usr/bin/env python
# coding: utf-8

# ## Random initialization and train all layers

# In[1]:


"""Script to finetune Unet using Tensorflow."""

# %%
from constants import *
import os
import re
import dill
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(N_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(N_THREADS)
import argparse
import time
import sys
import pickle
import copy
# import ansiwrap
from datetime import datetime
# In order to disable interactive backend when using matplotlib
# https://stackoverflow.com/questions/19518352/tkinter-tclerror-couldnt-connect-to-display-localhost18-0
# https://stackoverflow.com/questions/49284893/matplotlib-while-debugging-in-pycharm-how-to-turn-off-interactive-mode
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg') # Needed for running on quest
import matplotlib._color_data as mcd # For color palette
# import cv2
from seg_utils.generate_collages import *
from seg_utils.datagenerator import *
from seg_utils.utils import *
from Unet.unet import *
from DeepLab3.deeplab3 import *
from DeepLab3.deeplabv3plus_mobilenetv2_layers import *
from DeepLab3.deeplabv3plus_xception_layers import *


# To run on K80 GPU, with 12G memory, the float needs tf.float16 and batch size needs to be 4.

def main(_):
    # The random seed for np and tf are independent. In order to reproduce results, I need to set both seeds.
    np.random.seed(RND_SEED)
    tf.random.set_seed(RND_SEED)
    print(os.environ)
    tf.keras.backend.set_floatx(DTYPE_FLOAT_STR)
    print("The number of threads of independent and single operations are : {}, {}.".format(
        tf.config.threading.get_inter_op_parallelism_threads(),
        tf.config.threading.get_intra_op_parallelism_threads()))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("The default float type is : {}.".format(tf.keras.backend.floatx()))

    tf.autograph.set_verbosity(0)
    entire_start_time = time.time()

    """
    Configuration Part.

    The model training parts follow examples: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb
    and examples: https://www.tensorflow.org/tensorboard/get_started.

    Also use the TensorBoard examples in the tensorflow page above.

    TensorBoard profiling and showing graph: 
        https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
        https://www.tensorflow.org/tensorboard/graphs
        http://deeplearnphysics.org/Blog/2018-09-25-Profiling-Tensorflow.html
    """

    root_dir = ROOT_DIR # FLAGS.root_dir
    # plt_fd_name = '5_texture_images_5v2_5m'
    plt_fd_name = 'cla_incr_images'
    postfix = FLAGS.postfix
    plt_dir = os.path.join(os.path.expanduser(os.path.join(root_dir, '20200211_Unet_seg_res/figures/')), plt_fd_name+postfix)

    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)

    base_dir = os.path.expanduser(os.path.join(root_dir, 'Data/texture/Brodatz/{}/'.format(plt_fd_name)))

    db_fd_name = FLAGS.db_fd_name
    db_base_dir = os.path.join(base_dir, db_fd_name)
    ls_fnames = []
    for fn in list(os.listdir(db_base_dir)):
        if fn.endswith(FLAGS.img_ext):
            ls_fnames.append(fn)
    ls_fnames.sort()
    print(ls_fnames)
    new_fd_name = FLAGS.new_fd_name
    new_base_dir = os.path.join(base_dir, new_fd_name)
    ls_ext_fnames = []
    for fn in list(os.listdir(new_base_dir)):
        if fn.endswith(FLAGS.img_ext):
            ls_ext_fnames.append(fn)
    ls_ext_fnames.sort()
    print(ls_ext_fnames)
    if FLAGS.num_cla_incr>0:
        ls_ext_fnames = ls_ext_fnames[:FLAGS.num_cla_incr]
    print(ls_ext_fnames)

    # new_texture_tmpl = 'Nat-5m_*.pgm'

    # Learning params
    learning_rate = 0.0001 # Batch size 32
    # learning_rate = 0.000001 # Batch size 4
    num_epochs = FLAGS.num_epochs
    fine_tuning_num_epochs = FLAGS.fine_tuning_num_epochs
    batch_size = BATCH_SIZE
    weight_decay= 0.0005 # Caffe style regularization parameter
    num_classes = len(ls_fnames)
    # input_shapes = (256, 256, 3)
    input_shapes = (IMG_SIZE, IMG_SIZE, 3)
    new_size = IMG_SIZE

    # if FLAGS.model_name=='Unet':
    #     keep_prob = 0.5 # For Unet
    #     dropout_rate = 0.5 # For Unet
    #     train_layers = ['conv1_1', 'conv1_2',
    #                     'conv2_1', 'conv2_2',
    #                     'conv3_1', 'conv3_2',
    #                     'conv4_1', 'conv4_2',
    #                     'conv5_1', 'conv5_2',
    #                     'conv6_1', 'conv6_2', 'conv6_3',
    #                     'conv7_1', 'conv7_2', 'conv7_3',
    #                     'conv8_1', 'conv8_2', 'conv8_3',
    #                     'conv9_1', 'conv9_2', 'conv9_3',
    #                     'conv10'] # Train all trainable layers.
    #     last_layer_name = 'conv10'
    # elif FLAGS.model_name=='DeepLab3Plus':
    #     keep_prob = 0.9 # For DeepLab3Plus
    #     dropout_rate = 0.1 # For DeepLab3Plus
    #     train_layers = ['entry_conv', 'entry_bn_1', 'entry_conv_same', 'entry_bn_2', 
    #                     'entry_xception_1', 'entry_xception_2', 'entry_xception_3', 
    #                     'middle_xception_1', 'middle_xception_2', 'middle_xception_3', 
    #                     'middle_xception_4', 'middle_xception_5', 'middle_xception_6', 
    #                     'middle_xception_7', 'middle_xception_8', 'middle_xception_9', 
    #                     'middle_xception_10', 'middle_xception_11', 'middle_xception_12', 
    #                     'middle_xception_13', 'middle_xception_14', 'middle_xception_15', 
    #                     'middle_xception_16', 'exit_xception_1', 'exit_xception_2', 
    #                     'bran4_conv', 'bran4_bn', 'aspp0_conv', 'aspp0_bn', 'aspp1', 
    #                     'aspp2', 'aspp3', 'proj_conv', 'proj_bn', 'feature_proj_conv', 
    #                     'feature_proj_bn', 'deco_conv_0', 'deco_conv_1', 'last_conv'] # Train all trainable layers.
    #     last_layer_name = 'last_conv'
    # elif FLAGS.model_name=='DeepLab3PlusOrig':
    #     keep_prob = 0.9 # For DeepLab3Plus
    #     dropout_rate = 0.1 # For DeepLab3Plus
    #     train_layers = deeplabv3plus_xception_layers # Train all trainable layers.
    #     last_layer_name = 'custom_logits_semantic'
    #     # pretrained_wei_path = os.path.join(FLAGS.root_dir, 'Segmentation/DeepLab3/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')
    #     pretrained_wei_path = os.path.join(ROOT_DIR, 'Segmentation/DeepLab3/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')
    #     kwargs = {'keep_prob': keep_prob,
    #               'num_classes': num_classes,
    #               'output_stride': 16,
    #               'input_shapes': input_shapes,
    #               'weights_name': FLAGS.weights_name,
    #               'weights_path': pretrained_wei_path}
    # elif FLAGS.model_name in {'DeepLab3PlusMultiBackboneOrig_mobilenetv2', 'DeepLab3PlusMultiBackboneOrig_xception', 'DeepLab3PlusMultiBackboneModified_mobilenetv2', 'DeepLab3PlusMultiBackboneModified_xception'}:
    #     keep_prob = 0.9
    #     dropout_rate = 1-keep_prob
    #     train_layers = deeplabv3plus_mobilenetv2_layers if FLAGS.model_name.endswith('mobilenetv2') else deeplabv3plus_xception_layers
    #     last_layer_name = 'custom_logits_semantic'
    #     backbone = FLAGS.backbone
    #     alpha = 1.
    #     output_stride = 8 if FLAGS.model_name.endswith('mobilenetv2') else 16 # Does not useful for mobilenetv2 but useful for xception, because 'mobilenetv2' automatically use output_stride=8
    #     input_shapes = (IMG_SIZE,IMG_SIZE,3)
    #     weights_name = FLAGS.weights_name
    #     pretrained_wei_path = ''
    #     kwargs = {'keep_prob': keep_prob,
    #               'num_classes': num_classes,
    #               'backbone': backbone,
    #               'alpha': alpha,
    #               'output_stride': output_stride,
    #               'input_shapes': input_shapes,
    #               'weights_name': weights_name,
    #               'weights_path': pretrained_wei_path}

    (keep_prob, dropout_rate, train_layers, last_layer_name, 
     pretrained_wei_path, kwargs, backbone, alpha, output_stride,
     weights_name) = config_deeplab(num_classes, input_shapes, FLAGS)

    
    # # train_layers = ['fc8', 'fc7', 'fc6'] # Only train last few layers.

    # How often we want to write the tf.summary data to disk
    # display_step = FLAGS.display_step

    # Path for tf.summary.FileWriter and to store model checkpoints
    # log_dir = os.path.expanduser(os.path.join(root_dir, "logdir/Unet/"))
    log_dir = os.path.expanduser(os.path.join(root_dir, "logdir/{}/".format(FLAGS.model_name)))
    filewriter_path = os.path.join(log_dir, "finetune_{}/tensorboard".format(FLAGS.model_name))
    checkpoint_path = os.path.join(log_dir, "finetune_{}/checkpoints".format(FLAGS.model_name))

    # For plotting different colors in segementation
    # line_colors = ['blue', 'red', 'green', 'cyan', 'orange', 'magenta']
    # https://matplotlib.org/tutorials/colors/colors.html
    # line_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    # Use xkcd named colors with color names and rgb values. See https://matplotlib.org/tutorials/colors/colors.html
    num_colors = len(mcd.XKCD_COLORS)
    xkcd_color_names = list(mcd.XKCD_COLORS)
    line_colors = [xkcd_color_names[ci] for ci in np.random.choice(range(num_colors), size=len(ls_fnames+ls_ext_fnames)+1, replace=False)]
    print("The line_colors are: {}.".format(line_colors))
    
    rand_line_colors = [co for co in line_colors]
    np.random.shuffle(rand_line_colors)
    print("The rand_line_colors are: {}.".format(rand_line_colors))

    # Loss objects
    train_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    valid_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # train_loss_object = SparseSoftDiceLoss(from_logits=True, dtype=DTYPE_FLOAT)
    # test_loss_object = SparseSoftDiceLoss(from_logits=False, dtype=DTYPE_FLOAT)

    # Stochastic gradient descent optimizer.
    optimizer = tf.optimizers.Adam(learning_rate)

    # Define our metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=DTYPE_FLOAT)
    train_regu = tf.keras.metrics.Mean('train_regularization', dtype=DTYPE_FLOAT)
    train_obj_val = tf.keras.metrics.Mean('train_objective_value', dtype=DTYPE_FLOAT)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy', dtype=DTYPE_FLOAT)
    valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=DTYPE_FLOAT)
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('valid_accuracy', dtype=DTYPE_FLOAT)

    # # Optimization process.
    # # @tf.function # ValueError: tf.function-decorated function tried to create variables on non-first call.
    # def train_step(model, x, y, optimizer, var_list, weight_decay, train_loss, train_regu, train_obj_val, train_accuracy, grad_masks=None):
    #     # Wrap computation inside a GradientTape for automatic differentiation.
    #     with tf.GradientTape() as g:
    #         # Forward pass.
    #         pred = model(x, training=True)
    #         # Compute loss.
    #         loss = tf.cast(train_loss_object(y, pred), dtype=DTYPE_FLOAT) # The built-in function doesn't provide the dtype parameter
    #         # Be careful about the weight decay in caffe and L2 regularization.
    #         # https://bbabenko.github.io/weight-decay/
    #         # https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
    #         regu = weight_decay*tf.reduce_sum([tf.nn.l2_loss(var) for var in var_list if re.search(r'kernel', var.name)])
    #         # print("The dtype for loss and regu are {}, {}.".format(loss.dtype, regu.dtype))
    #         obj_val = loss + regu

    #     # Compute gradients.
    #     gradients = g.gradient(obj_val, var_list)
    #     if grad_masks is not None:
    #         # If mask is not None, element-wise mask out those weights that we don't want to update.
    #         for idx, mask in enumerate(grad_masks):
    #             gradients[idx] = gradients[idx]*mask
        
    #     # Update W and b following gradients.
    #     optimizer.apply_gradients(zip(gradients, var_list))

    #     # Log metrics
    #     train_loss(loss)
    #     train_regu(regu)
    #     train_obj_val(obj_val)
    #     train_accuracy(y, pred)

    # # @tf.function
    # def valid_step(model, x, y, valid_loss, valid_accuracy):
    #     # Forward pass.
    #     pred = model(x)
    #     # Compute loss.
    #     # Convert to tensor is necessary
    #     # https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
    #     # loss = test_loss_object(tf.convert_to_tensor(y), 
    #     #                         tf.convert_to_tensor(pred))
    #     loss = tf.cast(test_loss_object(y, pred), dtype=DTYPE_FLOAT)

    #     # Log metrics
    #     valid_loss(loss)
    #     valid_accuracy(y, pred)

    # def plt_seg_res(image, lab, lab_bd, pred, fig, num_samples, row_idx, rand_line_colors, num_classes, plt_dir, title_size=20):
    #     img_size = image.shape[0]
    #     # print(lab.dtype, pred.dtype)
    #     corr = lab==pred
    #     acc = np.sum(corr)/img_size**2
    #     # print(np.unique(lab).dtype, np.unique(pred).dtype)
    #     uni_labs = np.unique(lab).astype(np.int32)
    #     uni_preds = np.unique(pred).astype(np.int32)
    #     # print(np.concatenate((uni_labs, uni_preds), axis=-1).dtype)
    #     uni_all_labs = np.unique(np.concatenate((uni_labs, uni_preds), axis=-1)).astype(np.int32) # Returns the sorted unique elements of an array
    #     cla_tp = [] # True positive rate for each class
    #     cla_dice = [] # 
    #     cla_pred_all_num = []
    #     for ll in uni_all_labs:
    #         if ll in uni_labs:
    #             cla_all = lab==ll
    #             cla_tp.append(np.sum(corr*cla_all)/
    #                         np.sum(cla_all))
    #         else:
    #             cla_tp.append(-1)
    #         ll_lab, ll_pred = lab==ll, pred==ll
    #         cla_dice.append(dice(ll_lab, ll_pred))
    #         cla_pred_all_num.append(np.sum(ll_pred))

    #     def color_seg(ax, num_classes, img_lab, rand_line_colors):
    #         for l_idx in range(num_classes+1): # Notice that we have boundary color
    #             coord_y, coord_x = np.where(img_lab==l_idx)
    #             # Accepted color format from matplotlib: https://matplotlib.org/api/colors_api.html
    #             if l_idx == num_classes:
    #                 ax.scatter(coord_x, coord_y, c=rand_line_colors[l_idx], marker='o', s=0.5, alpha=0.8)
    #             else:
    #                 ax.scatter(coord_x, coord_y, c=rand_line_colors[l_idx], marker='o', s=0.5, alpha=0.2)

    #     def gen_str_color(txts, labs_idx, rand_line_colors):
    #         labs_str = '[_{}_]'.format('_,_'.join([str(ele) for ele in txts]))
    #         labs_color = [rand_line_colors[ll] for ll in labs_idx]
    #         labs_color = ['black' if ii%2==0 else labs_color[ii//2] for ii in range(2*len(labs_idx)+1)]
    #         return labs_str, labs_color

    #     # Original image
    #     ax = fig.add_subplot(num_samples, 3, (row_idx-1)*3+1)
    #     image = image - np.min(image)
    #     image = image/np.max(image)*255
    #     ax.imshow(image.astype(np.uint8), cmap='gray')

    #     uni_labs_str, uni_labs_str_color = gen_str_color(uni_labs, uni_labs, rand_line_colors)
    #     uni_preds_str, uni_preds_str_color = gen_str_color(uni_preds, uni_preds, rand_line_colors)
    #     cla_pred_all_num_str, cla_pred_all_num_str_color = gen_str_color(cla_pred_all_num, uni_all_labs, rand_line_colors)

    #     ax_texts = "Texture labs in this plot:_\n_{}_\n_Predicted labels:_\n_{}_\n_Number of Predicted:_\n_{}".format(
    #         uni_labs_str, uni_preds_str, cla_pred_all_num_str).split('_')
    #     ax_texts_color = ['black']*2+uni_labs_str_color+['black']*3+uni_preds_str_color+['black']*3+cla_pred_all_num_str_color
    #     if len(ax_texts)!=len(ax_texts_color):
    #         raise ValueError("The length of ax_texts({}) and ax_texts_str({}) should be the same.".format(len(ax_texts), len(ax_texts_str)))
    #     rainbow_text(fig, ax, -0.05, -0.15, ax_texts, ax_texts_color, sep='', size=title_size)
    #     # ax.set_title(ansiwrap.fill(, width=FIG_TITLE_WID), size=title_size)
        
    #     # True segmentation        
    #     ax = fig.add_subplot(num_samples, 3, (row_idx-1)*3+2)
    #     ax.imshow(lab, cmap='gray')
    #     color_seg(ax, num_classes, lab_bd, rand_line_colors) # Plot segmentation with boundaries
    #     cla_color_name = [rand_line_colors[ll] for ll in uni_all_labs]
    #     cla_color_str = '_\n_'.join(["{}:{}".format(ll, cname[5:]) for ll, cname in zip(uni_all_labs, cla_color_name)])
    #     cla_color_str_color = [rand_line_colors[uni_all_labs[ii//2]] if ii%2==0 else 'black' for ii in range(2*len(uni_all_labs)-1)]
    #     ax_texts = "The true segmentation_\n_{}".format(cla_color_str).split('_')
    #     ax_texts_color = ['black']*2+cla_color_str_color
    #     if len(ax_texts)!=len(ax_texts_color):
    #         raise ValueError("The length of ax_texts({}) and ax_texts_str({}) should be the same.".format(len(ax_texts), len(ax_texts_str)))
    #     rainbow_text(fig, ax, -0.05, -0.15, ax_texts, ax_texts_color, sep='', size=title_size)
    #     # ax.set_title(ansiwrap.fill(, width=FIG_TITLE_WID), size=title_size)

    #     # Predicted segmentation
    #     ax = fig.add_subplot(num_samples, 3, (row_idx-1)*3+3)
    #     ax.imshow(lab, cmap='gray')
    #     # for l_idx in list(range(num_classes)):
    #     #     coord_y, coord_x = np.where(pred==l_idx)
    #     #     ax.scatter(coord_x, coord_y, c=rand_line_colors[l_idx%len(rand_line_colors)], marker='o', s=0.5, alpha=0.2)
    #     color_seg(ax, num_classes, pred, rand_line_colors)
    #     cla_metric_str = '_\n_'.join(["{}_:(tp){:.3f},(dice){:.3f}".format(
    #         ll, c_tp, c_dice) for ll, c_tp, c_dice in zip(uni_all_labs, cla_tp, cla_dice)])
    #     cla_metric_str_color = [rand_line_colors[uni_all_labs[ii//3]] if ii%3==0 else 'black' for ii in range(3*len(uni_all_labs)-1)]
    #     ax_texts = "The segementation with_\n_the true: acc({:.4f})_\n_{}".format(acc, cla_metric_str).split('_')
    #     ax_texts_color = ['black']*4 + cla_metric_str_color
    #     if len(ax_texts)!=len(ax_texts_color):
    #         raise ValueError("The length of ax_texts({}) and ax_texts_str({}) should be the same.".format(len(ax_texts), len(ax_texts_str)))
    #     rainbow_text(fig, ax, -0.05, -0.15, ax_texts, ax_texts_color, sep='', size=title_size)
    #     # ax.set_title(ansiwrap.fill(, width=FIG_TITLE_WID), size=title_size)

    # def construct_model(FLAGS, keep_prob, step_num_classes, input_shapes, pretrained_wei_path, step_kwargs, skip_layer=[]):
    #     if FLAGS.model_name=='Unet':
    #         model = Unet(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=skip_layer)
    #     elif FLAGS.model_name=='DeepLab3Plus': 
    #         model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=skip_layer, input_shapes=input_shapes)
    #     elif FLAGS.model_name=='DeepLab3PlusOrig': 
    #         model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=skip_layer, input_shapes=input_shapes,
    #                                     pretrained_wei_path=pretrained_wei_path,
    #                                     deeplab3plusmodel=orig_deeplab3.DeepLab3PlusOrigModel,
    #                                     deeplab3plusmodel_kwargs=step_kwargs)
    #     elif FLAGS.model_name=='DeepLab3PlusMultiBackboneOrig_mobilenetv2' or FLAGS.model_name=='DeepLab3PlusMultiBackboneOrig_xception': # DeepLab3PlusMultiBackboneOrig_mobilenetv2 or DeepLab3PlusMultiBackboneOrig_xception 
    #         model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=skip_layer, input_shapes=input_shapes,
    #                              pretrained_wei_path=pretrained_wei_path,
    #                              deeplab3plusmodel=orig_multi_backbone_deeplab3.DeepLabV3PlusMultiBackendOrigModel,
    #                              deeplab3plusmodel_kwargs=step_kwargs)
    #     elif FLAGS.model_name=='DeepLab3PlusMultiBackboneModified_mobilenetv2' or FLAGS.model_name=='DeepLab3PlusMultiBackboneModified_xception':
    #         model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=skip_layer, input_shapes=input_shapes,
    #                              pretrained_wei_path=pretrained_wei_path,
    #                              deeplab3plusmodel=modified_multi_backbone_deeplab3.DeepLabV3PlusMultiBackendModifiedModel,
    #                              deeplab3plusmodel_kwargs=step_kwargs)
    #     return model


    """
    Training, Validating, Testing
    """

    if FLAGS.start_rd > 0:
        print("Reload previous model weights.")
        step_num_classes = len(ls_fnames)+FLAGS.start_rd-1
        if FLAGS.model_name!='DeepLab3Plus' and FLAGS.model_name!='Unet': 
            step_kwargs = copy.deepcopy(kwargs)
            step_kwargs.update({'num_classes': step_num_classes})
        else:
            step_kwargs = None
        loaded_model = construct_model(FLAGS, keep_prob, step_num_classes, input_shapes, pretrained_wei_path, step_kwargs, skip_layer=[])
        try:
            loaded_model.load_weights(FLAGS.start_model_path)
            model_path = FLAGS.start_model_path
        except FileNotFoundError as e:
            print(e)
        print("The weights before loading: \n{}.".format(loaded_model.model.model.get_layer(name=last_layer_name).trainable_variables))
        # Generate images in new file for this incremental learning step
        fd_name = '_'.join([FLAGS.gen_fd_prefix, db_fd_name, new_fd_name, str(FLAGS.start_rd-1)])
        step_base_dir = os.path.join(base_dir, fd_name)
        print(step_base_dir)
        
    # for t_i in range(FLAGS.start_rd, len(os.listdir(new_base_dir))+1):
    for t_i in range(FLAGS.start_rd, len(ls_ext_fnames)+1):
        # Path to the textfiles for the trainings and validation set
        # tr_file = '/path/to/train.txt'
        # val_file = '/path/to/val.txt'
        # fd_name = '5_texture_images_5c'
        
        print("==========Iteration {} starts!============".format(t_i))

        # Generate images in new file for this incremental learning step
        fd_name = '_'.join([FLAGS.gen_fd_prefix, db_fd_name, new_fd_name, str(t_i)])
        # root_dir = '~/scratch'
        step_base_dir = os.path.join(base_dir, fd_name)

        # Generate datasets
        if not os.path.exists(step_base_dir):
            os.makedirs(step_base_dir)
        step_ls_fnames = [fn for fn in ls_fnames]
        os.popen('cp {} {}'.format(os.path.join(db_base_dir, '*'+FLAGS.img_ext), step_base_dir))
        if t_i > 0:
            for t_j in range(t_i):
                # new_texture_fname = new_texture_tmpl.replace('*', str(t_j))
                os.popen('cp {} {}'.format(os.path.join(new_base_dir, ls_ext_fnames[t_j]), step_base_dir))
                step_ls_fnames.append(ls_ext_fnames[t_j])
        
        time.sleep(10)

        # Make sure all files has been copied.
        # When the number of images to be copied is large, the time waited is not long enough before all images required are copied.
        while True:
            cnt_img_files = 0
            for fn in os.listdir(step_base_dir):
                if fn.endswith(FLAGS.img_ext):
                    cnt_img_files+=1
            if cnt_img_files == len(step_ls_fnames):
                print("All image files {} has been copied.".format(step_ls_fnames))
                break
            else:
                time.sleep(10)
        
        step_num_epochs = num_epochs+t_i//4
            
        # Must past in the list of filenames to keep the order and label of textures.
        train_size, valid_size, test_size = FLAGS.train_size, FLAGS.valid_size, FLAGS.test_size
        print(step_base_dir, t_i, step_ls_fnames)
        with tf.device('/cpu:0'):
            gen_save_train_valid_test_dataset(step_base_dir, train_size, valid_size, test_size, ls_fnames=step_ls_fnames, 
                                              new_size=IMG_SIZE, pwei_flag=FLAGS.pwei_flag, normp=FLAGS.normp,
                                              trfm_flag=FLAGS.trfm_flag, num_gen_batch=FLAGS.num_gen_batch, nb=FLAGS.nb, max_rots=FLAGS.max_rots)

        tr_file = os.path.join(step_base_dir, 'train.txt')
        val_file = os.path.join(step_base_dir, 'valid.txt')
        
        step_num_classes = num_classes+t_i
        
        if FLAGS.model_name!='DeepLab3Plus' and FLAGS.model_name!='Unet': 
            step_kwargs = copy.deepcopy(kwargs)
            step_kwargs.update({'num_classes': step_num_classes})
        print("The number of classes at step {} is {}.".format(t_i, step_num_classes))

        """
        Main Part of the finetuning Script.
        """

        # Create parent path if it doesn't exist
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Place data loading and preprocessing on the cpu
        with tf.device('/cpu:0'):
            tr_data = HomoTextureDataGenerator(tr_file,
                                    mode='training',
                                    batch_size=batch_size,
                                    num_classes=step_num_classes,
                                    img_norm_flag=FLAGS.img_norm_flag,
                                    trfm_flag=FLAGS.trfm_flag)
            val_data = HomoTextureDataGenerator(val_file,
                                    mode='inference',
                                    batch_size=batch_size,
                                    num_classes=step_num_classes,
                                    img_norm_flag=FLAGS.img_norm_flag,
                                    trfm_flag=FLAGS.trfm_flag)

        # Initialize model
        # # Don't load the trainable layers
        
        # if FLAGS.model_name=='Unet':
        #     model = Unet(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[])
        # elif FLAGS.model_name=='DeepLab3Plus':
        #     model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[], input_shapes=input_shapes)
        # elif FLAGS.model_name=='DeepLab3PlusOrig':
        #     model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[], input_shapes=input_shapes,
        #                          pretrained_wei_path=pretrained_wei_path,
        #                          deeplab3plusmodel=orig_deeplab3.DeepLab3PlusOrigModel,
        #                          deeplab3plusmodel_kwargs=step_kwargs)
        # elif FLAGS.model_name=='DeepLab3PlusMultiBackboneOrig_mobilenetv2' or FLAGS.model_name=='DeepLab3PlusMultiBackboneOrig_xception': # DeepLab3PlusMultiBackboneOrig_mobilenetv2 or DeepLab3PlusMultiBackboneOrig_xception
        #     model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[], input_shapes=input_shapes,
        #                          pretrained_wei_path=pretrained_wei_path,
        #                          deeplab3plusmodel=orig_multi_backbone_deeplab3.DeepLabV3PlusMultiBackendOrigModel,
        #                          deeplab3plusmodel_kwargs=step_kwargs)
        # elif FLAGS.model_name=='DeepLab3PlusMultiBackboneModified_mobilenetv2' or FLAGS.model_name=='DeepLab3PlusMultiBackboneModified_xception': # DeepLab3PlusMultiBackboneModified_mobilenetv2 or DeepLab3PlusMultiBackboneModified_xception
        #     model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[], input_shapes=input_shapes,
        #                          pretrained_wei_path=pretrained_wei_path,
        #                          deeplab3plusmodel=modified_multi_backbone_deeplab3.DeepLabV3PlusMultiBackendModifiedModel,
        #                          deeplab3plusmodel_kwargs=step_kwargs)

        model = construct_model(FLAGS, keep_prob, step_num_classes, input_shapes, pretrained_wei_path, step_kwargs, skip_layer=[])

        # # Still load the trainable layers, but the last layer.
        # model = AlexNet(keep_prob, step_num_classes, ['fc8'], weights_path='../bvlc_alexnet.npy')

        # print(model.model.layers)

        # for var in model.model.variables:
        #     print(var.name, var.trainable)

        # Load the pretrained weights into the model
        if t_i > 0:
            # Load previous weights
            if FLAGS.model_name=='Unet' or FLAGS.model_name=='DeepLab3Plus':
                print("The weights before loading: \n{}.".format(loaded_model.model.get_layer(name=last_layer_name).trainable_variables))
            else:
                print("The weights before loading: \n{}.".format(loaded_model.model.model.get_layer(name=last_layer_name).trainable_variables))
            # model.load_layer_weights_expand_last_layer(model_path)
            # if FLAGS.model_name=='Unet':
            #     loaded_model = Unet(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[])
            # elif FLAGS.model_name=='DeepLab3Plus':
            #     loaded_model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[], input_shapes=input_shapes)
            # elif FLAGS.model_name=='DeepLab3PlusOrig':
            #     loaded_model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[], input_shapes=input_shapes,
            #                                 pretrained_wei_path=pretrained_wei_path,
            #                                 deeplab3plusmodel=orig_deeplab3.DeepLab3PlusOrigModel,
            #                                 deeplab3plusmodel_kwargs=step_kwargs)
            
            model.load_layer_weights_expand_last_layer(loaded_model)

        if FLAGS.model_name=='Unet' or FLAGS.model_name=='DeepLab3Plus':
            print("The weights after loading: \n{}.".format(model.model.get_layer(name=last_layer_name).trainable_variables))
        else:
            print("The weights after loading: \n{}.".format(model.model.model.get_layer(name=last_layer_name).trainable_variables))

        # List of trainable variables of the layers we want to train
        var_list = []
        for lname in train_layers:
            try:
                if FLAGS.model_name=='Unet' or FLAGS.model_name=='DeepLab3Plus':
                    tmp_var = model.model.get_layer(name=lname).trainable_variables
                else:
                    tmp_var = model.model.model.get_layer(name=lname).trainable_variables
            except ValueError as e:
                print("This layer {} doesn't exist.".format(lname))
                continue
            var_list.extend(tmp_var)

        fine_tuning_var_list = []
        if FLAGS.model_name.endswith('mobilenetv2'):
            fine_tuning_layers = deeplabv3plus_mobilenetv2_last_layers
        elif FLAGS.model_name.endswith('xception'):
            if FLAGS.last_layer:
                fine_tuning_layers = deeplabv3plus_xception_last_layers
            else:
                fine_tuning_layers = deeplabv3plus_xception_decoder_layers
        for lname in fine_tuning_layers:
            fine_tuning_var_list.extend(model.model.model.get_layer(name=lname).trainable_variables)

        grad_masks = [np.zeros(var.shape) for var in fine_tuning_var_list]
        for mask in grad_masks:
            mask[...,-1:] = 1.0
        grad_masks = [tf.convert_to_tensor(var, dtype=DTYPE_FLOAT) for var in grad_masks]
        print("The gradient mask is {}.".format(grad_masks))

        # for var in var_list:
        #     print("The variable name is {}.".format(var.name))

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
        valid_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

        # Create filer writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # train_log_dir = os.path.join(filewriter_path, current_time+postfix+'_step_{}'.format(t_i)+'/train')
        # valid_log_dir = os.path.join(filewriter_path, current_time+postfix+'_step_{}'.format(t_i)+'/valid')
        # profile_log_dir = os.path.join(filewriter_path, current_time+postfix+'_step_{}'.format(t_i)+'/profile')
        train_log_dir = os.path.join(filewriter_path, 'log'+postfix+'_step_{}'.format(t_i)+'/train')
        valid_log_dir = os.path.join(filewriter_path, 'log'+postfix+'_step_{}'.format(t_i)+'/valid')
        profile_log_dir = os.path.join(filewriter_path, 'log'+postfix+'_step_{}'.format(t_i)+'/profile')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
        profile_summary_writer = tf.summary.create_file_writer(profile_log_dir) # This can write graph structure.

        # Train the model
        start_time = time.time()

        # Run training for the given number of steps.
        print("{} Start training step {}...".format(datetime.now(), t_i))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), train_log_dir))
        accu_step = 0
        for epoch in range(step_num_epochs):

            print("{} Step {} Epoch number: {}/{}".format(datetime.now(), t_i, epoch+1, step_num_epochs))

            # On cpu (crunch), each epoch takes about 640s. On gpu (colab), each epoch takes about 170s.
            ep_start_time = time.time()
            # Training
            for step, (batch_x, batch_y, batch_y_bd) in enumerate(tr_data.data, 1):
                # Run the optimization to update W and b values.
                
                # Enable the trace
                # tf.summary.trace_on(graph=True, profiler=True)
                if t_i > 0 and epoch < fine_tuning_num_epochs:
                    # Fine tune the last layer.
                    if step == 1:
                        # print(len(fine_tuning_var_list), fine_tuning_layers)
                        print("The last layer is {}.".format(model.model.model.get_layer(name=lname).trainable_variables))
                    if FLAGS.last_layer_mask:
                        train_step(model, batch_x, batch_y, optimizer, fine_tuning_var_list, weight_decay, train_loss, train_regu, train_obj_val, train_accuracy, grad_masks=grad_masks)
                    else:
                        train_step(model, batch_x, batch_y, optimizer, fine_tuning_var_list, weight_decay, train_loss, train_regu, train_obj_val, train_accuracy)
                else:
                    if step == 1:
                        # print(len(var_list), train_layers)
                        print("The last layer is {}.".format(model.model.model.get_layer(name=lname).trainable_variables))
                    train_step(model, batch_x, batch_y, optimizer, var_list, weight_decay, train_loss, train_regu, train_obj_val, train_accuracy)
                # # Log profile tracing
                # with profile_summary_writer.as_default():
                #     tf.summary.trace_export("training_profile", step=accu_step+step, profiler_outdir=profile_log_dir)

                # Log metrics for training
                with train_summary_writer.as_default():
                    print("The training metrics at step {}: loss:{:.4f},regu:{:.4f},obj:{:.4f},acc:{:.4f}.".format(accu_step+step,
                        train_loss.result(), train_regu.result(), train_obj_val.result(), train_accuracy.result()))
                    tf.summary.scalar('loss', train_loss.result(), step=accu_step+step)
                    tf.summary.scalar('regularization', train_regu.result(), step=accu_step+step)
                    tf.summary.scalar('objective value', train_obj_val.result(), step=accu_step+step)
                    tf.summary.scalar('accuracy', train_accuracy.result(), step=accu_step+step)
                    # Reset metrics every step (batch)
                    train_loss.reset_states()
                    train_regu.reset_states()
                    train_obj_val.reset_states()
                    train_accuracy.reset_states()

                if step % display_step == 0:
                    pred = model(batch_x) # Here, we don't use dropout, so that here has some overfitting and is not exactly training metrics.
                    loss = cross_entropy_loss(pred, batch_y)
                    acc = accuracy(pred, batch_y)
                    cla_acc = class_tp(pred, batch_y, step_num_classes)
                    # print(pred[0].shape, batch_y[0].shape, batch_x[0].shape)
                    # print(pred[0].shape, pred[0])
                    fig = plt.figure(figsize=(15,12), facecolor='w')
                    plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy(), pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                                row_idx=1, rand_line_colors=rand_line_colors, num_classes=step_num_classes, plt_dir=plt_dir)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plt_dir, 'step_{}_ep_{}_train_{}.png'.format(t_i, epoch+1, step)))
                    plt.close()
                    # np.savetxt(os.path.join(plt_dir, 'train_true_lab_{}.csv'.format(step,)), batch_y[0].numpy(), fmt='%d', delimiter=',')
                    # np.savetxt(os.path.join(plt_dir, 'train_pred_lab_{}.csv'.format(step,)), pred[0].argmax(axis=-1), fmt='%d', delimiter=',')
                    # print("{} training step: %i, loss: %f, accuracy: %f" % (datetime.now(), step, loss, acc*100))
                    print("{} round {} training step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), t_i, step, loss, acc*100))
                    for lab, c_acc in enumerate(cla_acc):
                        print("The class lab {} has accuracy {:.4f}".format(lab, c_acc))

                if step >= train_batches_per_epoch:
                    accu_step += step
                    break

            # Validation
            for step, (batch_x, batch_y, batch_y_bd) in enumerate(val_data.data, 1):
                # Run validation.
                valid_step(model, batch_x, batch_y, valid_loss, valid_accuracy)
                
                # Log metrics for validation
                with valid_summary_writer.as_default():
                    print("The validating metrics at step {}: loss:{:.4f},acc:{:.4f}.".format(accu_step+step,
                        valid_loss.result(), valid_accuracy.result()))
                    tf.summary.scalar('loss', valid_loss.result(), step=accu_step+step)
                    tf.summary.scalar('accuracy', valid_accuracy.result(), step=accu_step+step)
                    # Reset metrics every step
                    valid_loss.reset_states()
                    valid_accuracy.reset_states()
                
                if step % display_step == 0:
                    pred = model(batch_x)
                    loss = cross_entropy_loss(pred, batch_y)
                    acc = accuracy(pred, batch_y)
                    cla_acc = class_tp(pred, batch_y, step_num_classes)
                    # print(pred[0], batch_y[0], batch_x[0])
                    fig = plt.figure(figsize=(15,12), facecolor='w')
                    plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy(), pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                                row_idx=1, rand_line_colors=rand_line_colors, num_classes=step_num_classes, plt_dir=plt_dir)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plt_dir, 'step_{}_ep_{}_valid_{}.png'.format(t_i, epoch+1, step)))
                    plt.close()
                    # np.savetxt(os.path.join(plt_dir, 'valid_true_lab_{}.csv'.format(step,)), batch_y[0].numpy(), fmt='%d', delimiter=',')
                    # np.savetxt(os.path.join(plt_dir, 'valid_pred_lab_{}.csv'.format(step,)), pred[0].argmax(axis=-1), fmt='%d', delimiter=',')
                    # print("{} validating step: %i, loss: %f, accuracy: %f" % (datetime.now(), step, loss, acc*100))
                    print("{} step {} validating step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), t_i, step, loss, acc*100))
                    for lab, c_acc in enumerate(cla_acc):
                        print("The class lab {} has accuracy {:.4f}".format(lab, c_acc))

                if step >= valid_batches_per_epoch:
                    break

            # # Reset metrics every epoch
            # train_loss.reset_states()
            # train_regu.reset_states()
            # train_obj_val.reset_states()
            # train_accuracy.reset_states()
            # valid_loss.reset_states()
            # valid_accuracy.reset_states()
            
            print("The step %i epoch %i takes %f s." % (t_i, epoch, time.time()-ep_start_time))

        print("Total training for step {} and {} steps is: {}s.".format(t_i, step_num_epochs, time.time()-start_time))

        # Save the model
        # Using pickle would generate error: TypeError: can't pickle weakref objects
        # pickle.dump(model, open(os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+current_time+'.h5'), 'wb'))
        # model_name = 'rand_init_full_train_model_epoch_'+str(step_num_epochs)+current_time+'.h5'
        
        # Talking about how to save weights: https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights
        # Talking about difference in Checkpoints and SavedModel: https://www.tensorflow.org/guide/checkpoint

        # # When enable tf.function, the following saving method would report error.
        # model_name = '_'.join(['step_{}_full_train_model_epoch'.format(t_i), fd_name, postfix, str(step_num_epochs), current_time])+'.h5'
        # model_path = os.path.join(checkpoint_path, model_name)
        # dill.dump(model, open(model_path, 'wb'))
        # loaded_model = dill.load(open(model_path, 'rb'))
        
        # # Save weights instead of the entire model
        model_weights_name = '_'.join(['{}_step_{}_epoch'.format(FLAGS.model_name, t_i), fd_name, postfix, current_time])+'.ckpt'
        model_path = os.path.join(checkpoint_path, model_weights_name)
        model.save_weights(model_path)
        
        # if FLAGS.model_name=='Unet':
        #     loaded_model = Unet(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[])
        # elif FLAGS.model_name=='DeepLab3Plus': 
        #     loaded_model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[], input_shapes=input_shapes)
        # elif FLAGS.model_name=='DeepLab3PlusOrig': 
        #     loaded_model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[], input_shapes=input_shapes,
        #                                 pretrained_wei_path=pretrained_wei_path,
        #                                 deeplab3plusmodel=orig_deeplab3.DeepLab3PlusOrigModel,
        #                                 deeplab3plusmodel_kwargs=step_kwargs)
        # elif FLAGS.model_name=='DeepLab3PlusMultiBackboneOrig_mobilenetv2' or FLAGS.model_name=='DeepLab3PlusMultiBackboneOrig_xception': # DeepLab3PlusMultiBackboneOrig_mobilenetv2 or DeepLab3PlusMultiBackboneOrig_xception 
        #     loaded_model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[], input_shapes=input_shapes,
        #                          pretrained_wei_path=pretrained_wei_path,
        #                          deeplab3plusmodel=orig_multi_backbone_deeplab3.DeepLabV3PlusMultiBackendOrigModel,
        #                          deeplab3plusmodel_kwargs=step_kwargs)
        # elif FLAGS.model_name=='DeepLab3PlusMultiBackboneModified_mobilenetv2' or FLAGS.model_name=='DeepLab3PlusMultiBackboneModified_xception':
        #     loaded_model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[], input_shapes=input_shapes,
        #                          pretrained_wei_path=pretrained_wei_path,
        #                          deeplab3plusmodel=modified_multi_backbone_deeplab3.DeepLabV3PlusMultiBackendModifiedModel,
        #                          deeplab3plusmodel_kwargs=step_kwargs)

        loaded_model = construct_model(FLAGS, keep_prob, step_num_classes, input_shapes, pretrained_wei_path, step_kwargs, skip_layer=[])
        loaded_model.load_weights(model_path)
        
        print("The step {} has model path {}.".format(t_i, model_path))

        # %% Validate on testing datasets
        te_file = os.path.join(step_base_dir, 'test.txt')
        with tf.device('/cpu:0'):
            te_data = HomoTextureDataGenerator(te_file,
                                    mode='inference',
                                    batch_size=batch_size,
                                    num_classes=step_num_classes,
                                    img_norm_flag=FLAGS.img_norm_flag,
                                    trfm_flag=FLAGS.trfm_flag)

        for step, (batch_x, batch_y, batch_y_bd) in enumerate(te_data.data, 1):
            # Run validation.
            valid_step(loaded_model, batch_x, batch_y, valid_loss, valid_accuracy)
            print("The testing metrics at step {}: loss:{:.4f},acc:{:.4f}.".format(step,
                    valid_loss.result(), valid_accuracy.result()))
            
            if step % display_step == 0:
                pred = loaded_model(batch_x)
                loss = cross_entropy_loss(pred, batch_y)
                acc = accuracy(pred, batch_y)
                cla_acc = class_tp(pred, batch_y, step_num_classes)
                fig = plt.figure(figsize=(15,12), facecolor='w')
                plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy(), pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                            row_idx=1, rand_line_colors=rand_line_colors, num_classes=step_num_classes, plt_dir=plt_dir)
                plt.tight_layout()
                plt.savefig(os.path.join(plt_dir, 'step_{}_test_{}.png'.format(t_i, step)))
                plt.close()
                print("{} step {} testing step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), t_i, step, loss, acc*100))
                for lab, c_acc in enumerate(cla_acc):
                        print("The class lab {} has accuracy {:.4f}".format(lab, c_acc))

        print("{} step {} The testing loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), t_i, valid_loss.result(), valid_accuracy.result()))
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        # %%[markdown]
        # Plot all base textures

        all_textures = generate_texture(step_base_dir, ls_fnames=step_ls_fnames, new_size=IMG_SIZE)

        fig = plt.figure(figsize=(step_num_classes*5, 8), facecolor='w')

        for i, texture in enumerate(all_textures):
            fig.add_subplot(1,step_num_classes, i+1)
            plt.imshow(texture.astype(np.uint8), cmap='gray')
            plt.title("Class label: {}.".format(i), size=20)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plt_dir, 'step_{}_all_textures_true_labs.png'.format(t_i)))
        plt.close()

        # %%[markdown]
        # Plot some validation segmentation results
        num_samples = 10

        fig = plt.figure(figsize=(15, 12*num_samples), facecolor='w')

        for i, (image, lab, lab_bd) in enumerate(te_data.data.unbatch().take(num_samples),1):
            image, lab, lab_bd = image.numpy(), lab.numpy(), lab_bd.numpy()

            pred = loaded_model(image)[0].numpy().argmax(axis=-1).astype(np.int32)

            plt_seg_res(image, lab, lab_bd, pred, fig, num_samples, i, rand_line_colors, step_num_classes, plt_dir)

        plt.tight_layout()
        plt.savefig(os.path.join(plt_dir, 'step_{}_{}_seg_res_testing.png'.format(t_i, FLAGS.model_name)))
        plt.close()

        # Remove previous generated files
        os.popen('rm -rf {}'.format(step_base_dir))
        time.sleep(30)

    print("The total time for this program takes {}h.\n".format((time.time()-entire_start_time)/3600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=4,
        help="Number of epochs in each step.")

    parser.add_argument(
        "--fine_tuning_num_epochs",
        type=int,
        default=1,
        help="Number of epochs for fine tuning.")

    parser.add_argument(
        "--decodor",
        type=int,
        default=0,
        help="Whether to fine tune decoder.")

    parser.add_argument(
        "--last_layer",
        type=int,
        default=1,
        help="Whether to fine tune the last layer.")

    parser.add_argument(
        "--last_layer_mask",
        type=int,
        default=1,
        help="Whether to fine tune the newly added class of the last layer.")

    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="The postfix for folder to store results.")

    parser.add_argument(
        "--start_rd",
        type=int,
        default=0,
        help="The default starting number of iterations.")

    parser.add_argument(
        "--start_model_path",
        type=str,
        default="",
        help="The model path for starting iteration.")

    parser.add_argument(
        "--root_dir",
        type=str,
        # crunch: '/home/ghhgkz/scratch/'
        # quest: '/projects/p30309/'
        # default='/projects/p30309/',
        default='/home/ghhgkz/scratch/',
        help="The root directorty.")

    parser.add_argument(
        "--db_fd_name",
        type=str,
        default='base_chosen', #'5v2',
        help="The folder name of database directory.")

    parser.add_argument(
        "--new_fd_name",
        type=str,
        default='extended_chosen', #'5m',
        help="The folder name of new textures directory.")

    parser.add_argument(
        "--img_ext",
        type=str,
        default='.png',
        help="The extension of image files.")

    parser.add_argument(
        "--num_cla_incr",
        type=int,
        default=-1,
        help="The number of incremental classes during incremental learning.")

    parser.add_argument(
        "--gen_fd_prefix",
        type=str,
        default='gen_fd',
        help="The prefix of folder name for generated data set.")

    parser.add_argument(
        "--model_name",
        type=str,
        default='Unet',
        help="The model name to do the segmentation.")

    parser.add_argument(
        "--weights_name",
        type=str,
        default='pascal_voc',
        help="The name of weights used in original DeepLab3Plus model.")

    parser.add_argument(
        "--backbone",
        type=str,
        default='mobilenetv2',
        help="The name of backbone in the DeepLabV3Plus model.") 

    parser.add_argument(
        "--train_size",
        type=int,
        default=20000,
        help="The number of training images.")

    parser.add_argument(
        "--valid_size",
        type=int,
        default=5000,
        help="The number of training images.")

    parser.add_argument(
        "--test_size",
        type=int,
        default=10000,
        help="The number of training images.")

    parser.add_argument(
        "--display_step",
        type=int,
        default=100,
        help="The number of training images.")

    parser.add_argument(
        "--n_threads",
        type=int,
        default=30,
        help="The number of threads for tensorflow.")

    parser.add_argument(
        "--pwei_flag",
        type=int,
        default=0,
        help="The flag whether add random weight in generating texture collages.")

    parser.add_argument(
        "--normp",
        type=float,
        default=2,
        help="The power of norm used to calculate distance for voroni distance and segmentation.")

    parser.add_argument(
        "--img_norm_flag",
        type=int,
        default=1,
        help="Whether to standardize image before loading to mean 0, std 1.")

    parser.add_argument(
        "--trfm_flag",
        type=int,
        default=1,
        help="Whether to transform(scale, rotate, flip) textures during generating images and loading images.")

    parser.add_argument(
        "--num_gen_batch",
        type=int,
        default=1,
        help="The number of batches in generating images (colleages of textures).")

    parser.add_argument(
        "--nb",
        type=int,
        default=1,
        help="The number of neighboring pixels to detect boundary for plotting. For nb=2, means we detect in the 5*5 neighboring pixels centering at the target pixel.")

    parser.add_argument(
        "--max_rots",
        type=int,
        default=3,
        help="The max number of rotation in generating the data set.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)