import numpy as np
import os
import tensorflow as tf
import six
import time
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from PIL import Image
from tensorflow.keras.losses import Loss, sparse_categorical_crossentropy
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops.losses import util as tf_losses_util

from constants import *
from Unet.unet import *
from DeepLab3.deeplab3 import *
from DeepLab3.deeplabv3plus_mobilenetv2_layers import *
from DeepLab3.deeplabv3plus_xception_layers import *
from DeepLab3.deeplabv3plus_modified_layers import *
from DeepLab3.deeplabv3plus_full_layers import *


"""
Utility functions
"""

# Optimization process.
# @tf.function # ValueError: tf.function-decorated function tried to create variables on non-first call.
def train_step(model, x, y, optimizer, var_list, weight_decay, train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, ls_train_cla_tp, grad_masks=None, sample_wei=None):
    """
        x: Image array.
        y: Label array. The number of dimension is 1 smaller than x.
    """
    # Wrap computation inside a GradientTape for automatic differentiation.
    # For debugging the interpolation of rotation.
    with tf.GradientTape() as gtape:
        # Forward pass.
        pred = model(x, training=True) # return logits
        # Compute loss.
        loss = tf.cast(train_loss_object(y, pred, sample_weight=sample_wei), dtype=DTYPE_FLOAT) # The built-in function doesn't provide the dtype parameter
        # Be careful about the weight decay in caffe and L2 regularization.
        # https://bbabenko.github.io/weight-decay/
        # https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
        # Cannot past in summed regularized term, because it won't update the regularized term.
        regu = weight_decay*tf.reduce_sum([tf.nn.l2_loss(var) for var in var_list if re.search(r'kernel', var.name)])
        # print("The dtype for loss and regu are {}, {}.".format(loss.dtype, regu.dtype))
        obj_val = loss + regu

    # Compute gradients.
    gradients = gtape.gradient(obj_val, var_list)
    if grad_masks is not None:
        # If mask is not None, element-wise mask out those weights that we don't want to update.
        for idx, mask in enumerate(grad_masks):
            gradients[idx] = gradients[idx]*mask
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, var_list))

    # Log metrics
    train_loss.update_state(loss)
    train_regu.update_state(regu)
    train_obj_val.update_state(obj_val)
    train_accuracy.update_state(y, pred)

    correct_prediction = tf.equal(tf.argmax(pred, axis=-1, output_type=tf.int32), y)
    for lab in np.unique(y.numpy()):
        cla_all = tf.equal(lab, y)        
        cla_corr = tf.logical_and(cla_all, correct_prediction)
        num_cla = tf.reduce_sum(tf.cast(cla_all, tf.float32)).numpy()
        ls_train_cla_tp[lab].update_state(tf.reduce_sum(tf.cast(cla_corr, tf.float32)).numpy()/num_cla)

# @tf.function
def valid_step(model, x, y, valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp):
    # Forward pass.
    pred = model(x)
    # Compute loss.
    # Convert to tensor is necessary
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
    # loss = test_loss_object(tf.convert_to_tensor(y), 
    #                         tf.convert_to_tensor(pred))
    loss = tf.cast(valid_loss_object(y, pred), dtype=DTYPE_FLOAT)

    # Log metrics
    valid_loss.update_state(loss)
    valid_accuracy.update_state(y, pred)

    correct_prediction = tf.equal(tf.argmax(pred, axis=-1, output_type=tf.int32), y)
    for lab in np.unique(y.numpy()):
        cla_all = tf.equal(lab, y)        
        cla_corr = tf.logical_and(cla_all, correct_prediction)
        num_cla = tf.reduce_sum(tf.cast(cla_all, tf.float32)).numpy()
        ls_valid_cla_tp[lab].update_state(tf.reduce_sum(tf.cast(cla_corr, tf.float32)).numpy()/num_cla)

def plt_seg_res(image, lab, lab_bd, pred, fig, num_samples, row_idx, rand_line_colors, num_classes, plt_dir, title_size=20, cla_names=None):
    img_size = image.shape[0]
    # print(lab.dtype, pred.dtype)
    corr = lab==pred
    acc = np.sum(corr)/img_size**2
    # print(np.unique(lab).dtype, np.unique(pred).dtype)
    uni_labs = np.unique(lab).astype(np.int32) # sorted
    uni_preds = np.unique(pred).astype(np.int32)
    # print(np.concatenate((uni_labs, uni_preds), axis=-1).dtype)
    uni_all_labs = np.unique(np.concatenate((uni_labs, uni_preds), axis=-1)).astype(np.int32) # Returns the sorted unique elements of an array
    cla_true_num = []
    cla_tp = [] # True positive rate for each class
    cla_dice = [] # 
    cla_pred_all_num = []
    for ll in uni_all_labs:
        if ll in uni_labs:
            cla_all = lab==ll
            cla_true_num.append(np.sum(cla_all))
            cla_tp.append(np.sum(corr*cla_all)/
                        np.sum(cla_all))
        else:
            cla_tp.append(-1)
        ll_lab, ll_pred = lab==ll, pred==ll
        cla_dice.append(dice(ll_lab, ll_pred))
        cla_pred_all_num.append(np.sum(ll_pred))

    def color_seg(ax, num_classes, img_lab, rand_line_colors):
        for l_idx in range(num_classes+1): # Notice that we have boundary color
            coord_y, coord_x = np.where(img_lab==l_idx)
            # Accepted color format from matplotlib: https://matplotlib.org/api/colors_api.html
            if l_idx == num_classes:
                ax.scatter(coord_x, coord_y, c=rand_line_colors[l_idx], marker='o', s=0.5, alpha=0.8) # Plot boundary
            else:
                ax.scatter(coord_x, coord_y, c=rand_line_colors[l_idx], marker='o', s=0.5, alpha=0.2) # Plot segmentation

    def gen_str_color(txts, labs_idx, rand_line_colors):
        labs_str = '[_{}_]'.format('_,_'.join([str(ele) for ele in txts]))
        labs_color = [rand_line_colors[ll] for ll in labs_idx]
        labs_color = ['black' if ii%2==0 else labs_color[ii//2] for ii in range(2*len(labs_idx)+1)]
        return labs_str, labs_color

    # Original image
    ax = fig.add_subplot(num_samples, 3, (row_idx-1)*3+1)
    image = image - np.min(image)
    image = image/np.max(image)*255
    ax.imshow(image.astype(np.uint8), cmap='gray')

    uni_labs_str, uni_labs_str_color = gen_str_color(uni_labs, uni_labs, rand_line_colors)
    cla_true_num_str, cla_true_num_str_color = gen_str_color(cla_true_num, uni_labs, rand_line_colors)
    uni_preds_str, uni_preds_str_color = gen_str_color(uni_preds, uni_preds, rand_line_colors)
    cla_pred_all_num_str, cla_pred_all_num_str_color = gen_str_color(cla_pred_all_num, uni_all_labs, rand_line_colors)

    ax_texts = "Texture labs in this plot:_\n_{}_\n_Number of True:_\n_{}_\n_Predicted labels:_\n_{}_\n_Number of Predicted:_\n_{}".format(
        uni_labs_str, cla_true_num_str, uni_preds_str, cla_pred_all_num_str).split('_')
    ax_texts_color = ['black']*2+uni_labs_str_color+['black']*3+cla_true_num_str_color+['black']*3+uni_preds_str_color+['black']*3+cla_pred_all_num_str_color
    if len(ax_texts)!=len(ax_texts_color):
        raise ValueError("The length of ax_texts({}) and ax_texts_color({}) should be the same.".format(len(ax_texts), len(ax_texts_color)))
    rainbow_text(fig, ax, -0.05, -0.15, ax_texts, ax_texts_color, sep='', size=title_size)
    # ax.set_title(ansiwrap.fill(, width=FIG_TITLE_WID), size=title_size)
    
    # True segmentation        
    ax = fig.add_subplot(num_samples, 3, (row_idx-1)*3+2)
    ax.imshow(lab, cmap='gray')
    color_seg(ax, num_classes, lab_bd if lab_bd is not None else lab, rand_line_colors) # Plot segmentation with boundaries
    if cla_names is None:
        cla_color_name = [rand_line_colors[ll] for ll in uni_all_labs]
        cla_str = '_\n_'.join(["{}:{}".format(ll, cname[5:]) for ll, cname in zip(uni_all_labs, cla_color_name)])
    else:
        cla_cur_name = [cla_names[ll] for ll in uni_all_labs]
        cla_str = '_\n_'.join(["{}:{}".format(ll, cname) for ll, cname in zip(uni_all_labs, cla_cur_name)])
    cla_color_str_color = [rand_line_colors[uni_all_labs[ii//2]] if ii%2==0 else 'black' for ii in range(2*len(uni_all_labs)-1)]
    ax_texts = "The true segmentation_\n_{}".format(cla_str).split('_')
    ax_texts_color = ['black']*2+cla_color_str_color
    if len(ax_texts)!=len(ax_texts_color):
        raise ValueError("The length of ax_texts({}) and ax_texts_color({}) should be the same.".format(len(ax_texts), len(ax_texts_color)))
    rainbow_text(fig, ax, -0.05, -0.15, ax_texts, ax_texts_color, sep='', size=title_size)
    # ax.set_title(ansiwrap.fill(, width=FIG_TITLE_WID), size=title_size)

    # Predicted segmentation
    ax = fig.add_subplot(num_samples, 3, (row_idx-1)*3+3)
    ax.imshow(lab, cmap='gray')
    # for l_idx in list(range(num_classes)):
    #     coord_y, coord_x = np.where(pred==l_idx)
    #     ax.scatter(coord_x, coord_y, c=rand_line_colors[l_idx%len(rand_line_colors)], marker='o', s=0.5, alpha=0.2)
    color_seg(ax, num_classes, pred, rand_line_colors)
    cla_metric_str = '_\n_'.join(["{}_:(tp){:.3f},(dice){:.3f}".format(
        ll, c_tp, c_dice) for ll, c_tp, c_dice in zip(uni_all_labs, cla_tp, cla_dice)])
    cla_metric_str_color = [rand_line_colors[uni_all_labs[ii//3]] if ii%3==0 else 'black' for ii in range(3*len(uni_all_labs)-1)]
    ax_texts = "The segementation with_\n_the true: acc({:.4f})_\n_{}".format(acc, cla_metric_str).split('_')
    ax_texts_color = ['black']*4 + cla_metric_str_color
    if len(ax_texts)!=len(ax_texts_color):
        raise ValueError("The length of ax_texts({}) and ax_texts_color({}) should be the same.".format(len(ax_texts), len(ax_texts_color)))
    rainbow_text(fig, ax, -0.05, -0.15, ax_texts, ax_texts_color, sep='', size=title_size)
    # ax.set_title(ansiwrap.fill(, width=FIG_TITLE_WID), size=title_size)


def plt_seg_res_non_ol(image, lab, lab_bd, pred, fig, num_samples, row_idx, rand_line_colors, num_classes, plt_dir, title_size=20, cla_names=None):
    """ Segmentation results non-overlap. """
    img_size = image.shape[0]
    # print(lab.dtype, pred.dtype)
    corr = lab==pred
    acc = np.sum(corr)/img_size**2
    # print(np.unique(lab).dtype, np.unique(pred).dtype)
    uni_labs = np.unique(lab).astype(np.int32) # sorted
    uni_preds = np.unique(pred).astype(np.int32)
    # print(np.concatenate((uni_labs, uni_preds), axis=-1).dtype)
    uni_all_labs = np.unique(np.concatenate((uni_labs, uni_preds), axis=-1)).astype(np.int32) # Returns the sorted unique elements of an array
    cla_true_num = []
    cla_tp = [] # True positive rate for each class
    cla_dice = [] # 
    cla_pred_all_num = []
    for ll in uni_all_labs:
        if ll in uni_labs:
            cla_all = lab==ll
            cla_true_num.append(np.sum(cla_all))
            cla_tp.append(np.sum(corr*cla_all)/
                        np.sum(cla_all))
        else:
            cla_tp.append(-1)
        ll_lab, ll_pred = lab==ll, pred==ll
        cla_dice.append(dice(ll_lab, ll_pred))
        cla_pred_all_num.append(np.sum(ll_pred))


    def color_seg(ax, num_classes, img_lab, rand_line_colors):
        for l_idx in range(num_classes+1): # Notice that we have boundary color
            coord_y, coord_x = np.where(img_lab==l_idx)
            # Accepted color format from matplotlib: https://matplotlib.org/api/colors_api.html
            if l_idx == num_classes:
                ax.scatter(coord_x, coord_y, c=rand_line_colors[l_idx], marker='o', s=0.5, alpha=1) # Plot boundary
            else:
                ax.scatter(coord_x, coord_y, c=rand_line_colors[l_idx], marker='o', s=0.5, alpha=1) # Plot segmentation

    def color_seg_err(ax, num_classes, true_lab, pred_lab, rand_line_colors):
        for l_idx in range(num_classes+1): # Notice that we have boundary color
            coord_y, coord_x = np.where(np.logical_and(true_lab==l_idx, pred_lab==l_idx))
            coord_y_err, coord_x_err = np.where(np.logical_and(true_lab==l_idx, pred_lab!=l_idx))
            # Accepted color format from matplotlib: https://matplotlib.org/api/colors_api.html
            
            ax.scatter(coord_x, coord_y, c=rand_line_colors[l_idx], marker='o', s=0.5, alpha=1) # Plot segmentation
            ax.scatter(coord_x_err, coord_y_err, c=ERROR_COL, marker='o', s=0.5, alpha=1) # Plot error pixel

    def gen_str_color(txts, labs_idx, rand_line_colors):
        labs_str = '[_{}_]'.format('_,_'.join([str(ele) for ele in txts]))
        labs_color = [rand_line_colors[ll] for ll in labs_idx]
        labs_color = ['black' if ii%2==0 else labs_color[ii//2] for ii in range(2*len(labs_idx)+1)]
        return labs_str, labs_color

    # Original image
    ax = fig.add_subplot(num_samples, 3, (row_idx-1)*3+1)
    image = image - np.min(image)
    image = image/np.max(image)*255
    ax.imshow(image.astype(np.uint8), cmap='gray')

    uni_labs_str, uni_labs_str_color = gen_str_color(uni_labs, uni_labs, rand_line_colors)
    cla_true_num_str, cla_true_num_str_color = gen_str_color(cla_true_num, uni_labs, rand_line_colors)
    uni_preds_str, uni_preds_str_color = gen_str_color(uni_preds, uni_preds, rand_line_colors)
    cla_pred_all_num_str, cla_pred_all_num_str_color = gen_str_color(cla_pred_all_num, uni_all_labs, rand_line_colors)

    ax_texts = "Texture labs in this plot:_\n_{}_\n_Number of True:_\n_{}_\n_Predicted labels:_\n_{}_\n_Number of Predicted:_\n_{}".format(
        uni_labs_str, cla_true_num_str, uni_preds_str, cla_pred_all_num_str).split('_')
    ax_texts_color = ['black']*2+uni_labs_str_color+['black']*3+cla_true_num_str_color+['black']*3+uni_preds_str_color+['black']*3+cla_pred_all_num_str_color
    if len(ax_texts)!=len(ax_texts_color):
        raise ValueError("The length of ax_texts({}) and ax_texts_color({}) should be the same.".format(len(ax_texts), len(ax_texts_color)))
    rainbow_text(fig, ax, -0.05, -0.15, ax_texts, ax_texts_color, sep='', size=title_size)
    # ax.set_title(ansiwrap.fill(, width=FIG_TITLE_WID), size=title_size)
    
    # True segmentation        
    ax = fig.add_subplot(num_samples, 3, (row_idx-1)*3+2)
    # ax.invert_yaxis()
    # ax.set_aspect(1.0)
    # ax.imshow(np.ones_like(lab)*255, cmap='gray')
    ax.imshow(np.ones_like(lab)*255)
    color_seg(ax, num_classes, lab_bd if lab_bd is not None else lab, rand_line_colors) # Plot segmentation with boundaries
    if cla_names is None:
        cla_color_name = [rand_line_colors[ll] for ll in uni_all_labs]
        cla_str = '_\n_'.join(["{}:{}".format(ll, cname[5:]) for ll, cname in zip(uni_all_labs, cla_color_name)])
    else:
        cla_cur_name = [cla_names[ll] for ll in uni_all_labs]
        cla_str = '_\n_'.join(["{}:{}".format(ll, cname) for ll, cname in zip(uni_all_labs, cla_cur_name)])
    cla_color_str_color = [rand_line_colors[uni_all_labs[ii//2]] if ii%2==0 else 'black' for ii in range(2*len(uni_all_labs)-1)]
    ax_texts = "The true segmentation_\n_{}".format(cla_str).split('_')
    ax_texts_color = ['black']*2+cla_color_str_color
    if len(ax_texts)!=len(ax_texts_color):
        raise ValueError("The length of ax_texts({}) and ax_texts_color({}) should be the same.".format(len(ax_texts), len(ax_texts_color)))
    rainbow_text(fig, ax, -0.05, -0.15, ax_texts, ax_texts_color, sep='', size=title_size)
    # ax.set_title(ansiwrap.fill(, width=FIG_TITLE_WID), size=title_size)

    # Predicted segmentation
    ax = fig.add_subplot(num_samples, 3, (row_idx-1)*3+3)
    # ax.invert_yaxis()
    # ax.set_aspect(1.0)
    # ax.imshow(np.ones_like(lab)*255, cmap='gray')
    ax.imshow(np.ones_like(lab)*255)
    # for l_idx in list(range(num_classes)):
    #     coord_y, coord_x = np.where(pred==l_idx)
    #     ax.scatter(coord_x, coord_y, c=rand_line_colors[l_idx%len(rand_line_colors)], marker='o', s=0.5, alpha=0.2)
    color_seg_err(ax, num_classes, lab, pred, rand_line_colors)
    cla_metric_str = '_\n_'.join(["{}_:(tp){:.3f},(dice){:.3f}".format(
        ll, c_tp, c_dice) for ll, c_tp, c_dice in zip(uni_all_labs, cla_tp, cla_dice)])
    cla_metric_str_color = [rand_line_colors[uni_all_labs[ii//3]] if ii%3==0 else 'black' for ii in range(3*len(uni_all_labs)-1)]
    ax_texts = "The segementation with_\n_the true: acc({:.4f})_\n_{}".format(acc, cla_metric_str).split('_')
    ax_texts_color = ['black']*4 + cla_metric_str_color
    if len(ax_texts)!=len(ax_texts_color):
        raise ValueError("The length of ax_texts({}) and ax_texts_color({}) should be the same.".format(len(ax_texts), len(ax_texts_color)))
    rainbow_text(fig, ax, -0.05, -0.15, ax_texts, ax_texts_color, sep='', size=title_size)
    # ax.set_title(ansiwrap.fill(, width=FIG_TITLE_WID), size=title_size)


def config_deeplab(num_classes, input_shapes, FLAGS):
    if FLAGS.model_name in {'DeepLab3PlusMultiBackboneOrig_mobilenetv2', 'DeepLab3PlusMultiBackboneOrig_xception', 'DeepLab3PlusMultiBackboneModified_mobilenetv2', 'DeepLab3PlusMultiBackboneModified_xception'}:
        keep_prob = 0.9
        dropout_rate = 1-keep_prob
        train_layers = deeplabv3plus_mobilenetv2_layers if FLAGS.model_name.endswith('mobilenetv2') else deeplabv3plus_xception_layers
        last_layer_name = 'custom_logits_semantic'
        backbone = FLAGS.backbone
        alpha = 1.
        output_stride = 8 if FLAGS.model_name.endswith('mobilenetv2') else 16 # Does not useful for mobilenetv2 but useful for xception, because 'mobilenetv2' automatically use output_stride=8
        weights_name = FLAGS.weights_name
        pretrained_wei_path = ''
        kwargs = {'keep_prob': keep_prob,
                  'num_classes': num_classes,
                  'backbone': backbone,
                  'alpha': alpha,
                  'output_stride': output_stride,
                  'input_shapes': input_shapes,
                  'weights_name': weights_name,
                  'weights_path': pretrained_wei_path}
    else:
        if FLAGS.model_name=='Unet':
            keep_prob = 0.5 # For Unet
            dropout_rate = 0.5 # For Unet
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
            last_layer_name = 'conv10'
            pretrained_wei_path = ""
            kwargs = {}

        elif FLAGS.model_name=='DeepLab3Plus':
            keep_prob = 0.9 # For DeepLab3Plus
            dropout_rate = 1-keep_prob # For DeepLab3Plus
            train_layers = deeplabv3plus_full_layers # Train all trainable layers.
            last_layer_name = 'last_conv'
            pretrained_wei_path = ""
            kwargs = {'keep_prob': keep_prob,
                    'num_classes': num_classes,
                    'output_stride': 16,
                    'input_shapes': input_shapes}
        elif FLAGS.model_name=='DeepLab3PlusModified': 
            keep_prob = 0.9 # For DeepLab3Plus
            dropout_rate = 1-keep_prob # For DeepLab3Plus
            train_layers = deeplabv3plus_modified_layers
            last_layer_name = 'last_conv'
            pretrained_wei_path = ""
            kwargs = {'keep_prob': keep_prob,
                    'num_classes': num_classes,
                    'output_stride': 16,
                    'input_shapes': input_shapes}
        elif FLAGS.model_name=='DeepLab3PlusOrig':
            keep_prob = 0.9 # For DeepLab3Plus
            dropout_rate = 1-keep_prob # For DeepLab3Plus
            train_layers = deeplabv3plus_xception_layers # Train all trainable layers.
            last_layer_name = 'custom_logits_semantic'
            pretrained_wei_path = os.path.join(ROOT_DIR, 'Segmentation/DeepLab3/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')
            kwargs = {'keep_prob': keep_prob,
                    'num_classes': num_classes,
                    'output_stride': 16,
                    'input_shapes': input_shapes,
                    'weights_name': FLAGS.weights_name,
                    'weights_path': pretrained_wei_path}
        backbone = None
        alpha = None
        output_stride = None
        weights_name = None
    return (keep_prob, dropout_rate, train_layers, last_layer_name, 
            pretrained_wei_path, kwargs, backbone, alpha, 
            output_stride, weights_name)


def try_makedirs(dir):
    try:
        os.makedirs(dir)
    except FileExistsError as error:
        print("The directory {} exists already: {}".format(dir, error))

def gen_fine_tuning_var_list(model, train_layers, FLAGS):
    var_list = []
    for lname in train_layers:
        try:
            if FLAGS.model_name=='Unet' or FLAGS.model_name=='DeepLab3Plus' or FLAGS.model_name=='DeepLab3PlusModified':
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
    elif FLAGS.model_name == 'DeepLab3Plus':
        if FLAGS.last_layer:
            fine_tuning_layers = deeplabv3plus_full_last_layers
        else:
            fine_tuning_layers = deeplabv3plus_full_decoder_layers
    elif FLAGS.model_name == 'DeepLab3PlusModified':
        if FLAGS.last_layer:
            fine_tuning_layers = deeplabv3plus_modified_last_layers
        else:
            fine_tuning_layers = deeplabv3plus_modified_decoder_layers
    for lname in fine_tuning_layers:
        if FLAGS.model_name=='DeepLab3Plus' or FLAGS.model_name=='DeepLab3PlusModified':
            fine_tuning_var_list.extend(model.model.get_layer(name=lname).trainable_variables)
        else:
            fine_tuning_var_list.extend(model.model.model.get_layer(name=lname).trainable_variables)
    return var_list, fine_tuning_var_list


def new_model(FLAGS, keep_prob, step_num_classes, input_shapes, pretrained_wei_path, step_kwargs, skip_layer=[]):
    if FLAGS.model_name=='Unet':
        model = Unet(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=skip_layer)
    elif FLAGS.model_name=='DeepLab3Plus': 
        model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=skip_layer, input_shapes=input_shapes,
                             pretrained_wei_path=pretrained_wei_path,
                             deeplab3plusmodel=DeepLab3PlusModel,
                             deeplab3plusmodel_kwargs=step_kwargs)
    elif FLAGS.model_name=='DeepLab3PlusModified': 
        model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=skip_layer, input_shapes=input_shapes,
                             pretrained_wei_path=pretrained_wei_path,
                             deeplab3plusmodel=DeepLab3PlusModifiedModel,
                             deeplab3plusmodel_kwargs=step_kwargs)
    elif FLAGS.model_name=='DeepLab3PlusOrig': 
        model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=skip_layer, input_shapes=input_shapes,
                             pretrained_wei_path=pretrained_wei_path,
                             deeplab3plusmodel=orig_deeplab3.DeepLab3PlusOrigModel,
                             deeplab3plusmodel_kwargs=step_kwargs)
    elif FLAGS.model_name=='DeepLab3PlusMultiBackboneOrig_mobilenetv2' or FLAGS.model_name=='DeepLab3PlusMultiBackboneOrig_xception': # DeepLab3PlusMultiBackboneOrig_mobilenetv2 or DeepLab3PlusMultiBackboneOrig_xception 
        model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=skip_layer, input_shapes=input_shapes,
                             pretrained_wei_path=pretrained_wei_path,
                             deeplab3plusmodel=orig_multi_backbone_deeplab3.DeepLabV3PlusMultiBackendOrigModel,
                             deeplab3plusmodel_kwargs=step_kwargs)
    elif FLAGS.model_name=='DeepLab3PlusMultiBackboneModified_mobilenetv2' or FLAGS.model_name=='DeepLab3PlusMultiBackboneModified_xception':
        model = DeepLab3Plus(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=skip_layer, input_shapes=input_shapes,
                             pretrained_wei_path=pretrained_wei_path,
                             deeplab3plusmodel=modified_multi_backbone_deeplab3.DeepLabV3PlusMultiBackendModifiedModel,
                             deeplab3plusmodel_kwargs=step_kwargs)
    return model

def construct_model(FLAGS, keep_prob, step_num_classes, input_shapes, pretrained_wei_path, step_kwargs, skip_layer=[], load_model_path=None, load_model_num_cla=7):
    model = new_model(FLAGS, keep_prob, step_num_classes, input_shapes, pretrained_wei_path, step_kwargs, skip_layer=skip_layer)
    if load_model_path is not None and len(load_model_path)>0:
        try:
            # In case that the load_model_path is non-empty but we should not load weight. For example, when we load weights initially, but during incremental training the initial weights are no longer compatible.
            model.load_weights(load_model_path, by_name=False, skip_mismatch=False) # NotImplementedError: Weights may only be loaded based on topology into Models when loading TensorFlow-formatted weights (got by_name=True to load_weights).
            print("The modified model has been loaded with pre-trained weights from other texture data sets.")
        except ValueError as err:
            print("The modified model has not been loaded with pre-trained weights from other texture data sets: {}.".format(err))
    return model

def open_image(img_path, num_tries=10, time_interval=10):
    cnt = 0
    img = None
    while img is None and cnt < num_tries:
        try:
            img = Image.open(img_path)
        except (IOError, OSError) as e:
            cnt += 1
            print("The image at {} is failed to be opened for {} times.".format(img_path, cnt+1))
            time.sleep(time_interval)
    return img

def get_xkcd_color_str(txt, ll, rand_line_colors, background=False):
    # Get the ANSI escape code SGR parameters to change text color.
    # https://stackoverflow.com/a/45782972/4307919
    # https://stackoverflow.com/a/33206814/4307919
    # https://stackoverflow.com/a/29643643/4307919
    # \033[: escape
    # 1: bold
    # 38;2;{r};{g};{b}m: foreground color
    # 48;2;{r};{g};{b}m: background color
    cname = rand_line_colors[ll%len(rand_line_colors)]
    hval = mcd.XKCD_COLORS[cname][1:] # exclude '#'
    (r, g, b) = tuple(int(hval[i:i+2], 16) for i in (0,2,4))
    return '\033[1;{};2;{};{};{}m{}{}'.format(48 if background else 38, r, g, b, txt, COLRESET)

def get_ax_size(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height

def rainbow_text(fig, ax, x, y, strings, colors, 
                 sep=' ', line_space=1.2, lmt_size=1, ext_ratio=1.2, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    https://matplotlib.org/gallery/text_labels_and_annotations/rainbow_text.html
    https://stackoverflow.com/a/19306776/4307919
    https://stackoverflow.com/a/8482667/4307919

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    # if ax is None:
    #     ax = plt.gca()

    t = ax.transAxes
    ax_width, ax_height = get_ax_size(fig, ax)
    ax_width *= ext_ratio
    # print(ax_width, ax_height)
    canvas = ax.figure.canvas

    emty_text = ax.text(x, y, "", transform=t)
    line_anchor = emty_text.get_transform()
    len_cnt = 0
    for s, c in zip(strings, colors):
        if s == '\n':
            line_anchor += Affine2D().translate(0, -line_space*ex.height)
            len_cnt = 0
            t = line_anchor
            continue

        text = ax.text(x, y, s + sep, color=c, transform=t, **kwargs)
        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()

        if len_cnt+ex.width > (lmt_size-x)*ax_width:
            text.set_visible(False)
            line_anchor += Affine2D().translate(0, -line_space*ex.height)
            len_cnt = 0
            text = ax.text(x, y, s + sep, color=c, transform=line_anchor, **kwargs)
            text.draw(canvas.get_renderer())
            ex = text.get_window_extent()    
        
        len_cnt += ex.width
        t = text.get_transform() + Affine2D().translate(ex.width, 0)


# Note that this will apply 'softmax' to the logits.
# When calculating those metrics, cannot use float16 and int16
def cross_entropy_loss(yhat, y):
    # Convert labels to int 32 for tf cross-entropy function.
    if y.dtype != tf.int32:
        y = tf.cast(y, tf.int32)
    # Apply softmax to logits and compute cross-entropy.
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=yhat) # This is not correct calculation
    loss = sparse_categorical_crossentropy(y_true=y, y_pred=yhat, from_logits=False)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=-1, output_type=tf.int32), 
                                tf.cast(y_true, tf.int32)) # tf.int16 doesn't have implementation
    print("The correct prediction shape {}.".format(correct_prediction.shape))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def class_tp(y_pred, y_true, num_classes):
    # This is actually true positive rate or recall.
    cla_tp = []
    if y_true.dtype != tf.int32:
        y_true = tf.cast(y_true, tf.int32) # This step won't change the original object.
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=-1, output_type=tf.int32), y_true)
    for lab in range(num_classes):
        cla_all = tf.equal(lab, y_true)
        cla_corr = tf.logical_and(cla_all, correct_prediction)
        num_cla = tf.reduce_sum(tf.cast(cla_all, tf.float32)).numpy()
        # print(num_cla)
        if num_cla>0:
            cla_tp.append(tf.reduce_sum(tf.cast(cla_corr, tf.float32)).numpy()/num_cla)
        else:
            cla_tp.append(-1)
    return cla_tp


def dice(y_true, y_pred, epsilon=1e-6):
    # The following line is necessary, because adding two boolean numpy array would give wrong results.
    # For calculating dice not in the tensorflow operation
    y_true, y_pred = y_true.astype(np.float32), y_pred.astype(np.float32)
    numerator = 2. * np.sum(y_pred * y_true)
    # denominator = tf.reduce_sum(tf.square(y_pred) + tf.square(y_true), axes)
    denominator = np.sum(y_pred + y_true)    
    return 1 - numerator / (denominator + epsilon)


def soft_dice_loss(y_true, y_pred, from_logits=False, epsilon=1e-6, dtype=DTYPE_FLOAT): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
    
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022

        Modified from https://www.jeremyjordan.me/semantic-segmentation/
    '''
    
    # skip the batch and class axis for calculating Dice score
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_pred, y_true = tf.cast(y_pred, dtype), tf.cast(y_true, dtype)
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * tf.reduce_sum(y_pred * y_true, axes)
    # denominator = tf.reduce_sum(tf.square(y_pred) + tf.square(y_true), axes)
    denominator = tf.reduce_sum(y_pred + y_true, axes)
    
    return 1 - tf.reduce_mean(numerator / (denominator + epsilon)) # average over classes and batch


def sparse_soft_dice_loss(y_true, y_pred, from_logits=False, epsilon=1e-6, dtype=DTYPE_FLOAT):
    # tf.one_hot default dtype is DTYPE_FLOAT
    # https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/ops/array_ops.py#L3539
    y_true_oh = tf.one_hot(indices=y_true, depth=y_pred.shape[-1])
    return soft_dice_loss(y_true_oh, y_pred, from_logits=from_logits, epsilon=epsilon, dtype=dtype)


class LossFunctionWrapper(Loss):
    """Wraps a loss function in the `Loss` class.
    Args:
        fn: The loss function to wrap, with signature `fn(y_true, y_pred,
        **kwargs)`.
        reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
        Default value is `AUTO`. `AUTO` indicates that the reduction option will
        be determined by the usage context. For almost all cases this defaults to
        `SUM_OVER_BATCH_SIZE`.
        When used with `tf.distribute.Strategy`, outside of built-in training
        loops such as `tf.keras` `compile` and `fit`, using `AUTO` or
        `SUM_OVER_BATCH_SIZE` will raise an error. Please see
        https://www.tensorflow.org/tutorials/distribute/custom_training
        for more details on this.
        name: (Optional) name for the loss.
        **kwargs: The keyword arguments that are passed on to `fn`.

    Reference: https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/losses.py#L180
    """

    def __init__(self,
                fn,
                reduction=losses_utils.ReductionV2.AUTO,
                name=None,
                **kwargs):
        super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.
        Args:
            y_true: Ground truth values.
            y_pred: The predicted values.
        Returns:
            Loss values per sample.
        """
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
            y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(y_pred, y_true)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SparseSoftDiceLoss(LossFunctionWrapper):
    """
        Reference: https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/losses.py#L473
        Other kinds of loss: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
    """
    def __init__(self,
               from_logits=False,
               reduction=losses_utils.ReductionV2.AUTO,
               name='sparse_soft_dice_loss',
               epsilon=1e-6,
               **kwargs):
        super(SparseSoftDiceLoss, self).__init__(
            sparse_soft_dice_loss,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            epsilon=epsilon,
            **kwargs)

class SoftDiceLoss(LossFunctionWrapper):
    def __init__(self,
               from_logits=False,
               reduction=losses_utils.ReductionV2.AUTO,
               name='soft_dice_loss',
               epsilon=1e-6,
               **kwargs):
        super(SoftDiceLoss, self).__init__(
            soft_dice_loss,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            epsilon=epsilon,
            **kwargs)
