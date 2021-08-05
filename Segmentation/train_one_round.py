from constants import *
import os
import re
import dill
import numpy as np
import math
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
from DeepLab3.deeplabv3plus_modified_layers import *

def train_one_round_bd(rd_idx, tr_data, val_data, te_data, 
                    log_fd_name, num_classes,
                    checkpoint_path, keep_prob, input_shapes,
                    pretrained_wei_path, loaded_model,
                    train_layers, batch_size, last_layer_name,
                    filewriter_path, optimizer, weight_decay,
                    train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, ls_train_cla_tp,
                    valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp,
                    rand_line_colors, plt_dir,
                    FLAGS, kwargs, cla_names=None, output_prefix=""):
    # root_dir = '~/scratch'
    
    rd_num_epochs = FLAGS.num_epochs+rd_idx//4

    rd_num_classes = num_classes+rd_idx
    # segmentation_regions = rd_num_classes
    if FLAGS.model_name!='Unet': 
        rd_kwargs = copy.deepcopy(kwargs)
        rd_kwargs.update({'num_classes': rd_num_classes})
    print("The number of classes at {} round {} is {}.".format(output_prefix, rd_idx, rd_num_classes))

    """
    Main Part of the finetuning Script.
    """

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    model = construct_model(FLAGS, keep_prob, rd_num_classes, input_shapes, pretrained_wei_path, rd_kwargs, skip_layer=[], 
                            load_model_path=os.path.join(ROOT_DIR, FLAGS.modified_pre_trained_path) if len(FLAGS.modified_pre_trained_path)>0 else "")

    # # Still load the trainable layers, but the last layer.
    # model = AlexNet(keep_prob, rd_num_classes, ['fc8'], weights_path='../bvlc_alexnet.npy')

    # print(model.model.layers)

    # for var in model.model.variables:
    #     print(var.name, var.trainable)

    # Load the pretrained weights into the model
    if rd_idx > 0 and loaded_model is not None:
        # Load previous weights
        if FLAGS.model_name=='Unet' or FLAGS.model_name=='DeepLab3Plus' or FLAGS.model_name=='DeepLab3PlusModified':
            print("The weights before loading: \n{}.".format(loaded_model.model.get_layer(name=last_layer_name).trainable_variables))
        else:
            print("The weights before loading: \n{}.".format(loaded_model.model.model.get_layer(name=last_layer_name).trainable_variables))
        # model.load_layer_weights_expand_last_layer(model_path)
        # if FLAGS.model_name=='Unet':
        #     loaded_model = Unet(keep_prob=keep_prob, num_classes=rd_num_classes, skip_layer=[])
        # elif FLAGS.model_name=='DeepLab3Plus':
        #     loaded_model = DeepLab3Plus(keep_prob=keep_prob, num_classes=rd_num_classes, skip_layer=[], input_shapes=input_shapes)
        # elif FLAGS.model_name=='DeepLab3PlusOrig':
        #     loaded_model = DeepLab3Plus(keep_prob=keep_prob, num_classes=rd_num_classes, skip_layer=[], input_shapes=input_shapes,
        #                                 pretrained_wei_path=pretrained_wei_path,
        #                                 deeplab3plusmodel=orig_deeplab3.DeepLab3PlusOrigModel,
        #                                 deeplab3plusmodel_kwargs=rd_kwargs)
        
        model.load_layer_weights_expand_last_layer(loaded_model)

    if FLAGS.model_name=='Unet' or FLAGS.model_name=='DeepLab3Plus' or FLAGS.model_name=='DeepLab3PlusModified':
        print("The weights after loading: \n{}.".format(model.model.get_layer(name=last_layer_name).trainable_variables))
    else:
        print("The weights after loading: \n{}.".format(model.model.model.get_layer(name=last_layer_name).trainable_variables))

    # List of trainable variables of the layers we want to train
    var_list, fine_tuning_var_list = gen_fine_tuning_var_list(model, train_layers, FLAGS)

    grad_masks = [np.zeros(var.shape) for var in fine_tuning_var_list]
    for mask in grad_masks:
        mask[...,-1:] = 1.0
    grad_masks = [tf.convert_to_tensor(var, dtype=DTYPE_FLOAT) for var in grad_masks]
    # print("The gradient mask is {}.".format(grad_masks))

    # for var in var_list:
    #     print("The variable name is {}.".format(var.name))

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.ceil(tr_data.data_size / batch_size))
    valid_batches_per_epoch = int(np.ceil(val_data.data_size / batch_size))

    # Create filer writer
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir = os.path.join(filewriter_path, current_time+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/train')
    # valid_log_dir = os.path.join(filewriter_path, current_time+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/valid')
    # profile_log_dir = os.path.join(filewriter_path, current_time+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/profile')
    train_log_dir = os.path.join(filewriter_path, 'log'+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/train')
    valid_log_dir = os.path.join(filewriter_path, 'log'+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/valid')
    profile_log_dir = os.path.join(filewriter_path, 'log'+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/profile')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    profile_summary_writer = tf.summary.create_file_writer(profile_log_dir) # This can write graph structure.

    # Train the model
    start_time = time.time()

    # Run training for the given number of steps.
    print("{} Start training {} round {}...".format(datetime.now(), output_prefix, rd_idx))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), train_log_dir))
    accu_step = 0
    val_start_step = 10
    for epoch in range(rd_num_epochs):

        print(output_prefix+" {} round {} Epoch number: {}/{}".format(datetime.now(), rd_idx, epoch+1, rd_num_epochs))

        # On cpu (crunch), each epoch takes about 640s. On gpu (colab), each epoch takes about 170s.
        ep_start_time = time.time()
        # Training
        for step, (batch_x, batch_y, batch_y_bd) in enumerate(tr_data.data, 1):
            # Run the optimization to update W and b values.
            
            # Enable the trace
            # tf.summary.trace_on(graph=True, profiler=True)
            if rd_idx > 0 and epoch < FLAGS.fine_tuning_num_epochs:
                # Fine tune the last layer.
                if step == 1:
                    # print(len(fine_tuning_var_list), fine_tuning_layers)
                    if FLAGS.model_name=='DeepLab3Plus' or FLAGS.model_name=='DeepLab3PlusModified':
                        print("The last layer is {}.".format(model.model.get_layer(name=last_layer_name).trainable_variables))
                    else:
                        print("The last layer is {}.".format(model.model.model.get_layer(name=last_layer_name).trainable_variables))
                if FLAGS.last_layer_mask:
                    train_step(model, batch_x, batch_y, optimizer, fine_tuning_var_list, weight_decay, train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, ls_train_cla_tp, grad_masks=grad_masks)
                else:
                    train_step(model, batch_x, batch_y, optimizer, fine_tuning_var_list, weight_decay, train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, ls_train_cla_tp)
            else:
                if step == 1:
                    # print(len(var_list), train_layers)
                    if FLAGS.model_name=='DeepLab3Plus' or FLAGS.model_name=='DeepLab3PlusModified':
                        print("The last layer is {}.".format(model.model.get_layer(name=last_layer_name).trainable_variables))
                    else:
                        print("The last layer is {}.".format(model.model.model.get_layer(name=last_layer_name).trainable_variables))
                train_step(model, batch_x, batch_y, optimizer, var_list, weight_decay, train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, ls_train_cla_tp)
            # # Log profile tracing
            # with profile_summary_writer.as_default():
            #     tf.summary.trace_export("training_profile", step=accu_step+step, profiler_outdir=profile_log_dir)

            # Log metrics for training
            with train_summary_writer.as_default():
                print("The training metrics at step {}: loss:{:.4f},regu:{:.4f},obj:{:.4f},acc:{:.4f}.".format(
                    accu_step+step, train_loss.result(), train_regu.result(), train_obj_val.result(), train_accuracy.result()))
                tf.summary.scalar('loss', train_loss.result(), step=accu_step+step)
                tf.summary.scalar('regularization', train_regu.result(), step=accu_step+step)
                tf.summary.scalar('objective value', train_obj_val.result(), step=accu_step+step)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=accu_step+step)

            if step % FLAGS.display_step == 0:
                pred = model(batch_x) # Here, we don't use dropout, so that here has some overfitting and is not exactly training metrics.
                loss = cross_entropy_loss(pred, batch_y)
                acc = accuracy(pred, batch_y)
                cla_tp = class_tp(pred, batch_y, rd_num_classes)
                # print(pred[0].shape, batch_y[0].shape, batch_x[0].shape)
                # print(pred[0].shape, pred[0])
                fig = plt.figure(figsize=(15,12), facecolor='w')
                plt_seg_res_non_ol(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy(), pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                            row_idx=1, rand_line_colors=rand_line_colors, num_classes=rd_num_classes, plt_dir=plt_dir, cla_names=cla_names)
                plt.tight_layout()
                plt.savefig(os.path.join(plt_dir, 'rd_{}_ep_{}_train_{}.png'.format(rd_idx, epoch+1, step)))
                plt.close()
                # np.savetxt(os.path.join(plt_dir, 'train_true_lab_{}.csv'.format(step,)), batch_y[0].numpy(), fmt='%d', delimiter=',')
                # np.savetxt(os.path.join(plt_dir, 'train_pred_lab_{}.csv'.format(step,)), pred[0].argmax(axis=-1), fmt='%d', delimiter=',')
                # print("{} training step: %i, loss: %f, accuracy: %f" % (datetime.now(), step, loss, acc*100))
                print(output_prefix+" {} round {} training step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), rd_idx, step, loss, acc*100))
                # The class are not for this example, but accumulated results.
                for lab, train_cla_tp in enumerate(ls_train_cla_tp):
                    print("The class lab {} has tp {:.4f}".format(lab, train_cla_tp.result()))

            # Reset metrics every step (batch)
            train_loss.reset_states()
            train_regu.reset_states()
            train_obj_val.reset_states()
            train_accuracy.reset_states()
            for train_cla_tp in ls_train_cla_tp:
                train_cla_tp.reset_states()

            # Validation
            if accu_step+step == val_start_step:
                val_start_step *= 2
                for val_step, (batch_x, batch_y, batch_y_bd) in enumerate(val_data.data, 1):
                    # Run validation.
                    valid_step(model, batch_x, batch_y, valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp)
                    
                    # Log metrics for validation
                    with valid_summary_writer.as_default():
                        print("The validating metrics at val_step {}: loss:{:.4f},acc:{:.4f}.".format(
                            step+val_step, valid_loss.result(), valid_accuracy.result()))
                        tf.summary.scalar('loss', valid_loss.result(), step=step+val_step)
                        tf.summary.scalar('accuracy', valid_accuracy.result(), step=step+val_step)
                        # # Reset metrics every val_step
                        # valid_loss.reset_states()
                        # valid_accuracy.reset_states()
                    
                    if val_step % FLAGS.display_step == 0:
                        pred = model(batch_x)
                        loss = cross_entropy_loss(pred, batch_y)
                        acc = accuracy(pred, batch_y)
                        cla_tp = class_tp(pred, batch_y, rd_num_classes)
                        # print(pred[0], batch_y[0], batch_x[0])
                        fig = plt.figure(figsize=(15,12), facecolor='w')
                        plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy(), pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                                    row_idx=1, rand_line_colors=rand_line_colors, num_classes=rd_num_classes, plt_dir=plt_dir, cla_names=cla_names)
                        plt.tight_layout()
                        plt.savefig(os.path.join(plt_dir, 'rd_{}_ep_{}_train_{}_valid_{}.png'.format(rd_idx, epoch+1, accu_step+step, val_step)))
                        plt.close()
                        # np.savetxt(os.path.join(plt_dir, 'valid_true_lab_{}.csv'.format(val_step,)), batch_y[0].numpy(), fmt='%d', delimiter=',')
                        # np.savetxt(os.path.join(plt_dir, 'valid_pred_lab_{}.csv'.format(val_step,)), pred[0].argmax(axis=-1), fmt='%d', delimiter=',')
                        # print("{} validating val_step: %i, loss: %f, accuracy: %f" % (datetime.now(), val_step, loss, acc*100))
                        print(output_prefix+" {} round {} validating val_step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), rd_idx, val_step, loss, acc*100))
                        # The class are not for this example, but accumulated results.
                        for lab, valid_cla_tp in enumerate(ls_valid_cla_tp):
                            print("The class lab {} has tp {:.4f}".format(lab, valid_cla_tp.result()))

                    if val_step >= valid_batches_per_epoch:
                        for lab, valid_cla_tp in enumerate(ls_valid_cla_tp):
                            print("Step {}: The class lab {} has tp {:.4f}".format(accu_step+step, lab, valid_cla_tp.result()))
                        valid_loss.reset_states()
                        valid_accuracy.reset_states()
                        for valid_cla_tp in ls_valid_cla_tp:
                            valid_cla_tp.reset_states()
                        break

            if step >= train_batches_per_epoch:
                accu_step += step
                break

        # # Reset metrics every epoch
        # train_loss.reset_states()
        # train_regu.reset_states()
        # train_obj_val.reset_states()
        # train_accuracy.reset_states()
        # valid_loss.reset_states()
        # valid_accuracy.reset_states()
        
        print("The round %i epoch %i takes %f s." % (rd_idx, epoch+1, time.time()-ep_start_time))

    print("Total training for {} round {} and {} steps is: {}s.".format(output_prefix, rd_idx, rd_num_epochs, time.time()-start_time))
    
    # Talking about how to save weights: https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights
    # Talking about difference in Checkpoints and SavedModel: https://www.tensorflow.org/guide/checkpoint
    
    # # Save weights instead of the entire model
    model_weights_name = '_'.join(['{}_rd_{}_epoch'.format(FLAGS.model_name, rd_idx), log_fd_name, FLAGS.postfix, current_time])+'.ckpt'
    model_path = os.path.join(checkpoint_path, model_weights_name)
    model.save_weights(model_path)

    loaded_model = construct_model(FLAGS, keep_prob, rd_num_classes, input_shapes, pretrained_wei_path, rd_kwargs, skip_layer=[])
    loaded_model.load_weights(model_path)
    
    print("The {} round {} has model path {}.".format(output_prefix, rd_idx, model_path))

    for step, (batch_x, batch_y, batch_y_bd) in enumerate(te_data.data, 1):
        # Run validation.
        valid_step(loaded_model, batch_x, batch_y, valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp)
        print("The testing metrics at step {}: loss:{:.4f},acc:{:.4f}.".format(step,
                valid_loss.result(), valid_accuracy.result()))
        
        if step % FLAGS.display_step == 0:
            pred = loaded_model(batch_x)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accuracy(pred, batch_y)
            cla_tp = class_tp(pred, batch_y, rd_num_classes)
            fig = plt.figure(figsize=(15,12), facecolor='w')
            plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy(), pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                        row_idx=1, rand_line_colors=rand_line_colors, num_classes=rd_num_classes, plt_dir=plt_dir, cla_names=cla_names)
            plt.tight_layout()
            plt.savefig(os.path.join(plt_dir, 'rd_{}_test_{}.png'.format(rd_idx, step)))
            plt.close()
            print(output_prefix+" {} round {} testing step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), rd_idx, step, loss, acc*100))
            # The class are not for this example, but accumulated results.
            for lab, valid_cla_tp in enumerate(ls_valid_cla_tp):
                print("The class lab {} has tp {:.4f}".format(lab, valid_cla_tp.result()))

    print(output_prefix+" {} round {} The testing loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), rd_idx, valid_loss.result(), valid_accuracy.result()))
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    for valid_cla_tp in ls_valid_cla_tp:
        valid_cla_tp.reset_states()

    num_samples = 40

    fig = plt.figure(figsize=(15, 12*num_samples), facecolor='w')

    for idx, (image, lab, lab_bd) in enumerate(te_data.data.unbatch().take(num_samples),1):
        image, lab, lab_bd = image.numpy(), lab.numpy(), lab_bd.numpy()

        pred = loaded_model(image)[0].numpy().argmax(axis=-1).astype(np.int32)

        plt_seg_res_non_ol(image, lab, lab_bd, pred, fig, num_samples, idx, rand_line_colors, rd_num_classes, plt_dir, cla_names=cla_names)
        np.savetxt(os.path.join(plt_dir, 'rd_{}_{}_seg_res_testing_{}.csv'.format(rd_idx, FLAGS.model_name, idx)), lab, fmt='%d', delimiter=',')
        np.savetxt(os.path.join(plt_dir, 'rd_{}_{}_seg_res_testing_{}.csv'.format(rd_idx, FLAGS.model_name, idx)), pred, fmt='%d', delimiter=',')

        np.savetxt(os.path.join(plt_dir, 'rd_{}_ep_{}_train_{}_valid_true_lab_{}.csv'.format(rd_idx, epoch, accu_step+step, val_step)), batch_y[0].numpy(), fmt='%d', delimiter=',')
        np.savetxt(os.path.join(plt_dir, 'rd_{}_ep_{}_train_{}_valid_pred_lab_{}.csv'.format(rd_idx, epoch, accu_step+step, val_step)), pred[0].numpy().argmax(axis=-1), fmt='%d', delimiter=',')

    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir, 'rd_{}_{}_seg_res_testing.png'.format(rd_idx, FLAGS.model_name)))
    plt.close()

    return loaded_model


def validation_round(accu_step, step, val_start_step, val_data, valid_summary_writer, rand_line_colors, 
                     rd_num_classes, plt_dir, cla_names, rd_idx, epoch, output_prefix, valid_batches_per_epoch, FLAGS, 
                     model, valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp, val_type_prefix="", bd_img_flag=False):
    val_start_time = time.time()
    for val_step, batch in enumerate(val_data.data, 1):
        
        if bd_img_flag:
            batch_x, batch_y, batch_y_bd = batch
        else:
            batch_x, batch_y = batch
            batch_y_bd = None
        # Run validation.
        valid_step(model, batch_x, batch_y, valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp)
        
        # Log metrics for validation
        with valid_summary_writer.as_default():
            print("{} The accumulated validating metrics at val_step {}: loss:{:.4f},acc:{:.4f}.".format(val_type_prefix, step+val_step,
                valid_loss.result(), valid_accuracy.result()))
            tf.summary.scalar('loss', valid_loss.result(), step=step+val_step)
            tf.summary.scalar('accuracy', valid_accuracy.result(), step=step+val_step)
            # # Reset metrics every val_step
            # valid_loss.reset_states()
            # valid_accuracy.reset_states()
        
        if val_step % (FLAGS.display_step//100) == 0:
            pred = model(batch_x)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accuracy(pred, batch_y)
            cla_tp = class_tp(pred, batch_y, rd_num_classes)
            # print(pred[0], batch_y[0], batch_x[0])
            pickle.dump(batch_x[0].numpy(), open(os.path.join(plt_dir, 'rd_{}_ep_{}_train_{}_valid_orig_img_{}.h5'.format(rd_idx, epoch, accu_step+step, val_step)), 'wb'))
            np.savetxt(os.path.join(plt_dir, 'rd_{}_ep_{}_train_{}_valid_true_lab_{}.csv'.format(rd_idx, epoch, accu_step+step, val_step)), batch_y[0].numpy(), fmt='%d', delimiter=',')
            np.savetxt(os.path.join(plt_dir, 'rd_{}_ep_{}_train_{}_valid_pred_lab_{}.csv'.format(rd_idx, epoch, accu_step+step, val_step)), pred[0].numpy().argmax(axis=-1), fmt='%d', delimiter=',')
            fig = plt.figure(figsize=(15,12), facecolor='w')
            plt_seg_res_non_ol(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy() if batch_y_bd is not None else None, pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                        row_idx=1, rand_line_colors=rand_line_colors, num_classes=rd_num_classes, plt_dir=plt_dir, cla_names=cla_names)
            plt.tight_layout()
            plt.savefig(os.path.join(plt_dir, 'rd_{}_ep_{}_train_{}_valid_{}.png'.format(rd_idx, epoch, accu_step+step, val_step)))
            plt.close()
            
            fig = plt.figure(figsize=(15,12), facecolor='w')
            plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy() if batch_y_bd is not None else None, pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                        row_idx=1, rand_line_colors=rand_line_colors, num_classes=rd_num_classes, plt_dir=plt_dir, cla_names=cla_names)
            plt.tight_layout()
            plt.savefig(os.path.join(plt_dir, 'rd_{}_ep_{}_train_{}_valid_ol_{}.png'.format(rd_idx, epoch, accu_step+step, val_step)))
            plt.close()
            
            # np.savetxt(os.path.join(plt_dir, 'valid_true_lab_{}.csv'.format(val_step,)), batch_y[0].numpy(), fmt='%d', delimiter=',')
            # np.savetxt(os.path.join(plt_dir, 'valid_pred_lab_{}.csv'.format(val_step,)), pred[0].argmax(axis=-1), fmt='%d', delimiter=',')
            # print("{} validating val_step: %i, loss: %f, accuracy: %f" % (datetime.now(), val_step, loss, acc*100))
            print(output_prefix+"{} {} round {} validating val_step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), val_type_prefix, rd_idx, val_step, loss, acc*100))
            # The class are not for this example, but accumulated results.
            for lab, valid_cla_tp in enumerate(ls_valid_cla_tp):
                print("{} The class lab {} has accumulated tp {:.4f}".format(val_type_prefix, lab, valid_cla_tp.result()))

        if val_step >= valid_batches_per_epoch:
            # for lab, valid_cla_tp in enumerate(ls_valid_cla_tp):
            #     print("Step {}: The class lab {} has tp {:.4f}".format(accu_step+step, lab, valid_cla_tp.result()))
            # valid_loss.reset_states()
            # valid_accuracy.reset_states()
            # for valid_cla_tp in ls_valid_cla_tp:
            #     valid_cla_tp.reset_states()
            break

    for lab, valid_cla_tp in enumerate(ls_valid_cla_tp):
        print("{} Step {}: The class lab {} has accumulated tp {:.4f}".format(val_type_prefix, accu_step+step, lab, valid_cla_tp.result()))
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    for valid_cla_tp in ls_valid_cla_tp:
        valid_cla_tp.reset_states()
    print("{} The validation round {:.0f} takes {:.2f}h.".format(val_type_prefix, math.log(val_start_step/FLAGS.init_valid_step)/math.log(2)-1, (time.time()-val_start_time)/3600))


def train_one_round(rd_idx, tr_data, val_data, te_data, 
                    log_fd_name, num_classes,
                    checkpoint_path, keep_prob, input_shapes,
                    pretrained_wei_path, loaded_model,
                    train_layers, last_layer_name,
                    filewriter_path, optimizer, 
                    train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, ls_train_cla_tp,
                    valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp,
                    rand_line_colors, plt_dir,
                    FLAGS, kwargs, cla_names=None, bd_img_flag=False, output_prefix=""):
    # root_dir = '~/scratch'
    
    # rd_num_epochs = FLAGS.num_epochs+rd_idx//4

    rd_num_classes = num_classes+rd_idx
    # segmentation_regions = rd_num_classes
    if FLAGS.model_name!='Unet': 
        rd_kwargs = copy.deepcopy(kwargs)
        rd_kwargs.update({'num_classes': rd_num_classes})
    print("The number of classes at {} round {} is {}.".format(output_prefix, rd_idx, rd_num_classes))

    """
    Main Part of the finetuning Script.
    """

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    model = construct_model(FLAGS, keep_prob, rd_num_classes, input_shapes, pretrained_wei_path, rd_kwargs, skip_layer=[], 
                            load_model_path=os.path.join(ROOT_DIR, FLAGS.modified_pre_trained_path) if len(FLAGS.modified_pre_trained_path)>0 else "")

    # # Still load the trainable layers, but the last layer.
    # model = AlexNet(keep_prob, rd_num_classes, ['fc8'], weights_path='../bvlc_alexnet.npy')

    # print(model.model.layers)

    # for var in model.model.variables:
    #     print(var.name, var.trainable)

    # Load the pretrained weights into the model
    if rd_idx > 0 and loaded_model is not None:
        # Load previous weights
        if FLAGS.model_name=='Unet' or FLAGS.model_name=='DeepLab3Plus' or FLAGS.model_name=='DeepLab3PlusModified':
            print("The weights before loading: \n{}.".format(loaded_model.model.get_layer(name=last_layer_name).trainable_variables))
        else:
            print("The weights before loading: \n{}.".format(loaded_model.model.model.get_layer(name=last_layer_name).trainable_variables))
        
        model.load_layer_weights_expand_last_layer(loaded_model)

    if FLAGS.model_name=='Unet' or FLAGS.model_name=='DeepLab3Plus' or FLAGS.model_name=='DeepLab3PlusModified':
        print("The weights after loading: \n{}.".format(model.model.get_layer(name=last_layer_name).trainable_variables))
    else:
        print("The weights after loading: \n{}.".format(model.model.model.get_layer(name=last_layer_name).trainable_variables))

    # List of trainable variables of the layers we want to train
    var_list, fine_tuning_var_list = gen_fine_tuning_var_list(model, train_layers, FLAGS)

    grad_masks = [np.zeros(var.shape) for var in fine_tuning_var_list]
    for mask in grad_masks:
        mask[...,-1:] = 1.0
    grad_masks = [tf.convert_to_tensor(var, dtype=DTYPE_FLOAT) for var in grad_masks]
    # print("The gradient mask is {}.".format(grad_masks))

    # for var in var_list:
    #     print("The variable name is {}.".format(var.name))

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.ceil(tr_data.data_size / FLAGS.batch_size))
    valid_batches_per_epoch = int(np.ceil(val_data.data_size / FLAGS.val_batch_ratio/ FLAGS.batch_size)/VAL_RATIO)
    tot_valid_batches_per_epoch = int(np.ceil(val_data.data_size / FLAGS.val_batch_ratio/ FLAGS.batch_size))

    # Calculate, for this round, how many epochs are there.
    rd_num_epochs = int(math.ceil((FLAGS.first_valid_step*2**FLAGS.last_idx_valids)/train_batches_per_epoch))

    # Create filer writer
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir = os.path.join(filewriter_path, current_time+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/train')
    # valid_log_dir = os.path.join(filewriter_path, current_time+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/valid')
    # profile_log_dir = os.path.join(filewriter_path, current_time+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/profile')
    train_log_dir = os.path.join(filewriter_path, 'log'+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/train')
    valid_log_dir = os.path.join(filewriter_path, 'log'+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/valid')
    profile_log_dir = os.path.join(filewriter_path, 'log'+FLAGS.postfix+'_rd_{}'.format(rd_idx)+'/profile')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    profile_summary_writer = tf.summary.create_file_writer(profile_log_dir) # This can write graph structure.

    # Train the model
    start_time = time.time()

    # Run training for the given number of steps.
    print("{} Start training {} round {}...".format(datetime.now(), output_prefix, rd_idx))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), train_log_dir))
    accu_step = FLAGS.accu_step
    val_start_step = FLAGS.init_valid_step
    # for epoch in range(rd_num_epochs):
    epoch = accu_step//train_batches_per_epoch+1
    while True:
        print(output_prefix+" {} round {} Epoch number: {}/{}".format(datetime.now(), rd_idx, epoch, rd_num_epochs))

        # On cpu (crunch), each epoch takes about 640s. On gpu (colab), each epoch takes about 170s.
        ep_start_time = time.time()
        # Training
        for step, batch in enumerate(tr_data.data, 1): # step starting from 1
            if bd_img_flag:
                batch_x, batch_y, batch_y_bd = batch
                sample_wei = None
            else:
                batch_x, batch_y, sample_wei = batch
                batch_y_bd = None
                if len(sample_wei.shape)==1:
                    # The sample_wei doesn't have the same shape as batch_y, meaning no sample_wei
                    sample_wei = None

            # Enable the trace
            # tf.summary.trace_on(graph=True, profiler=True)
            if rd_idx > 0 and epoch <= FLAGS.fine_tuning_num_epochs:
                # Fine tune the last layer.
                if step == 1:
                    # print(len(fine_tuning_var_list), fine_tuning_layers)
                    if FLAGS.model_name=='DeepLab3Plus' or FLAGS.model_name=='DeepLab3PlusModified':
                        print("The last layer is {}.".format(model.model.get_layer(name=last_layer_name).trainable_variables))
                    else:
                        print("The last layer is {}.".format(model.model.model.get_layer(name=last_layer_name).trainable_variables))
                if FLAGS.last_layer_mask:
                    train_step(model, batch_x, batch_y, optimizer, fine_tuning_var_list, FLAGS.weight_decay, train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, ls_train_cla_tp, grad_masks=grad_masks, sample_wei=sample_wei)
                else:
                    train_step(model, batch_x, batch_y, optimizer, fine_tuning_var_list, FLAGS.weight_decay, train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, ls_train_cla_tp, sample_wei=sample_wei)
            else:
                if step == 1:
                    # print(len(var_list), train_layers)
                    if FLAGS.model_name=='DeepLab3Plus' or FLAGS.model_name=='DeepLab3PlusModified':
                        print("The last layer is {}.".format(model.model.get_layer(name=last_layer_name).trainable_variables))
                    else:
                        print("The last layer is {}.".format(model.model.model.get_layer(name=last_layer_name).trainable_variables))
                    # print(sample_wei.shape, sample_wei[0], np.unique(sample_wei[0].numpy()))
                train_step(model, batch_x, batch_y, optimizer, var_list, FLAGS.weight_decay, train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, ls_train_cla_tp, sample_wei=sample_wei)
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

            if step % FLAGS.display_step == 0:
                pred = model(batch_x) # Here, we don't use dropout, so that here has some overfitting and is not exactly training metrics.
                loss = cross_entropy_loss(pred, batch_y)
                acc = accuracy(pred, batch_y)
                cla_tp = class_tp(pred, batch_y, rd_num_classes)
                # print(pred[0].shape, batch_y[0].shape, batch_x[0].shape)
                # print(pred[0].shape, pred[0])
                fig = plt.figure(figsize=(15,12), facecolor='w')
                plt_seg_res_non_ol(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy() if batch_y_bd is not None else None, pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                            row_idx=1, rand_line_colors=rand_line_colors, num_classes=rd_num_classes, plt_dir=plt_dir, cla_names=cla_names)
                np.savetxt(os.path.join(plt_dir, 'rd_{}_ep_{}_train_true_lab_{}.csv'.format(rd_idx, epoch, step)), batch_y[0].numpy(), fmt='%d', delimiter=',')
                np.savetxt(os.path.join(plt_dir, 'rd_{}_ep_{}_train_pred_lab_{}.csv'.format(rd_idx, epoch, step)), pred[0].numpy().argmax(axis=-1), fmt='%d', delimiter=',')
                plt.tight_layout()
                plt.savefig(os.path.join(plt_dir, 'rd_{}_ep_{}_train_{}.png'.format(rd_idx, epoch, step)))
                plt.close()
                # print("{} training step: %i, loss: %f, accuracy: %f" % (datetime.now(), step, loss, acc*100))
                print(output_prefix+" {} round {} training step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), rd_idx, step, loss, acc*100))
                # The class are not for this example, but accumulated results.
                for lab, train_cla_tp in enumerate(ls_train_cla_tp):
                    print("The class lab {} has tp {:.4f}".format(lab, train_cla_tp.result()))

            # Reset metrics every step (batch)
            train_loss.reset_states()
            train_regu.reset_states()
            train_obj_val.reset_states()
            train_accuracy.reset_states()
            for train_cla_tp in ls_train_cla_tp:
                train_cla_tp.reset_states()

            # Validation
            if accu_step + step == val_start_step:
                val_start_step *= 2
                if accu_step >= ACCU_STEP_VAL_PLOT:
                    validation_round(accu_step, step, val_start_step, val_data, valid_summary_writer, rand_line_colors, 
                                    rd_num_classes, plt_dir, cla_names, rd_idx, epoch, output_prefix, tot_valid_batches_per_epoch, FLAGS, 
                                    model, valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp, val_type_prefix='Logrithmic validation:', bd_img_flag=bd_img_flag)
                else:
                    validation_round(accu_step, step, val_start_step, val_data, valid_summary_writer, rand_line_colors, 
                                    rd_num_classes, plt_dir, cla_names, rd_idx, epoch, output_prefix, valid_batches_per_epoch, FLAGS, 
                                    model, valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp, val_type_prefix='Logrithmic validation:', bd_img_flag=bd_img_flag)
                # for val_step, (batch_x, batch_y) in enumerate(val_data.data, 1):
                #     # Run validation.
                #     valid_step(model, batch_x, batch_y, valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp)
                    
                #     # Log metrics for validation
                #     with valid_summary_writer.as_default():
                #         print("The accumulated validating metrics at val_step {}: loss:{:.4f},acc:{:.4f}.".format(step+val_step,
                #             valid_loss.result(), valid_accuracy.result()))
                #         tf.summary.scalar('loss', valid_loss.result(), step=step+val_step)
                #         tf.summary.scalar('accuracy', valid_accuracy.result(), step=step+val_step)
                #         # # Reset metrics every val_step
                #         # valid_loss.reset_states()
                #         # valid_accuracy.reset_states()
                    
                #     if val_step % FLAGS.display_step == 0:
                #         pred = model(batch_x)
                #         loss = cross_entropy_loss(pred, batch_y)
                #         acc = accuracy(pred, batch_y)
                #         cla_tp = class_tp(pred, batch_y, rd_num_classes)
                #         # print(pred[0], batch_y[0], batch_x[0])
                #         fig = plt.figure(figsize=(15,12), facecolor='w')
                #         plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), None, pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                #                     row_idx=1, rand_line_colors=rand_line_colors, num_classes=rd_num_classes, plt_dir=plt_dir, cla_names=cla_names)
                #         plt.tight_layout()
                #         plt.savefig(os.path.join(plt_dir, 'rd_{}_ep_{}_train_{}_valid_{}.png'.format(rd_idx, epoch, accu_step+step, val_step)))
                #         plt.close()
                #         # np.savetxt(os.path.join(plt_dir, 'valid_true_lab_{}.csv'.format(val_step,)), batch_y[0].numpy(), fmt='%d', delimiter=',')
                #         # np.savetxt(os.path.join(plt_dir, 'valid_pred_lab_{}.csv'.format(val_step,)), pred[0].argmax(axis=-1), fmt='%d', delimiter=',')
                #         # print("{} validating val_step: %i, loss: %f, accuracy: %f" % (datetime.now(), val_step, loss, acc*100))
                #         print(output_prefix+" {} round {} validating val_step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), rd_idx, val_step, loss, acc*100))
                #         # The class are not for this example, but accumulated results.
                #         for lab, valid_cla_tp in enumerate(ls_valid_cla_tp):
                #             print("The class lab {} has accumulated tp {:.4f}".format(lab, valid_cla_tp.result()))

                #     if val_step >= valid_batches_per_epoch:
                #         # for lab, valid_cla_tp in enumerate(ls_valid_cla_tp):
                #         #     print("Step {}: The class lab {} has tp {:.4f}".format(accu_step+step, lab, valid_cla_tp.result()))
                #         # valid_loss.reset_states()
                #         # valid_accuracy.reset_states()
                #         # for valid_cla_tp in ls_valid_cla_tp:
                #         #     valid_cla_tp.reset_states()
                #         break

                # for lab, valid_cla_tp in enumerate(ls_valid_cla_tp):
                #     print("Step {}: The class lab {} has accumulated tp {:.4f}".format(accu_step+step, lab, valid_cla_tp.result()))
                # valid_loss.reset_states()
                # valid_accuracy.reset_states()
                # for valid_cla_tp in ls_valid_cla_tp:
                #     valid_cla_tp.reset_states()
                # print("The validation round {:.0f} takes {:.2f}h.".format(math.log(val_start_step/FLAGS.init_valid_step)/math.log(2)-1, (time.time()-val_start_time)/3600))

            if step >= train_batches_per_epoch:
                if epoch % FACTOR_VAL_PLOT == 0:
                    validation_round(accu_step, step, val_start_step, val_data, valid_summary_writer, rand_line_colors, 
                                    rd_num_classes, plt_dir, cla_names, rd_idx, epoch, output_prefix, tot_valid_batches_per_epoch, FLAGS, 
                                    model, valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp, val_type_prefix='Epoch validation:', bd_img_flag=bd_img_flag)
                else:
                    validation_round(accu_step, step, val_start_step, val_data, valid_summary_writer, rand_line_colors, 
                                    rd_num_classes, plt_dir, cla_names, rd_idx, epoch, output_prefix, valid_batches_per_epoch, FLAGS, 
                                    model, valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp, val_type_prefix='Epoch validation:', bd_img_flag=bd_img_flag)
                accu_step += step
                break

        # # Reset metrics every epoch
        # train_loss.reset_states()
        # train_regu.reset_states()
        # train_obj_val.reset_states()
        # train_accuracy.reset_states()
        # valid_loss.reset_states()
        # valid_accuracy.reset_states()
        
        print("The round %i epoch %i takes %f s." % (rd_idx, epoch, time.time()-ep_start_time))
        epoch += 1

        if val_start_step > FLAGS.first_valid_step*2**FLAGS.last_idx_valids:
            break

    print("Total training for {} round {} and {} epochs is: {}s.".format(output_prefix, rd_idx, rd_num_epochs, time.time()-start_time))
    
    # Talking about how to save weights: https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights
    # Talking about difference in Checkpoints and SavedModel: https://www.tensorflow.org/guide/checkpoint
    
    # # Save weights instead of the entire model
    model_weights_name = '_'.join(['{}_rd_{}_epoch'.format(FLAGS.model_name, rd_idx), log_fd_name, FLAGS.postfix, current_time])+'.ckpt'
    model_path = os.path.join(checkpoint_path, model_weights_name)
    model.save_weights(model_path)

    loaded_model = construct_model(FLAGS, keep_prob, rd_num_classes, input_shapes, pretrained_wei_path, rd_kwargs, skip_layer=[])
    loaded_model.load_weights(model_path)
    
    print("The {} round {} has model path {}.".format(output_prefix, rd_idx, model_path))

    ls_test_acc = []
    ls_test_cla_tp = []

    for step, batch in enumerate(te_data.data, 1):

        if bd_img_flag:
            batch_x, batch_y, batch_y_bd = batch
        else:
            batch_x, batch_y = batch
            batch_y_bd = None

        # Run validation.
        valid_step(loaded_model, batch_x, batch_y, valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp)
        print("The accumulated testing metrics at step {}: loss:{:.4f},acc:{:.4f}.".format(step,
                valid_loss.result(), valid_accuracy.result()))
        
        # if step % FLAGS.display_step == 0:
        pred = loaded_model(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        cla_tp = class_tp(pred, batch_y, rd_num_classes)
        if step % (FLAGS.display_step//50) == 0:
            pickle.dump(batch_x[0].numpy(), open(os.path.join(plt_dir, 'rd_{}_test_orig_img_{}.h5'.format(rd_idx, step)), 'wb'))
            np.savetxt(os.path.join(plt_dir, 'rd_{}_test_true_lab_{}.csv'.format(rd_idx, step)), batch_y[0].numpy(), fmt='%d', delimiter=',')
            np.savetxt(os.path.join(plt_dir, 'rd_{}_test_pred_lab_{}.csv'.format(rd_idx, step)), pred[0].numpy().argmax(axis=-1), fmt='%d', delimiter=',')
            fig = plt.figure(figsize=(15,12), facecolor='w')
            plt_seg_res_non_ol(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy() if batch_y_bd is not None else None, pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                        row_idx=1, rand_line_colors=rand_line_colors, num_classes=rd_num_classes, plt_dir=plt_dir, cla_names=cla_names)
            plt.tight_layout()
            plt.savefig(os.path.join(plt_dir, 'rd_{}_test_{}.png'.format(rd_idx, step)))
            plt.close()
            
            fig = plt.figure(figsize=(15,12), facecolor='w')
            plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy() if batch_y_bd is not None else None, pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                        row_idx=1, rand_line_colors=rand_line_colors, num_classes=rd_num_classes, plt_dir=plt_dir, cla_names=cla_names)
            plt.tight_layout()
            plt.savefig(os.path.join(plt_dir, 'rd_{}_test_ol_{}.png'.format(rd_idx, step)))
            plt.close()

            # fig = plt.figure(figsize=(15,12), facecolor='w')
            # plt_seg_res_non_ol(batch_x[0].numpy(), batch_y[0].numpy(), batch_y_bd[0].numpy() if batch_y_bd is not None else None, pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
            #             row_idx=1, rand_line_colors=rand_line_colors, num_classes=rd_num_classes, plt_dir=plt_dir, cla_names=cla_names)
            # np.savetxt(os.path.join(plt_dir, 'rd_{}_test_true_lab_{}.csv'.format(rd_idx, step)), batch_y[0].numpy(), fmt='%d', delimiter=',')
            # np.savetxt(os.path.join(plt_dir, 'rd_{}_test_pred_lab_{}.csv'.format(rd_idx, step)), pred[0].numpy().argmax(axis=-1), fmt='%d', delimiter=',')
            # plt.tight_layout()
            # plt.savefig(os.path.join(plt_dir, 'rd_{}_test_{}.png'.format(rd_idx, step)))
            # plt.close()
        print(output_prefix+" {} round {} testing step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), rd_idx, step, loss, acc*100))
        # The class are not for this example, but accumulated results.
        for lab, valid_cla_tp in enumerate(ls_valid_cla_tp):
            print("The class lab {} has accumulated tp {:.4f}".format(lab, valid_cla_tp.result()))
        
        ls_test_acc.append(acc)
        ls_test_cla_tp.append(cla_tp)

    print(output_prefix+" {} round {} The accumulated testing loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), rd_idx, valid_loss.result(), valid_accuracy.result()))
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    for valid_cla_tp in ls_valid_cla_tp:
        valid_cla_tp.reset_states()

    num_samples = 40

    fig = plt.figure(figsize=(15, 12*num_samples), facecolor='w')

    for idx, batch in enumerate(te_data.data.unbatch().take(num_samples),1):
        
        if bd_img_flag:
            image, lab, lab_bd = batch
        else:
            image, lab = batch
            lab_bd = None

        image, lab, lab_bd = image.numpy(), lab.numpy(), lab_bd.numpy() if lab_bd is not None else None
        
        pred = loaded_model(image)[0].numpy().argmax(axis=-1).astype(np.int32)

        plt_seg_res_non_ol(image, lab, lab_bd, pred, fig, num_samples, idx, rand_line_colors, rd_num_classes, plt_dir, cla_names=cla_names)
        np.savetxt(os.path.join(plt_dir, 'rd_{}_{}_seg_res_testing_true_lab_{}.csv'.format(rd_idx, FLAGS.model_name, idx)), lab, fmt='%d', delimiter=',')
        np.savetxt(os.path.join(plt_dir, 'rd_{}_{}_seg_res_testing_pred_lab_{}.csv'.format(rd_idx, FLAGS.model_name, idx)), pred, fmt='%d', delimiter=',')

    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir, 'rd_{}_{}_seg_res_testing.png'.format(rd_idx, FLAGS.model_name)))
    plt.close()

    return loaded_model, np.array(ls_test_acc), np.array(ls_test_cla_tp)