"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

"""

# %%
import os
import sys
import re
import numpy as np
import tensorflow as tf
import argparse
import time
import pickle
import matplotlib.pyplot as plt
from scipy.special import softmax
from PIL import Image
from alexnet import AlexNet
from datagenerator import *
from alexnet_layers import *
from datetime import datetime

"""
Configuration Part.
"""

def main(_):
    # Path to the textfiles for the trainings and validation set
    # tr_file = '/path/to/train.txt'
    # val_file = '/path/to/val.txt'
    # base_dir = os.path.expanduser('~/scratch/Data/texture/Kylberg/')
    base_dir = os.path.join(FLAGS.root_dir, 'Data/texture/Kylberg_images/')
    plt_dir = os.path.join(FLAGS.root_dir, 'Experiments/Kylberg_examples/20210529_Kylberg_cla/figures/{}'.format('kylberg_cla'+FLAGS.postfix))

    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)

    tr_file = os.path.join(base_dir, 'train.txt')
    val_file = os.path.join(base_dir, 'valid.txt')
    te_file = os.path.join(base_dir, 'test.txt')

    dict_lab_cn = {}
    cla_names = []
    lab_cn_path = os.path.join(base_dir, 'class_names.txt')
    with open(lab_cn_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split(' ')
            dict_lab_cn[items[1]] = items[0]
    for ci in range(len(dict_lab_cn)):
        cla_names.append(dict_lab_cn[str(ci)])
    print(cla_names)
    print(dict_lab_cn)

    # Learning params
    learning_rate = 0.0001
    num_epochs = FLAGS.num_epochs
    batch_size = 32
    weight_decay= 0.0005 # Caffe style regularization parameter
    keep_prob = 0.5

    # Network params
    dropout_rate = 1-keep_prob
    num_classes = 28
    if FLAGS.trainable_layers == 'all':
        train_layers = alexnet_all_layers
    elif FLAGS.trainable_layers == 'fc':
        train_layers = alexnet_fc_layers
    # train_layers = ['fc8', 'fc7', 'fc6'] # Only train last few layers.
    # train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1'] # Train all trainable layers.

    # How often we want to write the tf.summary data to disk
    display_step = FLAGS.display_step

    # Path for tf.summary.FileWriter and to store model checkpoints
    # log_dir = os.path.expanduser("~/scratch/logdir/Kylberg/")
    log_dir = os.path.join(FLAGS.root_dir, 'logdir/Kylberg/')
    filewriter_path = os.path.join(log_dir, "finetune_alexnet/tensorboard")
    checkpoint_path = os.path.join(log_dir, "finetune_alexnet/checkpoints")
    if not os.path.isdir(filewriter_path):
        os.makedirs(filewriter_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    """
    Main Part of the finetuning Script.
    """
    # Create parent path if it doesn't exist

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        tr_data = KylbergDataGenerator(tr_file,
                                        mode='training',
                                        batch_size=batch_size,
                                        num_classes=num_classes)
        val_data = KylbergDataGenerator(val_file,
                                        mode='inference',
                                        batch_size=batch_size,
                                        num_classes=num_classes)
        te_data = KylbergDataGenerator(te_file,
                                        mode='inference',
                                        batch_size=batch_size,
                                        num_classes=num_classes)

    # Initialize model
    # # # Load the trainable layers
    skip_layer = []
    # model = AlexNet(keep_prob, num_classes, skip_layer, weights_path=FLAGS.weights_path)
    # Still load the trainable layers, but the last layer.
    print(FLAGS.weights_path)
    model = AlexNet(keep_prob, num_classes, ['fc8'], weights_path=FLAGS.weights_path)

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
        correct_prediction = tf.equal(tf.argmax(y_pred, axis=1, output_type=tf.int32), 
                                    tf.cast(y_true, tf.int32))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def accuracy_top_k(y_pred, y_true, k=FLAGS.top_k):
        # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        lab_idx = tf.argsort(y_pred, axis=-1, direction='DESCENDING', stable=True)
        correct_prediction = tf.reduce_any(
            tf.equal(lab_idx[...,:k], 
                     tf.reshape(tf.cast(y_true, tf.int32), shape=(-1,1))), 
            axis=-1)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # Loss objects
    train_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    valid_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    test_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Stochastic gradient descent optimizer.
    optimizer = tf.optimizers.Adam(learning_rate)

    # Define our metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_regu = tf.keras.metrics.Mean('train_regularization', dtype=tf.float32)
    train_obj_val = tf.keras.metrics.Mean('train_objective_value', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    train_top_k_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(FLAGS.top_k, 'train_top_k_accuracy')
    valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('valid_accuracy')
    valid_top_k_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(FLAGS.top_k, 'valid_top_k_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
    test_top_k_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(FLAGS.top_k, 'test_top_k_accuracy')

    # List of trainable variables of the layers we want to train
    var_list = []
    for lname in train_layers:
        var_list.extend(model.model.get_layer(lname).trainable_variables)
    for var in var_list:
        print("The variable name is {}.".format(var.name))

    # Optimization process. 
    def train_step(model, x, y, optimizer, var_list, weight_decay, train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, train_top_k_accuracy):
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
        train_loss.update_state(loss)
        train_regu.update_state(regu)
        train_obj_val.update_state(obj_val)
        train_accuracy.update_state(y, pred)
        train_top_k_accuracy.update_state(y, pred)
        return pred

    def valid_step(model, x, y, valid_loss_object, valid_loss, valid_accuracy, valid_top_k_accuracy):
        # Forward pass.
        pred = model(x)
        # Compute loss.
        loss = valid_loss_object(y, pred)

        # Log metrics
        valid_loss.update_state(loss)
        valid_accuracy.update_state(y, pred)
        valid_top_k_accuracy.update_state(y, pred)
        return pred

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
    valid_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
    test_batches_per_epoch = int(np.floor(te_data.data_size / batch_size))

    # Create filer writer
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(filewriter_path, current_time+'/train/')
    valid_log_dir = os.path.join(filewriter_path, current_time+'/valid/')
    test_log_dir = os.path.join(filewriter_path, current_time+'/test/')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def plt_cla_res(image, lab, pred, fig, num_samples, ax_idx, cla_names, plt_dir, title_size=12, top_k=5, ncol=4):
        # Top k probabilities and class names
        top_k_prob_idx = pred.argsort()[-1:-(top_k+1):-1]
        top_k_probs = [pred[j] for j in top_k_prob_idx]
        top_k_cla_names = [cla_names[j] for j in top_k_prob_idx]
        
        # Plot image with class name and prob in the title
        ax = fig.add_subplot(num_samples//ncol if num_samples%ncol==0 else num_samples//ncol+1, ncol, ax_idx)
        image = (image-np.min(image))/(np.max(image)-np.min(image))*255
        image = np.array(Image.fromarray(image.astype(np.uint8)).convert(mode='L'))
        image = (image-np.min(image))/(np.max(image)-np.min(image))*255
        ax.imshow(image.astype(np.uint8), cmap = plt.cm.gray) # Have to choose colormap using plt.imshow()
        ax.set_title(('True Class: {}\n'.format(cla_names[lab]) + 
                      '\n'.join(["Class: " + cn.split(',')[0] + ", prob: {:.6f}".format(pr) for cn, pr in zip(top_k_cla_names, top_k_probs)])),
                     size=title_size,
                     horizontalalignment='left',
                     loc='left')

    # Train the model
    start_time = time.time()
    # Run training for the given number of steps.
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), os.path.join(filewriter_path, current_time)))
    accu_step = 0
    val_start_step = 10
    ls_train_acc = []
    ls_train_top_k_acc = []
    for epoch in range(num_epochs):

        print("{} Epoch number: {}/{}".format(datetime.now(), epoch+1, num_epochs))

        # On cpu (crunch), each epoch takes about 640s. On gpu (colab), each epoch takes about 170s.
        ep_start_time = time.time()
        # Training
        for step, (batch_x, batch_y) in enumerate(tr_data.data, 1):
            # Run the optimization to update W and b values.
            batch_pred = train_step(model, batch_x, batch_y, optimizer, var_list, weight_decay, train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, train_top_k_accuracy)
            ls_train_acc.append(accuracy(batch_pred, batch_y))
            ls_train_top_k_acc.append(accuracy_top_k(batch_pred, batch_y))

            # Log metrics for training
            with train_summary_writer.as_default():
                print("The training metrics at step {}: loss:{:.6f},regu:{:.6f},obj:{:.6f},acc:{:.6f},top_{}_acc:{:.6f}.".format(
                    accu_step+step, train_loss.result(), train_regu.result(), train_obj_val.result(), train_accuracy.result(), FLAGS.top_k, train_top_k_accuracy.result()))
                tf.summary.scalar('loss', train_loss.result(), step=accu_step+step)
                tf.summary.scalar('regularization', train_regu.result(), step=accu_step+step)
                tf.summary.scalar('objective value', train_obj_val.result(), step=accu_step+step)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=accu_step+step)
                tf.summary.scalar('top_{}_accuracy'.format(FLAGS.top_k), train_top_k_accuracy.result(), step=accu_step+step)
                # Reset metrics every step (batch) so that we can track the accuracy at each batch, not accumulated accuracy.
                train_loss.reset_states()
                train_regu.reset_states()
                train_obj_val.reset_states()
                train_accuracy.reset_states()
                train_top_k_accuracy.reset_states()

            if step % display_step == 0:
                fig = plt.figure(figsize=(5,5), facecolor='w')
                plt_cla_res(batch_x[0].numpy(), batch_y[0].numpy(), softmax(batch_pred[0].numpy()), fig, num_samples=1, 
                            ax_idx=1, cla_names=cla_names, plt_dir=plt_dir, ncol=1)
                plt.tight_layout()
                plt.savefig(os.path.join(plt_dir, 'ep_{}_train_{}.png'.format(epoch+1, step)))
                plt.close()
                print("{} training step: {}, local accuracy: {:.6f}, local top_{} accuracy: {:.6f}".format(
                    datetime.now(), step, np.mean(ls_train_acc)*100, FLAGS.top_k, np.mean(ls_train_top_k_acc)*100))
                ls_train_acc, ls_train_top_k_acc = [], []

            # Validation
            if accu_step+step == val_start_step:
                val_start_step *= 2
                ls_ave_valid_acc = []
                ls_ave_valid_top_k_acc = []
                ls_valid_acc = []
                ls_valid_top_k_acc = []
                for val_step, (batch_x, batch_y) in enumerate(val_data.data, 1):
                    # Run validation.
                    batch_pred = valid_step(model, batch_x, batch_y, valid_loss_object, valid_loss, valid_accuracy, valid_top_k_accuracy)
                    ls_valid_acc.append(accuracy(batch_pred, batch_y))
                    ls_valid_top_k_acc.append(accuracy_top_k(batch_pred, batch_y))

                    # Log metrics for validation
                    with valid_summary_writer.as_default():
                        print("The validating metrics at val_step {}: loss:{:.6f},acc:{:.6f},top_{}_acc:{:.6f}.".format(
                            step+val_step, valid_loss.result(), valid_accuracy.result(), FLAGS.top_k, valid_top_k_accuracy.result()))
                        tf.summary.scalar('loss', valid_loss.result(), step=step+val_step)
                        tf.summary.scalar('accuracy', valid_accuracy.result(), step=step+val_step)
                        tf.summary.scalar('top_{}_accuracy'.format(FLAGS.top_k), valid_top_k_accuracy.result(), step=step+val_step)
                        # We want to track accumulated validation accuracy
                        # # Reset metrics every val_step
                        # valid_loss.reset_states()
                        # valid_accuracy.reset_states()
                        
                    if val_step % display_step == 0:
                        fig = plt.figure(figsize=(5,5), facecolor='w')
                        plt_cla_res(batch_x[0].numpy(), batch_y[0].numpy(), batch_pred[0].numpy(), fig, num_samples=1, 
                                    ax_idx=1, cla_names=cla_names, plt_dir=plt_dir, ncol=1)
                        plt.tight_layout()
                        plt.savefig(os.path.join(plt_dir, 'ep_{}_valid_{}.png'.format(epoch+1, val_step)))
                        plt.close()
                        print("{} validating val_step: {}, local accuracy: {:.6f}, local top_{} accuracy: {:.6f}".format(
                            datetime.now(), val_step, np.mean(ls_valid_acc)*100, FLAGS.top_k, np.mean(ls_valid_top_k_acc)*100))
                        ls_ave_valid_acc.append(np.mean(ls_valid_acc))
                        ls_ave_valid_top_k_acc.append(np.mean(ls_valid_top_k_acc))
                        ls_valid_acc, ls_valid_top_k_acc = [], []

                    if val_step >= valid_batches_per_epoch:
                        # Reset metrics every epoch
                        # train_loss.reset_states()
                        # train_regu.reset_states()
                        # train_obj_val.reset_states()
                        # train_accuracy.reset_states()
                        valid_loss.reset_states()
                        valid_accuracy.reset_states()
                        valid_top_k_accuracy.reset_states()
                        break
                print("The epoch {} takes {:.6f}s and validation accuracy is {:.6f} and validation top_{} accuracy is {:.6f}.".format(
                    epoch, time.time()-ep_start_time, np.mean(ls_ave_valid_acc)*100, FLAGS.top_k, np.mean(ls_ave_valid_top_k_acc)*100))

            if step >= train_batches_per_epoch:
                accu_step += step
                break

    print("Total training for {} steps is: {}s.".format(num_epochs, time.time()-start_time))

    # Save the model
    # Using pickle would generate error: TypeError: can't pickle weakref objects
    # pickle.dump(model, open(os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+current_time+'.h5'), 'wb'))
    model_weights_name = '_'.join(['{}_epoch'.format(FLAGS.model_name,), FLAGS.postfix])+'.ckpt'
    model_path = os.path.join(checkpoint_path, model_weights_name)
    model.save_weights(model_path)

    print("Model path {}.".format(model_path))

    loaded_model = AlexNet(keep_prob, num_classes, skip_layer, weights_path=FLAGS.weights_path)
    loaded_model.load_weights(model_path)

    # Testing
    ls_ave_test_acc = []
    ls_ave_test_top_k_acc = []
    ls_test_acc = []
    ls_test_top_k_acc = []
    for step, (batch_x, batch_y) in enumerate(val_data.data, 1):
        # Run test.
        batch_pred = valid_step(model, batch_x, batch_y, test_loss_object, test_loss, test_accuracy, test_top_k_accuracy)
        ls_test_acc.append(accuracy(batch_pred, batch_y))
        ls_test_top_k_acc.append(accuracy_top_k(batch_pred, batch_y))

        # Log metrics for testation
        with test_summary_writer.as_default():
            print("The testing metrics at step {}: loss:{:.6f},acc:{:.6f},top_{}_acc:{:.6f}.".format(
                accu_step+step, test_loss.result(), test_accuracy.result(), FLAGS.top_k, test_top_k_accuracy.result()))
            tf.summary.scalar('loss', test_loss.result(), step=accu_step+step)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=accu_step+step)
            tf.summary.scalar('top_{}_accuracy'.format(FLAGS.top_k), test_top_k_accuracy.result(), step=accu_step+step)
            # We want to track accumulated testing accuracy
            # # Reset metrics every step
            # test_loss.reset_states()
            # test_accuracy.reset_states()
            
        if step % display_step == 0:
            fig = plt.figure(figsize=(5,5), facecolor='w')
            plt_cla_res(batch_x[0].numpy(), batch_y[0].numpy(), batch_pred[0].numpy(), fig, num_samples=1, 
                        ax_idx=1, cla_names=cla_names, plt_dir=plt_dir, ncol=1)
            plt.tight_layout()
            plt.savefig(os.path.join(plt_dir, 'ep_{}_test_{}.png'.format(epoch+1, step)))
            plt.close()
            print("{} testing step: {}, local accuracy: {:.6f}, local top_{} accuracy: {:.6f}".format(
                datetime.now(), step, np.mean(ls_test_acc)*100, FLAGS.top_k, np.mean(ls_test_top_k_acc)*100))
            ls_ave_test_acc.append(np.mean(ls_test_acc))
            ls_ave_test_top_k_acc.append(np.mean(ls_test_top_k_acc))
            ls_test_acc, ls_test_top_k_acc = [], []

        if step >= test_batches_per_epoch:
            test_loss.reset_states()
            test_accuracy.reset_states()
            test_top_k_accuracy.reset_states()
            break

    print("The testing accuracy after {} epochs is {:.6f} and top_{} accuracy is {:.6f}.".format(
        num_epochs, np.mean(ls_ave_test_acc)*100, FLAGS.top_k, np.mean(ls_ave_test_top_k_acc)*100))

    # Show some examples of validating texture images

    # Create figure handle
    fig = plt.figure(figsize=(20,280), facecolor='w')

    num_samples = 160

    # Loop over all imageis
    for i, (image, lab) in enumerate(val_data.data.unbatch().take(num_samples), 1):
        image, lab = image.numpy(), lab.numpy()

        # Run the session and calculate the class probability
        pred = model(image).numpy().reshape((-1,))
        
        plt_cla_res(image, lab, pred, fig, num_samples=num_samples, 
                    ax_idx=i, cla_names=cla_names, plt_dir=plt_dir, top_k=FLAGS.top_k, title_size=18, ncol=4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir, '{}_seg_res_testing.png'.format(FLAGS.model_name)))
    plt.close()

    # Plot all true textures and class labels

    fig = plt.figure(figsize=(42,26), facecolor='w')

    all_cls_image = Image.open(os.path.join(base_dir, 'example_imgs_labs.png'))

    plt.imshow(all_cls_image, cmap = plt.cm.gray)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir, 'imgs_labs.png'))
    plt.show()


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
        default='/projects/p30309/neurips2021/',
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
        default='AlexNet',
        help="The model name to do the classification.")

    parser.add_argument(
        "--weights_path",
        type=str,
        default='/projects/p30309/neurips2021/Classification/bvlc_alexnet.npy',
        help="The path of weights used in AlexNet model.")

    parser.add_argument(
        "--trainable_layers",
        type=str,
        default='all',
        help="String to flag which layers ned to be trained.")

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
        default=30,
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

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="The top_k predictions used to calculate accuracy.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)