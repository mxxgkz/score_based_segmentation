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
import argparse
import time
import sys
import pickle
from datetime import datetime
# In order to disable interactive backend when using matplotlib
# https://stackoverflow.com/questions/19518352/tkinter-tclerror-couldnt-connect-to-display-localhost18-0
# https://stackoverflow.com/questions/49284893/matplotlib-while-debugging-in-pycharm-how-to-turn-off-interactive-mode
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg') # Needed for running on quest
import cv2
from generate_collages import *
from datagenerator import *
from utils import *
from unet import *


def main(_):
    # The random seed for np and tf are independent. In order to reproduce results, I need to set both seeds.
    np.random.seed(123)
    tf.random.set_seed(123)
    print(os.environ)

    """
    Configuration Part.

    The model training parts follow examples: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb
    and examples: https://www.tensorflow.org/tensorboard/get_started.

    Also use the TensorBoard examples in the tensorflow page above.

    """

    root_dir = FLAGS.root_dir
    plt_fd_name = '5_texture_images_5v2_5m'
    postfix = FLAGS.postfix
    plt_dir = os.path.join(os.path.expanduser(os.path.join(root_dir, '20200211_Unet_seg_res/figures/')), plt_fd_name+postfix)

    if not os.path.exists(plt_dir):
            os.makedirs(plt_dir)

    base_dir = os.path.expanduser(os.path.join(root_dir, 'Data/texture/Brodatz/{}/'.format(plt_fd_name)))

    db_fd_name = FLAGS.db_fd_name
    db_base_dir = os.path.join(base_dir, db_fd_name)
    ls_fnames = []
    for fn in list(os.listdir(db_base_dir)):
        if fn.endswith('.pgm'):
            ls_fnames.append(fn)
    ls_fnames.sort()
    print(ls_fnames)
    new_fd_name = FLAGS.new_fd_name
    new_base_dir = os.path.join(base_dir, new_fd_name)

    new_texture_tmpl = 'Nat-5m_*.pgm'

    # Learning params
    learning_rate = 0.0001
    num_epochs = FLAGS.num_epochs
    batch_size = 32
    weight_decay= 0.0005 # Caffe style regularization parameter
    keep_prob = 0.5
    num_classes = len(ls_fnames)

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

    # For plotting different colors in segementation
    # line_colors = ['blue', 'red', 'green', 'cyan', 'orange', 'magenta']
    # https://matplotlib.org/tutorials/colors/colors.html
    line_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    rand_line_colors = [co for co in line_colors]
    np.random.shuffle(rand_line_colors)

    # Loss objects
    # train_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # test_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    train_loss_object = SparseSoftDiceLoss(from_logits=True)
    test_loss_object = SparseSoftDiceLoss(from_logits=False)

    # Stochastic gradient descent optimizer.
    optimizer = tf.optimizers.Adam(learning_rate)

    # Define our metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_regu = tf.keras.metrics.Mean('train_regularization', dtype=tf.float32)
    train_obj_val = tf.keras.metrics.Mean('train_objective_value', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('valid_accuracy')

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

    def plt_seg_res(image, lab, pred, fig, num_samples, i, rand_line_colors, num_classes, plt_dir, title_size=20):
        img_size = image.shape[0]
        corr = lab==pred
        acc = np.sum(corr)/img_size**2
        uni_labs = np.unique(lab)
        cla_acc = []
        for ll in uni_labs:
            cla_all = lab==ll
            cla_acc.append(np.sum(corr*cla_all)/
                        np.sum(cla_all))

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
        cla_acc_str = '\n'.join(["{}: {:.4f}".format(ll, c_acc) for ll, c_acc in zip(uni_labs, cla_acc)])
        ax.set_title("The segementation with\nthe true: acc({:.4f})\n{}".format(acc, cla_acc_str), size=title_size)


    """
    Training, Validating, Testing
    """

    if FLAGS.start_rd > 0:
        try:
            loaded_model = dill.load(open(FLAGS.start_model_path,'rb'))
            model_path = FLAGS.start_model_path
        except FileNotFoundError as e:
            print(e)
        
    for t_i in range(FLAGS.start_rd, len(os.listdir(new_base_dir))+1):
        # Path to the textfiles for the trainings and validation set
        # tr_file = '/path/to/train.txt'
        # val_file = '/path/to/val.txt'
        # fd_name = '5_texture_images_5c'

        fd_name = '_'.join([db_fd_name, new_fd_name, str(t_i)])
        # root_dir = '~/scratch'
        step_base_dir = os.path.join(base_dir, fd_name)

        # Generate datasets
        if not os.path.exists(step_base_dir):
            os.makedirs(step_base_dir)
        step_ls_fnames = [fn for fn in ls_fnames]
        os.popen('cp {} {}'.format(os.path.join(db_base_dir, '*.pgm'), step_base_dir))
        if t_i > 0:
            for t_j in range(1, t_i+1):
                new_texture_fname = new_texture_tmpl.replace('*', str(t_j))
                os.popen('cp {} {}'.format(os.path.join(new_base_dir, new_texture_fname), step_base_dir))
                step_ls_fnames.append(new_texture_fname)
        time.sleep(10)
            
        # Must past in the list of filenames to keep the order and label of textures.
        train_size, valid_size, test_size = 5000, 1000, 2000
        print(step_base_dir, t_i, step_ls_fnames)
        gen_save_train_valid_test_dataset(step_base_dir, train_size, valid_size, test_size, ls_fnames=step_ls_fnames)

        tr_file = os.path.join(step_base_dir, 'train.txt')
        val_file = os.path.join(step_base_dir, 'valid.txt')
        
        step_num_classes = num_classes+t_i
        segmentation_regions = step_num_classes
        print("The number of classes at step {} is {}.".format(t_i, step_num_classes))

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
                                        num_classes=step_num_classes)
            val_data = UnetDataGenerator(val_file,
                                        mode='inference',
                                        batch_size=batch_size,
                                        num_classes=step_num_classes)

        # Initialize model
        # # Don't load the trainable layers
        model = Unet(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[])
        # # Still load the trainable layers, but the last layer.
        # model = AlexNet(keep_prob, step_num_classes, ['fc8'], weights_path='../bvlc_alexnet.npy')

        # print(model.model.layers)

        # for var in model.model.variables:
        #     print(var.name, var.trainable)

        # Load the pretrained weights into the model
        if t_i > 0:
            # Load previous weights
            # print("The weights before loading: \n{}.".format(loaded_model.model.get_layer('conv10').trainable_variables))
            # model.load_layer_weights_expand_last_layer(model_path)
            loaded_model = Unet(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[])
            model.load_layer_weights_expand_last_layer(loaded_model)

        print("The weights after loading: \n{}.".format(model.model.get_layer('conv10').trainable_variables))

        # List of trainable variables of the layers we want to train
        var_list = []
        for lname in train_layers:
            var_list.extend(model.model.get_layer(lname).trainable_variables)
        for var in var_list:
            print("The variable name is {}.".format(var.name))

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
        valid_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

        # Create filer writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(filewriter_path, current_time+'/train/')
        valid_log_dir = os.path.join(filewriter_path, current_time+'/valid/')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        # Train the model
        start_time = time.time()

        # Run training for the given number of steps.
        print("{} Start training step {}...".format(datetime.now(), t_i))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), os.path.join(filewriter_path, current_time)))
        for epoch in range(num_epochs):

            print("{} Step {} Epoch number: {}/{}".format(datetime.now(), t_i, epoch+1, num_epochs))

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
                    cla_acc = class_tp(pred, batch_y, step_num_classes)
                    # print(pred[0].shape, batch_y[0].shape, batch_x[0].shape)
                    # print(pred[0], batch_y[0], batch_x[0])
                    fig = plt.figure(figsize=(15,12), facecolor='w')
                    plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                                i=1, rand_line_colors=rand_line_colors, num_classes=step_num_classes, plt_dir=plt_dir)
                    plt.savefig(os.path.join(plt_dir, 'step_{}_ep_{}_train_{}.png'.format(t_i, epoch+1, step)))
                    plt.close()
                    # np.savetxt(os.path.join(plt_dir, 'train_true_lab_{}.csv'.format(step,)), batch_y[0].numpy(), fmt='%d', delimiter=',')
                    # np.savetxt(os.path.join(plt_dir, 'train_pred_lab_{}.csv'.format(step,)), pred[0].numpy().argmax(axis=-1), fmt='%d', delimiter=',')
                    # print("{} training step: %i, loss: %f, accuracy: %f" % (datetime.now(), step, loss, acc*100))
                    print("{} step {} training step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), t_i, step, loss, acc*100))
                    for lab, c_acc in enumerate(cla_acc):
                        print("The class lab {} has accuracy {:.4f}".format(lab, c_acc))

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
                    cla_acc = class_tp(pred, batch_y, step_num_classes)
                    # print(pred[0], batch_y[0], batch_x[0])
                    fig = plt.figure(figsize=(15,12), facecolor='w')
                    plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                                i=1, rand_line_colors=rand_line_colors, num_classes=step_num_classes, plt_dir=plt_dir)
                    plt.savefig(os.path.join(plt_dir, 'step_{}_ep_{}_valid_{}.png'.format(t_i, epoch+1, step)))
                    plt.close()
                    # np.savetxt(os.path.join(plt_dir, 'valid_true_lab_{}.csv'.format(step,)), batch_y[0].numpy(), fmt='%d', delimiter=',')
                    # np.savetxt(os.path.join(plt_dir, 'valid_pred_lab_{}.csv'.format(step,)), pred[0].numpy().argmax(axis=-1), fmt='%d', delimiter=',')
                    # print("{} validating step: %i, loss: %f, accuracy: %f" % (datetime.now(), step, loss, acc*100))
                    print("{} step {} validating step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), t_i, step, loss, acc*100))
                    for lab, c_acc in enumerate(cla_acc):
                        print("The class lab {} has accuracy {:.4f}".format(lab, c_acc))

                if step >= valid_batches_per_epoch:
                    break
            
            # Log metrics for validation
            with valid_summary_writer.as_default():
                tf.summary.scalar('loss', valid_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)

            # Reset metrics every epoch
            train_loss.reset_states()
            train_regu.reset_states()
            train_obj_val.reset_states()
            train_accuracy.reset_states()
            valid_loss.reset_states()
            valid_accuracy.reset_states()
            
            print("The step %i epoch %i takes %f s." % (t_i, epoch, time.time()-ep_start_time))

        print("Total training for step {} and {} steps is: {}s.".format(t_i, num_epochs, time.time()-start_time))

        # Save the model
        # Using pickle would generate error: TypeError: can't pickle weakref objects
        # pickle.dump(model, open(os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+current_time+'.h5'), 'wb'))
        # model_name = 'rand_init_full_train_model_epoch_'+str(num_epochs)+current_time+'.h5'
        
        # Talking about how to save weights: https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights
        # Talking about difference in Checkpoints and SavedModel: https://www.tensorflow.org/guide/checkpoint

        # # When enable tf.function, the following saving method would report error.
        # model_name = '_'.join(['step_{}_full_train_model_epoch'.format(t_i), fd_name, postfix, str(num_epochs), current_time])+'.h5'
        # model_path = os.path.join(checkpoint_path, model_name)
        # dill.dump(model, open(model_path, 'wb'))
        # loaded_model = dill.load(open(model_path, 'rb'))
        
        # # Save weights instead of the entire model
        model_weights_name = '_'.join(['step_{}_full_train_model_epoch'.format(t_i), fd_name, postfix, str(num_epochs), current_time])+'.ckpt'
        model_path = os.path.join(checkpoint_path, model_weights_name)
        model.save_weights(model_path)
        loaded_model = Unet(keep_prob=keep_prob, num_classes=step_num_classes, skip_layer=[])
        loaded_model.load_weights(model_path)
        
        print("The step {} has model path {}.".format(t_i, model_path))

        # %% Validate on testing datasets
        te_file = os.path.join(step_base_dir, 'test.txt')
        te_data = UnetDataGenerator(te_file,
                                    mode='inference',
                                    batch_size=batch_size,
                                    num_classes=step_num_classes)

        for step, (batch_x, batch_y) in enumerate(te_data.data, 1):
            # Run validation.
            valid_step(loaded_model, batch_x, batch_y)
            
            if step % display_step == 0:
                pred = loaded_model(batch_x)
                loss = cross_entropy_loss(pred, batch_y)
                acc = accuracy(pred, batch_y)
                cla_acc = class_tp(pred, batch_y, step_num_classes)
                fig = plt.figure(figsize=(15,12), facecolor='w')
                plt_seg_res(batch_x[0].numpy(), batch_y[0].numpy(), pred[0].numpy().argmax(axis=-1), fig, num_samples=1, 
                            i=1, rand_line_colors=rand_line_colors, num_classes=step_num_classes, plt_dir=plt_dir)
                plt.savefig(os.path.join(plt_dir, 'step_{}_test_{}.png'.format(t_i, step)))
                plt.close()
                print("{} step {} testing step: {}, loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), t_i, step, loss, acc*100))
                for lab, c_acc in enumerate(cla_acc):
                        print("The class lab {} has accuracy {:.4f}".format(lab, c_acc))

        print("{} step {} The testing loss: {:.4f}, accuracy: {:.4f}".format(datetime.now(), t_i, valid_loss.result(), valid_accuracy.result()))
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        # %%[markdown]
        # Plot some validation segmentation results

        all_textures = generate_texture(step_base_dir, ls_fnames=step_ls_fnames)

        fig = plt.figure(figsize=(step_num_classes*5, 8), facecolor='w')

        for i, texture in enumerate(all_textures):
            fig.add_subplot(1,step_num_classes, i+1)
            plt.imshow(texture.astype(np.uint8), cmap='gray')
            plt.title("Class label: {}.".format(i), size=20)
            plt.axis('off')
        plt.savefig(os.path.join(plt_dir, 'step_{}_all_textures_true_labs.png'.format(t_i)))
        plt.close()

        # %%[markdown]
        # Plot some validation segmentation results
        num_samples = 10

        fig = plt.figure(figsize=(15, 12*num_samples), facecolor='w')

        for i, (image, lab) in enumerate(te_data.data.unbatch().take(num_samples),1):
            image, lab = image.numpy(), lab.numpy()

            pred = loaded_model(image).numpy()[0].argmax(axis=-1).astype(np.int32)

            plt_seg_res(image, lab, pred, fig, num_samples, i, rand_line_colors, step_num_classes, plt_dir)

        plt.savefig(os.path.join(plt_dir, 'step_{}_unet_seg_res_testing.png'.format(t_i)))
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=4,
        help="Number of epochs in each step.")

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
        default='/home/ghhgkz/scratch/',
        help="The root directorty.")

    parser.add_argument(
        "--db_fd_name",
        type=str,
        default='5v2',
        help="The folder name of database directory.")

    parser.add_argument(
        "--new_fd_name",
        type=str,
        default='5m',
        help="The folder name of new textures directory.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)