# %%
from constants import *
import os
import re
import dill
import numpy as np
import pandas as pd
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

from train_one_round import *
from seg_utils.generate_collages import *
from seg_utils.datagenerator import *
from seg_utils.utils import *
from Unet.unet import *
from DeepLab3.deeplab3 import *
from DeepLab3.deeplabv3plus_mobilenetv2_layers import *
from DeepLab3.deeplabv3plus_xception_layers import *
from DeepLab3.deeplabv3plus_modified_layers import *
from DeepLab3.deeplabv3plus_full_layers import *

def lab_fname_func(fn):
    return fn

def main(_):
    # The random seed for np and tf are independent. In order to reproduce results, I need to set both seeds.
    print("Random seed is {}.".format(FLAGS.rand_seed))
    np.random.seed(FLAGS.rand_seed)
    tf.random.set_seed(FLAGS.rand_seed)
    print(os.environ)
    print(FLAGS)
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

    root_dir = FLAGS.root_dir # Data root directory
    postfix = FLAGS.postfix
    plt_dir = os.path.join(os.path.expanduser(os.path.join(PLT_ROOT_DIR, '20210701_uhcs_data/figures/')), FLAGS.gen_fd_prefix+postfix)

    # if not os.path.exists(plt_dir):
    try_makedirs(plt_dir)

    base_dir = os.path.expanduser(os.path.join(root_dir, 'Data/texture/UHCSA_seg_images/uhcs/'))
    save_dir = os.path.expanduser(os.path.join(os.path.join(base_dir, 'tmp_folder'), FLAGS.gen_fd_prefix+'_cv_proc_data'))

    # new_texture_tmpl = 'Nat-5m_*.pgm'

    # Learning params
    FLAGS.learning_rate = 0.0001 # 0.0005 # Normally we choose 0.0001 # Batch size 32
    # learning_rate = 0.000001 # Batch size 4
    FLAGS.batch_size = BATCH_SIZE
    FLAGS.val_batch_ratio = VAL_BATCH_RATIO
    # weight_decay= FLAGS.weight_decay # Caffe style regularization parameter
    cla_names = ['ferritic', 'proeutectoid', 'spheroidite', 'Widmanstätten']
    num_classes = len(cla_names)
    # input_shapes = (256, 256, 3)
    input_shapes = (IMG_SIZE, IMG_SIZE, 3)
    (keep_prob, dropout_rate, train_layers, last_layer_name, 
     pretrained_wei_path, kwargs, backbone, alpha, output_stride,
     weights_name) = config_deeplab(num_classes, input_shapes, FLAGS)

    # Path for tf.summary.FileWriter and to store model checkpoints
    # log_dir = os.path.expanduser(os.path.join(root_dir, "logdir/Unet/"))
    log_dir = os.path.expanduser(os.path.join(PLT_ROOT_DIR, "logdir/{}/".format(FLAGS.model_name)))
    filewriter_path = os.path.join(log_dir, "finetune_{}/tensorboard".format(FLAGS.model_name))
    checkpoint_path = os.path.join(log_dir, "finetune_{}/checkpoints".format(FLAGS.model_name))

    # For plotting different colors in segementation
    # line_colors = ['blue', 'red', 'green', 'cyan', 'orange', 'magenta']
    # https://matplotlib.org/tutorials/colors/colors.html
    # line_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    # Use xkcd named colors with color names and rgb values. See https://matplotlib.org/tutorials/colors/colors.html
    num_colors = len(mcd.XKCD_COLORS)
    xkcd_color_names = list(mcd.XKCD_COLORS)
    # line_colors = [xkcd_color_names[ci] for ci in np.random.choice(range(num_colors), size=num_classes+1, replace=False)]
    line_colors = ['xkcd:true blue', 'xkcd:baby blue', 'xkcd:clay brown', 'xkcd:black', 'xkcd:pale lime green']
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
    optimizer = tf.optimizers.Adam(FLAGS.learning_rate)

    # Define our metrics for logging
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=DTYPE_FLOAT)
    train_regu = tf.keras.metrics.Mean('train_regularization', dtype=DTYPE_FLOAT)
    train_obj_val = tf.keras.metrics.Mean('train_objective_value', dtype=DTYPE_FLOAT)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy', dtype=DTYPE_FLOAT)
    valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=DTYPE_FLOAT)
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('valid_accuracy', dtype=DTYPE_FLOAT)

    """
    Training, Validating, Testing
    """

    loaded_model = None

    # Start the cross-validation
    if FLAGS.norm_data:
        img_base_dir = os.path.join(base_dir, 'png_images_norm')
        img_lab_base_dir = os.path.join(base_dir, 'png_labels_norm')
    else:
        img_base_dir = os.path.join(base_dir, 'png_images')
        img_lab_base_dir = os.path.join(base_dir, 'png_labels')
    ls_all_fnames = list(os.listdir(img_base_dir))
    ls_all_image_fnames = []
    for fn in ls_all_fnames:
        if fn.endswith('.png'):
            ls_all_image_fnames.append(fn)
    fidx_rand_arr = np.random.choice(len(ls_all_image_fnames), size=len(ls_all_image_fnames), replace=False)
    # Fix the fold index for fair comparison
    # split 1: (rand_seed 123)
    # fidx_rand_arr = [6, 5, 7, 8, 2, 11, 20, 21, 12, 13, 4, 3, 16, 18, 14, 22, 9, 15, 23, 0, 1, 10, 19, 17] 
    # # split 2: (rand_seed 124)
    # fidx_rand_arr = [5,13,3,7,6,4,11,17,18,10,1,12,2,23,15,19,8,16,21,9,14,22,20,0]
    ls_all_image_fnames = ['uhcs1219.png', 'uhcs0312.png', 'uhcs0220.png', 'uhcs0579.png', 'uhcs0124.png', 
                           'uhcs0360.png', 'uhcs0495.png', 'uhcs1648.png', 'uhcs1176.png', 'uhcs0075.png', 
                           'uhcs0477.png', 'uhcs0006.png', 'uhcs0007.png', 'uhcs1150.png', 'uhcs1061.png', 
                           'uhcs1528.png', 'uhcs0599.png', 'uhcs1289.png', 'uhcs0357.png', 'uhcs1579.png', 
                           'uhcs0235.png', 'uhcs0333.png', 'uhcs0596.png', 'uhcs0295.png']
    fold_size = len(fidx_rand_arr)//FLAGS.cv_fold if len(fidx_rand_arr)%FLAGS.cv_fold==0 else len(fidx_rand_arr)//FLAGS.cv_fold+1
    print("The random file index is {}.".format(fidx_rand_arr))
    print("The list of image file names is {}.".format(ls_all_image_fnames))
    print(FLAGS)
    for fold_idx in range(FLAGS.start_fold_idx, min(FLAGS.end_fold_idx, FLAGS.cv_fold)):
        fold_start_time = time.time()

        cv_save_dir = os.path.join(save_dir, 'cv_fold_{}'.format(fold_idx))
        cv_aug_save_dir = os.path.join(cv_save_dir, 'aug_data')
        try_makedirs(cv_aug_save_dir)
        cv_plt_dir = os.path.join(plt_dir, 'cv_fold_{}'.format(fold_idx))
        try_makedirs(cv_plt_dir)

        # Generate images in new file for this incremental learning step
        log_fd_name = '_'.join(['uhcs', str(fold_idx)])
        
        # Generate datasets
        if FLAGS.gen_data_flag:
            tr_img_folder = os.path.join(cv_save_dir, 'train_img')
            tr_img_lab_folder = os.path.join(cv_save_dir, 'train_img_lab')
            val_img_folder = os.path.join(cv_save_dir, 'val_img')
            val_img_lab_folder = os.path.join(cv_save_dir, 'val_img_lab')
            if not os.path.exists(tr_img_folder):
                try_makedirs(tr_img_folder)
                try_makedirs(tr_img_lab_folder)
                try_makedirs(val_img_folder)
                try_makedirs(val_img_lab_folder)
            else:
                os.popen('rm {}'.format(os.path.join(tr_img_folder,'*')))
                os.popen('rm {}'.format(os.path.join(tr_img_lab_folder,'*')))
                os.popen('rm {}'.format(os.path.join(val_img_folder,'*')))
                os.popen('rm {}'.format(os.path.join(val_img_lab_folder,'*')))
                time.sleep(10)
            val_fidx = fidx_rand_arr[fold_idx*fold_size:(fold_idx+1)*fold_size]
            for fidx in fidx_rand_arr:
                if fidx in val_fidx:
                    os.popen('cp {} {}'.format(os.path.join(img_base_dir, ls_all_image_fnames[fidx]), val_img_folder))
                    os.popen('cp {} {}'.format(os.path.join(img_lab_base_dir, ls_all_image_fnames[fidx]), val_img_lab_folder))
                else:
                    os.popen('cp {} {}'.format(os.path.join(img_base_dir, ls_all_image_fnames[fidx]), tr_img_folder))
                    os.popen('cp {} {}'.format(os.path.join(img_lab_base_dir, ls_all_image_fnames[fidx]), tr_img_lab_folder))
            time.sleep(10)

            lab_wei_map, _, _ = gen_save_train_valid_test_patch_dataset(tr_img_folder, tr_img_lab_folder, 
                                                    val_img_folder, val_img_lab_folder,
                                                    val_img_folder, val_img_lab_folder,
                                                    cv_aug_save_dir, lab_fname_func, 
                                                    targ_labs=None, stride=FLAGS.img_stride, patch_size=IMG_SIZE, num_workers=PIPLINE_JOBS, test_img_flag=False)
            pickle.dump(lab_wei_map, open(os.path.join(cv_save_dir, 'lab_wei_map.h5'), 'wb'))
        else:
            lab_wei_map = pickle.load(open(os.path.join(cv_save_dir, 'lab_wei_map.h5'), 'rb'))
        
        print(lab_wei_map)

        tr_file = os.path.join(cv_aug_save_dir, 'train.txt')
        val_file = os.path.join(cv_aug_save_dir, 'valid.txt')

        # Place data loading and preprocessing on the cpu
        # print("The trfm flag is {}.".format(FLAGS.trfm_flag))
        with tf.device('/cpu:0'):
            tr_data = LabTextureDataGenerator(tr_file,
                                    mode='training',
                                    batch_size=FLAGS.batch_size,
                                    num_classes=num_classes,
                                    img_norm_flag=FLAGS.img_norm_flag,
                                    trfm_flag=FLAGS.trfm_flag,
                                    sample_wei_flag=FLAGS.sample_wei_flag,
                                    lab_wei_map=lab_wei_map,
                                    lab_wei_alp=FLAGS.lab_wei_alp)
            val_data = LabTextureDataGenerator(val_file,
                                    mode='inference',
                                    batch_size=FLAGS.batch_size,
                                    num_classes=num_classes,
                                    val_batch_ratio=FLAGS.val_batch_ratio,
                                    img_norm_flag=FLAGS.img_norm_flag,
                                    trfm_flag=False)

        """
        Main Part of the finetuning Script.
        """

        ls_train_cla_tp, ls_valid_cla_tp = [], []
        for lab in range(num_classes):
            ls_train_cla_tp.append(tf.keras.metrics.Mean('train_cla_{}_tp'.format(lab), dtype=DTYPE_FLOAT))
            ls_valid_cla_tp.append(tf.keras.metrics.Mean('valid_cla_{}_tp'.format(lab), dtype=DTYPE_FLOAT))
            
        _, val_acc, val_cla_acc = train_one_round(0, tr_data, val_data, val_data, 
                        log_fd_name, num_classes,
                        checkpoint_path, keep_prob, input_shapes,
                        pretrained_wei_path, loaded_model,
                        train_layers, last_layer_name,
                        filewriter_path, optimizer, 
                        train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, ls_train_cla_tp,
                        valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp,
                        rand_line_colors, cv_plt_dir,
                        FLAGS, kwargs, cla_names=cla_names, output_prefix='fold {}'.format(fold_idx))

        # Remove previous generated files
        os.popen('rm -rf {}'.format(os.path.join(cv_aug_save_dir, '*')))
        time.sleep(180)

        print("--------------Fold {}------------------".format(fold_idx))
        print("Validation accuracy:")
        print(val_acc)
        print("Validation class accuracy:")
        print(val_cla_acc)

        df_val_acc = pd.DataFrame(data=val_acc, columns=['acc'])
        df_val_cla_tp = pd.DataFrame(data=val_cla_acc, columns=[str(idx) for idx in range(num_classes)])
        df_val_acc.to_csv(os.path.join(plt_dir, 'val_acc_fold_{}.csv'.format(fold_idx)))
        df_val_cla_tp.to_csv(os.path.join(plt_dir, 'val_cla_tp_fold_{}.csv'.format(fold_idx)))

        print("The total time for fold {} is {}h.\n".format(fold_idx, (time.time()-fold_start_time)/3600))

    # Save configurations
    FLAGS.fidx_rand_arr = fidx_rand_arr
    FLAGS.ls_all_image_fnames = ls_all_image_fnames 
    pickle.dump(FLAGS, open(os.path.join(save_dir, 'flags.h5'), 'wb'))
    with open(os.path.join(CODE_ROOT_DIR, './Segmentation/constants.py'), 'r') as readf:
        const_lines = readf.readlines()
    pickle.dump(const_lines, open(os.path.join(save_dir, 'constants.h5'), 'wb'))

    print("The total time for this program takes {}h.\n".format((time.time()-entire_start_time)/3600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=127,
        help="The random seed of this program.")

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=4,
        help="Number of epochs in each step.")

    parser.add_argument(
        "--last_idx_valids",
        type=int,
        default=12,
        help="Number of validation rounds.")

    parser.add_argument(
        "--accu_step",
        type=int,
        default=0,
        help="The number of steps has been accumulated before.")

    # parser.add_argument(
    #     "--accu_epoch",
    #     type=int,
    #     default=0,
    #     help="The number of epochs has been accumulated before.")

    parser.add_argument(
        "--first_valid_step",
        type=int,
        default=10,
        help="The training step to start validation.")
    
    parser.add_argument(
        "--init_valid_step",
        type=int,
        default=10,
        help="The training step to start validation.")

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
        # default='/home/ghhgkz/scratch/',
        # default='/pylon5/mrz3a6p/ghhgkz/',
        default=ROOT_DIR,
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
        # default='pascal_voc', # Will report error
        default='',
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

    parser.add_argument(
        "--sample_wei_flag",
        type=int,
        default=1,
        help="Whether to use sample weight to combat class imbalance.")

    parser.add_argument(
        "--lab_wei_alp",
        type=float,
        default=0.5,
        help="The power for sample weights.")

    parser.add_argument(
        "--gen_data_flag",
        type=int,
        default=1,
        help="Whether to generate data set.")

    parser.add_argument(
        "--img_stride",
        type=int,
        default=10,
        help="The stride used to get image patches.")

    parser.add_argument(
        "--cv_fold",
        type=int,
        default=6,
        help="The number of folds of cross-validation.")

    parser.add_argument(
        "--norm_data",
        type=int,
        default=1,
        help="Whether to use normalized data set which the scale bar are the same.")

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005, # Most of time, we use 0.0005
        help="The weight penalizaton parameter.")

    parser.add_argument(
        "--start_fold_idx",
        type=int,
        default=0,
        help="The starting fold idx.")

    parser.add_argument(
        "--end_fold_idx",
        type=int,
        default=10,
        help="The end fold idx.")

    parser.add_argument(
        "--modified_pre_trained_path",
        type=str,
        # Original modified model
        # default="/projects/p30309/logdir/pre_trained_model/DeepLab3PlusMultiBackboneModified_xception_rd_0_epoch_incr_entr_ft_1_full_mod_no_scal_pp1_base_chosen_extended_chosen_0__incr_entr_ft_1_full_mod_no_scal_pp1_20200419-053645.ckpt",
        # Modified model
        # default="/projects/p30309/logdir/pre_trained_model/DeepLab3PlusModified_rd_0_epoch_incr_modified_entr_ft_1_full_mod_no_scal_pp1_base_chosen_extended_chosen_0__incr_modified_full_entr_ft_1_full_mod_no_scal_pp1_20200428-035157.ckpt",
        # Full model
        default="/projects/p30309/logdir/pre_trained_model/DeepLab3Plus_rd_0_epoch_incr_full_entr_ft_1_full_mod_no_scal_pp1_base_chosen_extended_chosen_0__incr_full_entr_ft_1_full_mod_no_scal_pp1_20200428-035211.ckpt",
        # default="",
        help="The pre-trained weights for modified deeplabv3+ network.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
