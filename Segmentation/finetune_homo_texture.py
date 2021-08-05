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

from train_one_round import *
from seg_utils.generate_collages import *
from seg_utils.datagenerator import *
from seg_utils.utils import *
from Unet.unet import *
from DeepLab3.deeplab3 import *
from DeepLab3.deeplabv3plus_mobilenetv2_layers import *
from DeepLab3.deeplabv3plus_xception_layers import *


# %%
# To run on K80 GPU, with 12G memory, the float needs tf.float16 and batch size needs to be 4.

def main(_):
    # The random seed for np and tf are independent. In order to reproduce results, I need to set both seeds.
    np.random.seed(RND_SEED)
    tf.random.set_seed(RND_SEED)
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

    root_dir = ROOT_DIR # FLAGS.root_dir
    # plt_fd_name = '5_texture_images_5v2_5m'
    plt_fd_name = 'cla_incr_images'
    postfix = FLAGS.postfix
    plt_dir = os.path.join(os.path.expanduser(os.path.join(PLT_ROOT_DIR, '20210701_homo_seg_res/figures/')), plt_fd_name+postfix)

    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)

    base_dir = os.path.expanduser(os.path.join(root_dir, 'Data/texture/Brodatz_seg_images/{}/'.format(plt_fd_name)))

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
    FLAGS.batch_size = BATCH_SIZE
    FLAGS.val_batch_ratio = VAL_BATCH_RATIO
    # weight_decay= 0.0005 # Caffe style regularization parameter
    num_classes = len(ls_fnames)
    # input_shapes = (256, 256, 3)
    input_shapes = (IMG_SIZE, IMG_SIZE, 3)
    new_size = IMG_SIZE
    (keep_prob, dropout_rate, train_layers, last_layer_name, 
     pretrained_wei_path, kwargs, backbone, alpha, output_stride,
     weights_name) = config_deeplab(num_classes, input_shapes, FLAGS)

    
    # # train_layers = ['fc8', 'fc7', 'fc6'] # Only train last few layers.

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
    # line_colors = [xkcd_color_names[ci] for ci in np.random.choice(range(num_colors), size=len(ls_fnames+ls_ext_fnames)+1, replace=False)]
    # For paper plots
    line_colors = ['xkcd:tree green', 'xkcd:dusky blue', 'xkcd:grey', 'xkcd:orangish brown', 'xkcd:midnight purple', 'xkcd:iris', 'xkcd:merlot', 'xkcd:eggshell blue', 'xkcd:deep orange']
    print("The line_colors are: {}.".format(line_colors))
    
    rand_line_colors = [co for co in line_colors]
    # np.random.shuffle(rand_line_colors)
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

    """
    Training, Validating, Testing
    """

    if FLAGS.start_rd > 0:
        print("Reload previous model weights.")
        rd_num_classes = len(ls_fnames)+FLAGS.start_rd-1
        # if FLAGS.model_name!='DeepLab3Plus' and FLAGS.model_name!='Unet': 
        #     rd_kwargs = copy.deepcopy(kwargs)
        #     rd_kwargs.update({'num_classes': rd_num_classes})
        # else:
        #     rd_kwargs = None
        if FLAGS.model_name!='Unet': 
            rd_kwargs = copy.deepcopy(kwargs)
            rd_kwargs.update({'num_classes': rd_num_classes})
        if os.path.isfile(FLAGS.start_model_path):
            loaded_model = construct_model(FLAGS, keep_prob, rd_num_classes, input_shapes, pretrained_wei_path, rd_kwargs, skip_layer=[])
            try:
                loaded_model.load_weights(FLAGS.start_model_path)
                model_path = FLAGS.start_model_path
                if FLAGS.model_name=='Unet' or FLAGS.model_name=='DeepLab3Plus' or FLAGS.model_name=='DeepLab3PlusModified':
                    print("The weights before loading: \n{}.".format(loaded_model.model.get_layer(name=last_layer_name).trainable_variables))
                else:
                    print("The weights before loading: \n{}.".format(loaded_model.model.model.get_layer(name=last_layer_name).trainable_variables))
            except (FileNotFoundError, ValueError) as e:
                loaded_model = None
                model_path = ''
                print(e)
        else:
            print("No previous model weigths: {}.".format(FLAGS.start_model_path))
            loaded_model = None
            model_path = ''
            
        # Generate images in new file for this incremental learning step
        log_fd_name = '_'.join([FLAGS.gen_fd_prefix, db_fd_name, new_fd_name, str(FLAGS.start_rd-1)])
        rd_base_dir = os.path.join(base_dir, log_fd_name)
        print(rd_base_dir)
    else:
        loaded_model = None
        model_path = ''
        
    # for rd_idx in range(FLAGS.start_rd, len(os.listdir(new_base_dir))+1):
    for rd_idx in range(FLAGS.start_rd, len(ls_ext_fnames)+1):
        # Path to the textfiles for the trainings and validation set
        # tr_file = '/path/to/train.txt'
        # val_file = '/path/to/val.txt'
        # log_fd_name = '5_texture_images_5c'
        
        print("==========Iteration {} starts!============".format(rd_idx))
        rd_start_time = time.time()

        # Generate images in new file for this incremental learning step
        log_fd_name = '_'.join([FLAGS.gen_fd_prefix, db_fd_name, new_fd_name, str(rd_idx)])
        # root_dir = '~/scratch'
        rd_base_dir = os.path.join(base_dir, log_fd_name)

        # Generate datasets
        if not os.path.exists(rd_base_dir):
            os.makedirs(rd_base_dir)
        rd_ls_fnames = [fn for fn in ls_fnames]
        os.popen('cp {} {}'.format(os.path.join(db_base_dir, '*'+FLAGS.img_ext), rd_base_dir))
        if rd_idx > 0:
            for t_j in range(rd_idx):
                # new_texture_fname = new_texture_tmpl.replace('*', str(t_j))
                os.popen('cp {} {}'.format(os.path.join(new_base_dir, ls_ext_fnames[t_j]), rd_base_dir))
                rd_ls_fnames.append(ls_ext_fnames[t_j])
        
        time.sleep(10)

        # Make sure all files has been copied.
        # When the number of images to be copied is large, the time waited is not long enough before all images required are copied.
        while True:
            cnrd_idxmg_files = 0
            for fn in os.listdir(rd_base_dir):
                if fn.endswith(FLAGS.img_ext):
                    cnrd_idxmg_files+=1
            if cnrd_idxmg_files == len(rd_ls_fnames):
                print("All image files {} has been copied.".format(rd_ls_fnames))
                break
            else:
                time.sleep(10)
            
        # Must past in the list of filenames to keep the order and label of textures.
        print(rd_base_dir, rd_idx, rd_ls_fnames)
        if FLAGS.gen_data_flag:
            train_size, valid_size, test_size = FLAGS.train_size, FLAGS.valid_size, FLAGS.test_size
            with tf.device('/cpu:0'):
                gen_save_train_valid_test_dataset(rd_base_dir, train_size, valid_size, test_size, ls_fnames=rd_ls_fnames, 
                                                new_size=IMG_SIZE, pwei_flag=FLAGS.pwei_flag, normp=FLAGS.normp,
                                                trfm_flag=FLAGS.trfm_flag, num_gen_batch=FLAGS.num_gen_batch, nb=FLAGS.nb, max_rots=FLAGS.max_rots)

        # Generate data pipline
        rd_num_classes = num_classes+rd_idx
        if FLAGS.model_name!='DeepLab3Plus' and FLAGS.model_name!='Unet': 
            rd_kwargs = copy.deepcopy(kwargs)
            rd_kwargs.update({'num_classes': rd_num_classes})
        print("The number of classes at step {} is {}.".format(rd_idx, rd_num_classes))

        tr_file = os.path.join(rd_base_dir, 'train.txt')
        val_file = os.path.join(rd_base_dir, 'valid.txt')
        te_file = os.path.join(rd_base_dir, 'test.txt')

        # Place data loading and preprocessing on the cpu
        with tf.device('/cpu:0'):
            tr_data = HomoTextureDataGenerator(tr_file,
                                    mode='training',
                                    batch_size=FLAGS.batch_size,
                                    num_classes=rd_num_classes,
                                    img_norm_flag=FLAGS.img_norm_flag,
                                    trfm_flag=FLAGS.trfm_flag)
            val_data = HomoTextureDataGenerator(val_file,
                                    mode='inference',
                                    batch_size=FLAGS.batch_size,
                                    num_classes=rd_num_classes,
                                    img_norm_flag=FLAGS.img_norm_flag,
                                    trfm_flag=FLAGS.trfm_flag)
            te_data = HomoTextureDataGenerator(te_file,
                                    mode='inference',
                                    batch_size=FLAGS.batch_size,
                                    num_classes=rd_num_classes,
                                    img_norm_flag=FLAGS.img_norm_flag,
                                    trfm_flag=FLAGS.trfm_flag)

        all_textures = generate_texture(rd_base_dir, ls_fnames=rd_ls_fnames, new_size=IMG_SIZE)

        fig = plt.figure(figsize=(rd_num_classes*5, 8), facecolor='w')

        for i, texture in enumerate(all_textures):
            fig.add_subplot(1,rd_num_classes, i+1)
            plt.imshow(texture.astype(np.uint8), cmap='gray')
            plt.title("Class label: {}.".format(i), size=20)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plt_dir, 'rd_{}_all_textures_true_labs.png'.format(rd_idx)))
        plt.close()

        """
        Main Part of the finetuning Script.
        """

        ls_train_cla_tp, ls_valid_cla_tp = [], []
        for lab in range(rd_num_classes): 
            ls_train_cla_tp.append(tf.keras.metrics.Mean('train_cla_{}_tp'.format(lab), dtype=DTYPE_FLOAT))
            ls_valid_cla_tp.append(tf.keras.metrics.Mean('valid_cla_{}_tp'.format(lab), dtype=DTYPE_FLOAT))

        loaded_model, val_acc, val_cla_acc = train_one_round(rd_idx, tr_data, val_data, te_data, 
                                        log_fd_name, num_classes,
                                        checkpoint_path, keep_prob, input_shapes,
                                        pretrained_wei_path, loaded_model,
                                        train_layers, last_layer_name,
                                        filewriter_path, optimizer,
                                        train_loss_object, train_loss, train_regu, train_obj_val, train_accuracy, ls_train_cla_tp,
                                        valid_loss_object, valid_loss, valid_accuracy, ls_valid_cla_tp,
                                        rand_line_colors, plt_dir,
                                        FLAGS, kwargs, bd_img_flag=True)

        # loaded_model = None

        # Remove previous generated files
        os.popen('rm -rf {}'.format(rd_base_dir))
        time.sleep(30)

        print("--------------Iteration {} with {} classes------------------".format(rd_idx, rd_num_classes))
        print("Validation accuracy:")
        print(val_acc)
        print("Validation class accuracy:")
        print(val_cla_acc)

        print("The total time for round {} with {} classes is {}h.\n".format(rd_idx, rd_num_classes, (time.time()-rd_start_time)/3600))

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
        "--last_idx_valids",
        type=int,
        default=12,
        help="Number of validation rounds.")

    parser.add_argument(
        "--accu_step",
        type=int,
        default=0,
        help="The number of steps has been accumulated before.")

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
        default=0,
        help="Number of epochs for fine tuning.")

    # parser.add_argument(
    #     "--decodor",
    #     type=int,
    #     default=0,
    #     help="Whether to fine tune decoder.")

    parser.add_argument(
        "--last_layer",
        type=int,
        default=0,
        help="Whether to fine tune the last layer.")

    parser.add_argument(
        "--last_layer_mask",
        type=int,
        default=0,
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
        default=4,
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
        help="The flag whether add random weight for each anchor points (in calculating distance) in generating texture collages.")

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
        default=0,
        help="Whether to use sample weight to combat class imbalance.")

    parser.add_argument(
        "--gen_data_flag",
        type=int,
        default=1,
        help="Whether to generate data set.")

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005, # Most of time, we use 0.0005
        help="The weight penalizaton parameter.")

    parser.add_argument(
        "--modified_pre_trained_path",
        type=str,
        # default="/projects/p30309/logdir/pre_trained_model/DeepLab3PlusMultiBackboneModified_xception_rd_0_epoch_incr_entr_ft_1_full_mod_no_scal_pp1_base_chosen_extended_chosen_0__incr_entr_ft_1_full_mod_no_scal_pp1_20200419-053645.ckpt",
        default="",
        help="The pre-trained weights for modified deeplabv3+ network.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)