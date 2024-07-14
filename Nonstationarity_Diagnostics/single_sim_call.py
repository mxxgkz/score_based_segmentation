import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import argparse
import sys
import os
import time
import datetime as dt
import pickle
from joblib import Parallel, delayed, parallel_backend

from constants import DATA_ROOT_DIR, RES_ROOT_DIR, CODE_ROOT_DIR
from control_chart.utils import *
from control_chart.hotelling import *

from simulation_real_img_reg_retro import simulation_real_img_reg_retro


# Without putting this, the loss_val_ls[-1] is a tf.Tensor and cannot be evaluated at that place.
# # tf.enable_eager_execution()

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s(%(funcName)s)[%(lineno)d]: %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'zkg'}
logger = logging.getLogger('single_simu_call')
logging.getLogger('single_simu_call').setLevel(logging.INFO)

# Basic example of linear regression with abrupt concept drift and with multi-colinearity


def main(_):
    if FLAGS.single_exp_plot == 3131:
        simulation_real_img_reg_retro(FLAGS)

    # multi_logis_cla_find_gamma((0, 1, FLAGS))

    flags_archive_path = os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), "sim_flags.h5")
    pickle.dump(FLAGS, open(flags_archive_path, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50000,
        help="Number of steps to run trainer.")
    parser.add_argument(
        "--training_batch_size",
        type=int,
        default=1000,
        help="Batch size used during training.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001, # 0.01,
        help="Initial learning rate.")
    parser.add_argument(
        "--penal_param",
        type=float,
        default=0.01, #0.001,
        help="The L2 penalization parameter.")
    parser.add_argument(
        "--decay_steps",
        type=int,
        default=1, # Looks like larger than 1 is not good for training.
        help="The number of step for decaying the learning rate.")
    parser.add_argument(
        "--training_rounds",
        type=int,
        default=1,
        help="Each round the learning rate is divided by 2.")
    parser.add_argument(
        "--stopping_lag",
        type=int,
        default=200,
        help="The stopping lag for early stopping.")
    parser.add_argument(
        "--ui_type",
        type=str,
        default="curses",
        help="Command-line user interface type (curses | readline)")
    parser.add_argument(
        "--debug",
        type="bool",
        nargs="?",
        const=True,
        default=False,
        help="Use debugger to track down bad values during training")
    parser.add_argument(
        "--data_file_folder",
        type=str,
        default="",
        help="The path of folder to store input data file.")
    parser.add_argument(
        "--Xy_fname",
        type=str,
        default="",
        help="The input data file name.")
    parser.add_argument(
        "--res_root_dir",
        type=str,
        default=RES_ROOT_DIR,
        help="The path of folder to store model and parameter file.")
    parser.add_argument(
        "--model_file_folder",
        type=str,
        default="./train_model/",
        help="The path of folder to store model and parameter file.")
    parser.add_argument(
        "--best_model",
        type=str,
        default="best_model.h5",
        help="The parameter values of the current best model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="concept_drift_model",
        help="The model name.")
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=3,
        help="The random seed.")
    parser.add_argument(
        "--nnet",
        type=int,
        default=1,
        help="Whether use neural network model.")
    parser.add_argument(
        "--clf_thr",
        type=float,
        default=0.5, # 0.1435,  # 0.5,  # 0.1435,
        help="Classification threshold based on cost or other metric.")
    parser.add_argument(
        "--training_res_folder",
        type=str,
        default="",
        help="Some result to oberve during training.")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Add flag to allow to change parameters
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.005,
        help="Whether use neural network model.")

    parser.add_argument(
        "--nugget",
        type=float,
        default=0.000001,
        help="The nugget parameter added to the Fisher Information Matrix to make is not singular.")

    parser.add_argument(
        "--corr_pred",
        type=int,
        default=0,
        help="Whether use neural network model.")

    # Add alarm-level for Phase-I when we set control limits
    parser.add_argument(
        "--alarm_level",
        type=float,
        default=99.9,
        help="False positive level (alarm level) for Phase-I.")

    # Adjust the effective window length to rule out the initial effect of EWMA
    # in calculating Phase-I data and setting control limits
    parser.add_argument(
        "--eff_wind_len_factor",
        type=float,
        default=2.0,
        help="Multiplier of effective window length (times 1/ewma_parameter).")

    # Add just the size of Phase-I and Phase-II(assume they are the same)
    parser.add_argument(
        "--PII_len",
        type=int,
        default=20000,
        help="Length of the Phase-I and Phase-II.")

    parser.add_argument(
        "--model_idx",
        type=int,
        default=0,
        help="Whether use logistic(0),  multinomial(1), linear(2), poisson(3), autoregressive(4), autoregressive 2d(5).")

    parser.add_argument(
        "--num_class",
        type=int,
        default=2,
        help="For classification problem.")

    # Cross-validation
    parser.add_argument(
        "--cv_flag",
        type=int,
        default=0,
        help="Whether to cross-validate models using training and validation datasets.")

    parser.add_argument(
        "--cv_N_rep",
        type=int,
        default=3,
        help="The number of replication of cross-validation.")

    parser.add_argument(
        "--cv_K_fold",
        type=int,
        default=5,
        help="The number of folds of cross-validation.")

    parser.add_argument(
        "--cv_param_ls",
        type=str,
        default="penal_param-stopping_lag-training_batch_size-learning_rate",
        help="The list of names of parameters in cv.")

    parser.add_argument(
        "--cv_task_param_ls",
        type=str,
        default="penal_param-stopping_lag-training_batch_size-learning_rate",
        help="The list of names of parameters that actually vary in cv.")

    parser.add_argument(
        "--cv_rand_search",
        type=int,
        default=0,
        help="The number of combinations of parameters of randomized search in cross-validation.")

    parser.add_argument(
        "--cv_n_jobs",
        type=int,
        default=24,
        help="The number of cpus to run cross-validation.")

    parser.add_argument(
        "--cv_pre_dispatch",
        type=str,
        default="0.15*n_jobs",
        help="The number of pre-dispatch jobs for cross-validation.")

    # Different models using different loss function.
    parser.add_argument(
        "--reg_model",
        type=str,
        default="",
        help="The name for regression model."
             "lin: linear regression"
             "logi: logistic regression"
             "tree: decision tree"
             "pois: poisson regression"
             "multi-logis: multi-nomial regression")

    parser.add_argument(
        "--activation",
        type=str,
        default="sigmoid",
        help="The activation in hidden layers.")
    parser.add_argument(
        "--output_acti",
        type=int,
        default=0,
        help="Whether add activation at output layer.")

    parser.add_argument(
        "--N_rep_find_gamma",
        type=int,
        default=1,
        help="Number of repetition to find gamma.")

    parser.add_argument(
        "--single_exp_plot",
        type=int,
        default=1,
        help="The index of single examples for plotting.")

    # ---------------------------------------------------------------------------
    # For spatial ewma Hotelling T2
    parser.add_argument(
        "--materials_model",
        type=str,
        default="causal",
        help="Whether the model (use neighbors to predict a pixel) in materials is causal or non-causal.")

    parser.add_argument(
        "--wind_hei",
        type=int,
        default=20,
        help="The window used in the prediction model: height, better to be odd number.")

    parser.add_argument(
        "--wind_wid",
        type=int,
        default=20,
        help="The window used in the prediction model: width, better to be odd number.")

    parser.add_argument(
        "--nois_sigma",
        type=float,
        default=1.0,
        help="The noise standard deviation of data generation process.")

    parser.add_argument(
        "--intcp",
        type=float,
        default=0.1,
        help="The intercept in the model of data generation process.")

    parser.add_argument(
        "--z_scale",
        type=float,
        default=1.0,
        help="The scale of latent varaible when generating autoregressive process (mainly for classification).")

    parser.add_argument(
        "--spatial_ewma_sigma",
        type=float,
        default=3,
        help="The bandwidth parameter for weight function of 2D spatial ewma.")

    parser.add_argument(
        "--spatial_ewma_wind_len",
        type=int,
        default=30,
        help="The half of window length of the spatial ewma window. The real window length is 2*wind_len-1.")

    parser.add_argument(
        "--img_file_name",
        type=str,
        default="",
        help="The file of simulated materials images.")

    parser.add_argument(
        "--PI_img_file_name",
        type=str,
        default="",
        help="The Phase-I file of simulated materials images.")

    parser.add_argument(
        "--PII_img_file_name",
        type=str,
        default="",
        help="The Phase-II file of simulated materials images.")

    # ---------------------------------------------------------------------------
    # For generating samples using trained model.
    parser.add_argument(
        "--regen_grid_size",
        type=int,
        default=200,
        help="The size of image of artificially generated samples.")

    # ---------------------------------------------------------------------------
    # For autoregressive model. Read the coefficient file.
    parser.add_argument(
        "--ar_model_coef_folder_path",
        type=str,
        default='../Data/ar_model/',
        help="The folder path to read autoregressive coefficients from. Those coefficients are stable.")

    parser.add_argument(
        "--img_hei",
        type=int,
        default=300,
        help="The height of image for autoregressive 2D image, better to be even number.")

    parser.add_argument(
        "--img_wid",
        type=int,
        default=300,
        help="The width of image for autoregressive 2D image, better to be even number.")

    parser.add_argument(
        "--gen_wind_hei",
        type=int,
        default=4,
        help="The window height for generating autoregressive image, better to be odd number.")

    parser.add_argument(
        "--gen_wind_wid",
        type=int,
        default=4,
        help="The window width for generating autoregressive image, better to be odd number.")

    # ---------------------------------------------------------------------------
    # For running CNN on cpu.
    parser.add_argument(
        "--base_dir",
        type=str,
        default="",
        help="The base directory for inputting and outputting results.")

    parser.add_argument(
        "--exp_folder",
        type=str,
        default="",
        help="The subfolder to store experiment results.")

    parser.add_argument(
        "--model_prefix",
        type=str,
        default="",
        help="The prefix of saved model files.")

    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="The postfix for file folder name.")

    parser.add_argument(
        "--cnn_model",
        type=str,
        default="resnet18",
        help="The name of CNN model.")

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device for running the algoirthm, cpu or cuda.")

    parser.add_argument(
        "--fine_tuning_epochs",
        type=str,
        default="",
        help="The name of epoches for different iteration of training (4-4-6-20).")

    parser.add_argument(
        "--load_model_name",
        type=str,
        default="",
        help="The model name to load.")

    parser.add_argument(
        "--epochs_before_fine_tuning",
        type=int,
        default=4,
        help="The number of  of CNN model.")

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-4,
        help="The minimum learning rate for fine tuning.")

    parser.add_argument(
        "--max_lr",
        type=float,
        default=1e-3,
        help="The maximum learning rate for fine tuning.")

    parser.add_argument(
        "--gen_img_pat_flag",
        type=int,
        default=1,
        help="Whether to generate image patches.")

    # ---------------------------------------------------------------------------
    # For generating noisy data from two existing materials samples.
    parser.add_argument(
        "--nois_profile_sigma",
        type=float,
        default=2,
        help="The sigma for superimpose two existing materials samples.")

    parser.add_argument(
        "--nois_size",
        type=int,
        default=100,
        help="The number of noise points of stochastic materials.")

    parser.add_argument(
        "--nois_scale",
        type=float,
        default=1,
        help="The scaler of noise profile (Gaussian) distribution.")

    # ---------------------------------------------------------------------------
    # For clustering scores in 3D.
    parser.add_argument(
        "--km_max_n_clusters",
        type=int,
        default=10,
        help="The maximum number of clusters to try using kmeans.")
    
    parser.add_argument(
        "--loc_coord_wei",
        type=float,
        default=1e-6, # 0.001,
        help="The weight of 2D spatial coordinates when clustering scores in 3D space.")

    parser.add_argument(
        "--n_comp",
        type=int,
        default=4,
        help="The number of components used in k-means clustering.")

    # ---------------------------------------------------------------------------
    # For clustering scores for real images.
    parser.add_argument(
        "--norm_input_data",
        type=int,
        default=0,
        help="Whether we normalize the input data, i.e. image or np.ndarray.") 

    parser.add_argument(
        "--real_img_path",
        type=str,
        # default='/home/ghhgkz/scratch/Data/texture/Brodatz/Nat-5c.pgm',
        help="The path of real image to be processed.")

    parser.add_argument(
        "--real_img_folder",
        type=str,
        default='/projects/p30309/CD/Data/texture/GP/prospective/data1/',
        help="The path of real images to be processed for PI_PII analysis.")

    # ========================================
    # Gaussian Process generated random field to test different monitoring metrics.
    # One set of data
    parser.add_argument(
        "--train_img",
        type=str,
        default='./gp_obse_noise_5_lscal_5_0_5_5/one_gp_obse_micro_struct_noise_5_lscal_5_0_5_5.png',
        help="The path of real images to be processed for PI_PII analysis (train).") 

    parser.add_argument(
        "--val_img",
        type=str,
        default='./gp_obse_noise_5_lscal_5_1_5_5/one_gp_obse_micro_struct_noise_5_lscal_5_1_5_5.png',
        help="The path of real images to be processed for PI_PII analysis (val).")

    parser.add_argument(
        "--PI_img",
        type=str,
        default='./gp_obse_noise_5_lscal_5_2_5_5/one_gp_obse_micro_struct_noise_5_lscal_5_2_5_5.png',
        help="The path of real images to be processed for PI_PII analysis (PI).")

    parser.add_argument(
        "--PII_img",
        type=str,
        default='targ_noise_5_lscal_5_noise_5_lscal_10.png',
        help="The path of real images to be processed for PI_PII analysis (PII).")

    parser.add_argument(
        "--PII_img_ls",
        type=str,
        default='',
        help="The list of PII images separated by comma.")

    parser.add_argument(
        "--PII_img_ratio",
        type=float,
        default=1.0,
        help="The ratio of the size of PII images to be taken.")

    # # -----------------
    # # The other set of data
    # parser.add_argument(
    #     "--train_img",
    #     type=str,
    #     default='./gp_obse_noise_5_lscal_10_0_10_10/one_gp_obse_micro_struct_noise_5_lscal_10_0_10_10.png',
    #     help="The path of real images to be processed for PI_PII analysis (train).") 

    # parser.add_argument(
    #     "--val_img",
    #     type=str,
    #     default='./gp_obse_noise_5_lscal_10_1_10_10/one_gp_obse_micro_struct_noise_5_lscal_10_1_10_10.png',
    #     help="The path of real images to be processed for PI_PII analysis (val).")

    # parser.add_argument(
    #     "--PI_img",
    #     type=str,
    #     default='./gp_obse_noise_5_lscal_10_2_10_10/one_gp_obse_micro_struct_noise_5_lscal_10_2_10_10.png',
    #     help="The path of real images to be processed for PI_PII analysis (PI).")

    # parser.add_argument(
    #     "--PII_img",
    #     type=str,
    #     default='targ_noise_5_lscal_10_noise_5_lscal_5.png',
    #     help="The path of real images to be processed for PI_PII analysis (PII).")

    # =======================
    # One PIPII analysis dataset

    # parser.add_argument(
    #     "--train_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_1.csv',
    #     help="The path of real images to be processed for PI_PII analysis (train).") 

    # parser.add_argument(
    #     "--val_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_2.csv',
    #     help="The path of real images to be processed for PI_PII analysis (val).")

    # parser.add_argument(
    #     "--PI_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_3.csv',
    #     help="The path of real images to be processed for PI_PII analysis (PI).")

    # parser.add_argument(
    #     "--PII_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_300_RPM_4000_KJ_KG.csv',
    #     help="The path of real images to be processed for PI_PII analysis (PII).")

    # =======================
    # Another PIPII analysis dataset

    # parser.add_argument(
    #     "--train_img",
    #     type=str,
    #     default='300_RPM_4000_KJ_KG_1.csv',
    #     help="The path of real images to be processed for PI_PII analysis (train).") 

    # parser.add_argument(
    #     "--val_img",
    #     type=str,
    #     default='300_RPM_4000_KJ_KG_2.csv',
    #     help="The path of real images to be processed for PI_PII analysis (val).")

    # parser.add_argument(
    #     "--PI_img",
    #     type=str,
    #     default='300_RPM_4000_KJ_KG_3.csv',
    #     help="The path of real images to be processed for PI_PII analysis (PI).")

    # parser.add_argument(
    #     "--PII_img",
    #     type=str,
    #     default='300_RPM_4000_KJ_KG_100_RPM_100_KJ_KG.csv',
    #     help="The path of real images to be processed for PI_PII analysis (PII).")

    # =================
    # Similar micro-structure, different samples

    # parser.add_argument(
    #     "--train_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_17_1.csv',
    #     help="The path of real images to be processed for PI_PII analysis (train).") 

    # parser.add_argument(
    #     "--val_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_17_2.csv',
    #     help="The path of real images to be processed for PI_PII analysis (val).")

    # parser.add_argument(
    #     "--PI_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_17_3.csv',
    #     help="The path of real images to be processed for PI_PII analysis (PI).")

    # parser.add_argument(
    #     "--PII_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_17_100_RPM_100_KJ_KG_19.csv',
    #     help="The path of real images to be processed for PI_PII analysis (PII).")

    # # .................
    # # For anomaly detection
    # parser.add_argument(
    #     "--train_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_19_1.csv',
    #     help="The path of real images to be processed for PI_PII analysis (train).") 

    # parser.add_argument(
    #     "--val_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_19_2.csv',
    #     help="The path of real images to be processed for PI_PII analysis (val).")

    # parser.add_argument(
    #     "--PI_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_19_3.csv',
    #     help="The path of real images to be processed for PI_PII analysis (PI).")

    # parser.add_argument(
    #     "--PII_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_19_4.csv',
    #     help="The path of real images to be processed for PI_PII analysis (PII).")

    # ----------------------

    # parser.add_argument(
    #     "--train_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_19_1.csv',
    #     help="The path of real images to be processed for PI_PII analysis (train).") 

    # parser.add_argument(
    #     "--val_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_19_2.csv',
    #     help="The path of real images to be processed for PI_PII analysis (val).")

    # parser.add_argument(
    #     "--PI_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_19_3.csv',
    #     help="The path of real images to be processed for PI_PII analysis (PI).")

    # parser.add_argument(
    #     "--PII_img",
    #     type=str,
    #     default='100_RPM_100_KJ_KG_19_100_RPM_100_KJ_KG_17.csv',
    #     help="The path of real images to be processed for PI_PII analysis (PII).")


    parser.add_argument(
        "--only_plot",
        type=int,
        default=ONLY_PLOT,
        help="Only plot don't compute.")
    
    parser.add_argument(
        "--noise_level_shape",
        type=float,
        default=7.5,
        help="The shape of gamma distribution of noise level.")

    parser.add_argument(
        "--noise_level_scale",
        type=float,
        default=2,
        help="The scale of gamma distribution of noise level.")

    parser.add_argument(
        "--noise_level_lambda",
        type=float,
        default=100,
        help="The lambda of gamma distribution of noise level. Negative means that the noise_level should be zero.")

    parser.add_argument(
        "--gen_func_str_train_PI",
        type=str,
        default='max_min_cap_exp:reg',
        help="The generation functions for train_PI.")

    parser.add_argument(
        "--num_train_imgs",
        type=int,
        default=1,
        help="The number of training images for prospective analysis.")

    parser.add_argument(
        "--num_val_imgs",
        type=int,
        default=1,
        help="The number of validating images for prospective analysis.")

    parser.add_argument(
        "--num_PI_imgs",
        type=int,
        default=1,
        help="The number of PI images for prospective analysis.")

    parser.add_argument(
        "--num_PII_imgs",
        type=int,
        default=4,
        help="The number of PII images for prospective analysis.")

    parser.add_argument(
        "--train_PI_coeff",
        type=str,
        default='',
        help="The file name of coefficients for generating 2D AR images from training to PI.")

    parser.add_argument(
        "--PII_coeff",
        type=str,
        default='',
        help="The file name of coefficients for generating 2D AR images in PII.")

    parser.add_argument(
        "--large_scale_sim_num",
        type=int,
        default=-1,
        help="The number of large-scale simulation replicates.")

    parser.add_argument(
        "--sim_block_size",
        type=int,
        default=10,
        help="The number of the block of large-scale simulation replicates.")

    parser.add_argument(
        "--sim_block_idx",
        type=int,
        default=0,
        help="The index of the block of large-scale simulation replicates.")

    parser.add_argument(
        "--large_scale_sim_n_jobs",
        type=int,
        default=2,
        help="The number of jobes for replicates.")

    parser.add_argument(
        "--PII_portion",
        type=float,
        default=1,
        help="The portion of cofficients of PII contributing to the real PII coefficients in generating AR images.")

    parser.add_argument(
        "--gen_img_data",
        type=int,
        default=1,
        help="Whether generate image data or read from files.")

    parser.add_argument(
        "--plot_img_PI_idx_str",
        type=str,
        default='all',
        help="Which index of PI image to plot.")

    parser.add_argument(
        "--plot_img_PII_idx_str",
        type=str,
        default='all',
        help="Which index of PI image to plot.")

    parser.add_argument(
        "--plot_metric_idx_str",
        type=str,
        default='all',
        help="Which index of metrics to plot.")

    parser.add_argument(
        "--multi_chart_scale_flag",
        type=str,
        default='mid',
        help="What kinds of scale scheme used to combine multiple control charts together.")

    # ---------------------
    # Experiments with bike sharing 2010-2020 dataset
    parser.add_argument(
        "--pow",
        type=float,
        default=1,
        help="The power of yearly mean used to normalize the bike rental count.")
    
    parser.add_argument(
        "--bs_dataset",
        type=str,
        default="",
        help="Which data set to run experiments.")

    parser.add_argument(
        "--training_t_len_yr",
        type=float,
        default=2,
        help="The number of years used for training Phase.")

    parser.add_argument(
        "--PI_t_len_yr",
        type=float,
        default=1,
        help="The number of years used for Phase-I Phase.")

    parser.add_argument(
        "--start_yr",
        type=int,
        default=2013,
        help="The starting year of training data.")

    parser.add_argument(
        "--end_yr",
        type=int,
        default=2020,
        help="The ending year of PII data.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
