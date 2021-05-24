# An example of showing concept drift methods on real 2d regression model.
# %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg') # Needed for running on quest
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import sys
import os
import time
import datetime as dt
from PIL import Image

from control_chart.utils import *
from control_chart.data_generation import *
from regression.regressors import *
from control_chart.hotelling import *

# %%
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s(%(funcName)s)[%(lineno)d]: %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'zkg'}
logger = logging.getLogger('simulation_real_img_reg_retro')
logging.getLogger('simulation_real_img_reg_retro').setLevel(logging.INFO)

# This function reads real image and do the unsupervised learning to find different materials.

def simulation_real_img_reg_retro(FLAGS):
    """ Real 2d regression model simulation. """
    tf.random.set_seed(FLAGS.rand_seed)
    np.random.seed(seed=FLAGS.rand_seed)

    # Read real image
    # In case the input image is not in .csv format.
    
    FLAGS.real_img_abs_path = os.path.join(FLAGS.res_root_dir, FLAGS.real_img_path)

    print(FLAGS.real_img_abs_path)
    if not FLAGS.real_img_abs_path.endswith('.csv'):
        csv_file_path = FLAGS.real_img_abs_path[:-4]+'.csv'
        if not os.path.exists(csv_file_path):
            img = Image.open(FLAGS.real_img_abs_path).convert('L')
            img_arr = np.array(img)[:,:]
            img_arr -= np.min(img_arr)
            if np.max(img_arr)>255:
                img_arr = img_arr/np.max(img_arr)*255
            img_arr = img_arr.astype(np.uint8)
            np.savetxt(csv_file_path, img_arr, fmt='%d', delimiter=',')
        FLAGS.real_img_abs_path = csv_file_path
    
    img_arr = np.genfromtxt(FLAGS.real_img_abs_path, delimiter=',')
    # We need to normalize the image for the purpose of computation.
    img_arr = (img_arr.astype(np.float32)-np.mean(img_arr))/np.std(img_arr)

    # # Normalize the image array.
    # img_arr = np.array(img)
    print("Std before scaling: {}".format(np.std(img_arr)))
    # img_arr -= np.min(img_arr)
    # img_arr = img_arr/np.max(img_arr)
    # print("Std after scaling: {}".format(np.std(img_arr)))

    FLAGS.img_hei, FLAGS.img_wid = img_arr.shape
    print(img_arr.shape)
    
    # Training data

    def Gen_Plot_Save_Real_2D_Reg_Data(img_arr, fig_name, title, FLAGS):
        X, y, moni_stat_hei, moni_stat_wid = Generate_Materials_Data(img_arr, FLAGS)
        N = X.shape[0]
        PlotSaveSpatialHeatMap(img_arr, os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), fig_name=fig_name, cmap=GRAY_CMAP, title=title)
        print(img_arr.shape)
        # np.savetxt(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), fig_name.split(
        #     '.')[0]+'.csv'), y.reshape(moni_stat_hei, moni_stat_wid), delimiter=",")
        return X, y, N, moni_stat_hei, moni_stat_wid

    # Generate data for training supervised learning model
    fig_name = 'sim_real_reg_2d_data.png'
    title = 'y'
    X, y, N, moni_stat_hei, moni_stat_wid = Gen_Plot_Save_Real_2D_Reg_Data(img_arr, fig_name, title, FLAGS)
    N_PIIs = [moni_stat_hei*moni_stat_wid]

    print("The monitoring statistics figure shape is ({}, {}).\n".format(moni_stat_hei, moni_stat_wid))
    
    # Preparation for parameters
    alarm_level = FLAGS.alarm_level
    FLAGS.training_res_folder = os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder)
    eff_wind_len_factor = FLAGS.eff_wind_len_factor
    rcond_num = RCOND_NUM

    if not os.path.exists(FLAGS.training_res_folder):
        os.makedirs(FLAGS.training_res_folder)

    train_PI_flag = True

    model_time_stamp = dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

    if FLAGS.nnet:
        penal = "l2"
        hidden_layer_sizes = [10]
        stopping_lag = FLAGS.stopping_lag
        training_batch_size = FLAGS.training_batch_size

        if FLAGS.activation == 'sigmoid':
            # penal_param = 0.001
            # learning_rate = 0.01
            penal_param = FLAGS.penal_param
            learning_rate = FLAGS.learning_rate
            print(penal_param, stopping_lag, training_batch_size, learning_rate)
        elif FLAGS.activation == 'relu':
            penal_param = 0.0001
            learning_rate = 0.01

        # Don't have trained parameters for neural network
        # if not os.path.isfile(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), '_'.join(['nnet_real_reg', str(penal_param).replace('.', '_'), 'score_PII.csv']))):
        if True:
            if not os.path.exists(FLAGS.training_res_folder):
                os.makedirs(FLAGS.training_res_folder)

            if train_PI_flag and FLAGS.cv_flag:
                cv_tasks_info = Gen_CV_Info(
                    penal_param, stopping_lag, training_batch_size, learning_rate, FLAGS)
            else:
                cv_tasks_info = None

            # (model, model_ckpt_fname, mu_train, Sinv_train, _, _, _, _, _, _, _, _, 
            #  grads_PI, grads_PII,
            #  fisher_info_mat_train, _, 
            #  cum_abs_resi_PI, cum_abs_resi_PII,
            #  _, _, abs_resi_PI, abs_resi_PII, 
            #  _, _, dev_PI, dev_PII, best_cv_param) = Neural_Network_Reg(
            #     # X_train, y_train, X_val, y_val, X_PI, y_PI, X_PII, y_PII,
            #     X, y, X, y, X, y, X, y,
            #     N_PIIs, 0, hidden_layer_sizes, penal, penal_param, stopping_lag,
            #     training_batch_size, learning_rate, train_PI_flag, model_time_stamp, FLAGS,
            #     cv_tasks_info=cv_tasks_info, plot_trace_flag=True)

            reg_model = Nnet_Reg(X, y, X, y, hidden_layer_sizes, penal, penal_param, stopping_lag, training_batch_size,
                                learning_rate, train_PI_flag, model_time_stamp, FLAGS, 
                                normal_flag=False, cv_tasks_info=cv_tasks_info, plot_trace_flag=True) # normal_flag=False: Don't normalize within this function.
            _, _, grads_PII, _, _, _ = reg_model.cal_metrics(X, y, data_info='train') # Training and PII images are the same
            FLAGS = reg_model.FLAGS

            # score_PI = np.array((-1)*grads_PI)
            score_PII = np.array((-1)*grads_PII)

            # np.savetxt(
            #     os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), '_'.join(
            #         ['nnet_real_reg', str(penal_param).replace('.', '_'), 'score_PI.csv'])),
            #     score_PI, fmt='%.5e', delimiter=',')
            PII_score_path = os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), '_'.join(['nnet_real_reg', str(penal_param).replace('.', '_'), 'score_PII.csv']))
            np.savetxt(PII_score_path, score_PII, fmt='%.5e', delimiter=',')
            logger.info("The score data has been stored at {}.".format(PII_score_path))
        else:
            FLAGS = pickle.load(open(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'sim_flags.h5'), 'rb'))
            # score_PI = np.genfromtxt(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), '_'.join(
            #     ['nnet_real_reg', str(penal_param).replace('.', '_'), 'score_PI.csv'])), delimiter=',')
            score_PII = np.genfromtxt(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), '_'.join(
                ['nnet_real_reg', str(penal_param).replace('.', '_'), 'score_PII.csv'])), delimiter=',')
    else:  # Linear regression
        penal_param = 10**(-8)

        # if not os.path.exists(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), '_'.join(['lin_reg', str(penal_param).replace('.', '_'), 'score_PII.csv']))):
        if True:
            # (model, mu_train, Sinv_train, _, _, _, _, _, _, grads_PI, grads_PII,
            #  fisher_info_mat_train, _, cum_abs_resi_PI, cum_abs_resi_PII,
            #  _, abs_resi_PI, abs_resi_PII, _, dev_PI, dev_PII) = Linear_Model_Cum_ewma_resi_ewma_dev(
            #     # np.vstack((X_train, X_val)), np.hstack((y_train, y_val)),
            #     np.vstack(X), np.hstack(y), X, y, X, y, N_PIIs, 0, 0, 
            #     penal_param, train_PI_flag, model_time_stamp, FLAGS)

            reg_model = Linear_Reg(np.vstack(X), np.hstack(y), penal_param, train_PI_flag, model_time_stamp, FLAGS)
            _, _, grads_PII, _, _, _ = reg_model.cal_metrics(np.vstack(X), np.hstack(y), data_info='train')
            FLAGS = reg_model.FLAGS

            # score_PI = np.array((-1)*grads_PI)
            score_PII = np.array((-1)*grads_PII)

            inv_fisher_info_mat_train = Inv_Mat_Rcond(
                FLAGS.fisher_info_mat_train , FLAGS.nugget)

            S_train = Inv_Mat_Rcond(FLAGS.Sinv_train, FLAGS.nugget)

            # scaled_score_PI = - np.matmul(grads_PI, inv_fisher_info_mat_train)
            scaled_score_PII = - np.matmul(grads_PII, inv_fisher_info_mat_train)
            scaled_mu_train = np.matmul(FLAGS.mu_train, inv_fisher_info_mat_train)
            scaled_S_train = np.matmul(np.matmul(inv_fisher_info_mat_train, S_train), inv_fisher_info_mat_train)
            scaled_Sinv_train = np.linalg.inv(scaled_S_train)

            # np.savetxt(
            #     os.path.join(FLAGS.training_res_folder, '_'.join(
            #         ['lin_real_reg', str(penal_param).replace('.', '_'), 'score_PI.csv'])),
            #     score_PI, fmt='%.5e', delimiter=',')
            PII_score_path = os.path.join(FLAGS.training_res_folder, '_'.join(['lin_real_reg', str(penal_param).replace('.', '_'), 'score_PII.csv']))
            np.savetxt(PII_score_path, score_PII, fmt='%.5e', delimiter=',')
            logger.info("The score data has been stored at {}.".format(PII_score_path))
        else:
            FLAGS = pickle.load(open(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'sim_flags.h5'), 'rb'))
            # score_PI = np.genfromtxt(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), '_'.join(
            #     ['lin_real_reg', str(penal_param).replace('.', '_'), 'score_PI.csv'])), delimiter=',')
            score_PII = np.genfromtxt(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), '_'.join(
                ['lin_real_reg', str(penal_param).replace('.', '_'), 'score_PII.csv'])), delimiter=',')

    
    # (FLAGS.mu_train, FLAGS.Sinv_train, FLAGS.moni_stat_hei, FLAGS.moni_stat_wid) = (
    #     mu_train, Sinv_train, moni_stat_hei, moni_stat_wid)

    (FLAGS.moni_stat_hei, FLAGS.moni_stat_wid) = (moni_stat_hei, moni_stat_wid)

    # Save FLAGS
    flags_archive = os.path.join(FLAGS.training_res_folder, "retro_sim_flags.h5")
    # if not os.path.isfile(flags_archive):
    print(FLAGS)
    pickle.dump(FLAGS, open(flags_archive, 'wb'))
    logger.info("The FLAGS has been stored at {}.".format(flags_archive))

    # Retrospective analysis:
    fig_name = '_'.join([str(FLAGS.rand_seed), 'sim_real_reg_2d_score_retro', str(
        penal_param).replace('.', '_')])+'.png'
    
    # Setting save_sep to True would mess up the tight_layout().
    SpatialHotellingT2Retro(score_PII, moni_stat_hei, moni_stat_wid, FLAGS.spatial_ewma_sigma, FLAGS.spatial_ewma_wind_len, fig_name, FLAGS, save_sep=True)

    # Increment the number of finished job by 1.
    if not os.path.exists('./command_script/num_job_finished.txt'):
        os.popen('echo 0, 0 > ./command_script/num_job_finished.txt')
    num_finished_jobs = np.genfromtxt('./command_script/num_job_finished.txt', delimiter=',')
    num_finished_jobs[0] += 1
    np.savetxt('./command_script/num_job_finished.txt', num_finished_jobs, delimiter=',', fmt='%d')

