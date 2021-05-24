import numpy as np
import pickle
import pandas as pd
import logging
import os
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import itertools
import tensorflow as tf
import math
from datetime import datetime

from scipy.sparse.linalg import eigsh, eigs
from tensorflow.keras.constraints import max_norm
from joblib import Parallel, delayed, parallel_backend

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
# from control_chart.hotelling import EwmaT2PI, calEwmaT2StatisticsPI, calEwmaT2StatisticsPII, EwmaPI, calEwmaStatisticsPI, calEwmaStatisticsPII, calEwmaStatisticsHelper
from constants import *

# Without putting this, the loss_val_ls[-1] is a tf.Tensor and cannot be evaluated at that place.
# # tf.enable_eager_execution()

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s(%(funcName)s)[%(lineno)d]: %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'zkg'}
logger = logging.getLogger('utils')
logging.getLogger('utils').setLevel(logging.INFO)

def Set_Axis_Prop(ax, ls_axis_lab, labelsize=LAB_SIZE, labelpad=LAB_PAD, title="", rot=0, as_ratio=1.0, axis_tick_flag=[True]*3):
    ax.set_xlabel(ls_axis_lab[0], size=0.75*labelsize, labelpad=labelpad)
    ax.set_ylabel(ls_axis_lab[1], size=0.75*labelsize, labelpad=labelpad)
    # x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    # print(x_lim, y_lim)
    # if tick_int:
    #     x_tick_num, y_tick_num = TICK_NUM, TICK_NUM
    #     while int(x_lim[1]-x_lim[0])%x_tick_num!=0:
    #         x_tick_num += 1
    #     while int(y_lim[1]-y_lim[0])%y_tick_num!=0:
    #         y_tick_num += 1
    # Set the number of tick labels: https://stackoverflow.com/a/55690467/4307919
    ax.xaxis.set_major_locator(ticker.MaxNLocator(TICK_NUM))
    # ax.xaxis.set_minor_locator(ticker.MaxNLocator(x_tick_num*5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(TICK_NUM))
    # ax.yaxis.set_minor_locator(ticker.MaxNLocator(y_tick_num*5))
    # ax.set_xticks(list(np.arange(*x_lim,(x_lim[1]-x_lim[0])/TICK_NUM)))
    # ax.set_yticks(list(np.arange(*y_lim,(y_lim[1]-y_lim[0])/TICK_NUM)))
    ax.xaxis.set_tick_params(labelsize=0.65*labelsize, rotation=rot)
    # ax.xaxis.set_tick_params(labelsize=0.65*labelsize)
    ax.yaxis.set_tick_params(labelsize=0.65*labelsize, rotation=rot)
    if len(ls_axis_lab)==3:
        ax.set_zlabel(ls_axis_lab[2], size=0.75*labelsize, labelpad=1.5*labelpad)
        # z_lim = ax.get_zlim()
        # print(z_lim, list(np.arange(*z_lim,(z_lim[1]-z_lim[0])/TICK_NUM)))
        # ax.set_zticks(list(np.arange(*z_lim,(z_lim[1]-z_lim[0])//TICK_NUM)))
        ax.zaxis.set_major_locator(ticker.MaxNLocator(TICK_NUM))
        # ax.zaxis.set_minor_locator(ticker.MaxNLocator(TICK_NUM*5))
        ax.zaxis.set_tick_params(labelsize=0.65*labelsize)
        # ax.pbaspect = [1.0, 1.0, 0.2]
        ax.view_init(elev=ELEV, azim=AZIM)
        if not axis_tick_flag[2]:
            ax.set_zticklabels(['']*len(ax.get_zticks()))
    else:
        if as_ratio>0:
            ax.set_aspect(as_ratio)
    
    if not axis_tick_flag[0]:
        ax.set_xticklabels(['']*len(ax.get_xticks()))
    if not axis_tick_flag[1]:
        ax.set_yticklabels(['']*len(ax.get_yticks()))

    if title!='' and (title is not None):
        ax.set_title(title, size=labelsize, pad=labelpad)
        # ax_title.set_position([.5, 1.05])


def Read_Pickle(X_fname, y_fname, T2_fname, data_path):
    X = np.array(
        pickle.load(
            open(
                data_path +
                X_fname,
                "rb")).astype(
            np.float32))  # N_sample = 196,587
    # X = (X (-1)*np.mean(X, axis=0)) / np.std(X, axis=0)  # Standardize features
    # X_std = np.std(X, axis=0)
    # print(np.std(X, axis=0))
    y = np.array(
        pickle.load(
            open(
                data_path +
                y_fname,
                "rb")).astype(
            np.int32))
    T2 = np.array(
        pickle.load(
            open(
                data_path +
                T2_fname,
                "rb")).astype(
            np.float32))

    logger.info("Shape of X, y, and T2 are: %s, %s, %s",
                X.shape, y.shape, T2.shape, extra=d)

    return X, y, T2


def Prepare_X_y(df_data, X_drop_cols, y_col, 
                X_mean_train=None, X_std_train=None, 
                y_mean_train=None, y_std_train=None, dtype=np.float32):
    df_X = df_data.drop(columns=X_drop_cols)
    cov_cols = list(df_X)
    X = np.array(df_X, dtype=dtype)
    y = np.array(
        df_data[y_col],
        dtype=dtype)
    if df_data.shape[0]>0:
        X_mean, X_std = X.mean(axis=0), X.std(axis=0)
        std_incr_idx = np.argsort(X_std)
        print(std_incr_idx[:100], [X_std[idx] for idx in std_incr_idx[:100]], [cov_cols[idx] for idx in std_incr_idx[:100]])
        print([X_std[idx] for idx in std_incr_idx[-40:]])
        if np.sum(X_std<1e-6):
            std_ill_idx = np.where(X_std<1e-6)[0]
            print(std_ill_idx, type(std_ill_idx)) 
            ill_colnames = [cov_cols[idx] for idx in std_ill_idx]
            print(ill_colnames)
            # print(df_X.loc[:100, ill_colnames])
            X_std[std_ill_idx] = 1.0
        if X_mean_train is None:
            X = (X-X_mean) / X_std
        else:
            X = (X-X_mean_train) / X_std_train
        # y_mean, y_std = y.mean(), y.std()
        y_mean, y_std = min(y), max(y)-min(y)
        if y_mean_train is None:
            y = (y-y_mean) / y_std
        else:
            y = (y-y_mean_train) / y_std_train
        # y = np.log(1+y_col)
        return cov_cols, X, y, X_mean, X_std, y_mean, y_std
    return cov_cols, X, y, 0, 0, 0, 0


def CR_Read_CSV_Train_And_All_Data(
        X_rest_fname, X_train_fname, y_fname, T2_fname, data_path):
    X_rest = pd.read_csv(
        data_path +
        X_rest_fname,
        sep=',',
        header=None, dtype=np.float32).values
    X_train = pd.read_csv(
        data_path +
        X_train_fname,
        sep=',',
        header=None, dtype=np.float32).values
    y = pd.read_csv(
        data_path + y_fname,
        sep=',',
        header=None,
        dtype=np.int32).values.T[0]  # An one-dimensional vector
    # The starting index is 1, corresponding to 2003 Jan.
    T2 = pd.read_csv(
        data_path + T2_fname,
        sep=',',
        header=None,
        dtype=np.float32).values.T[0]  # An one-dimensional vector

    logger.info("Shape of X_rest, X_train, y, and T2 are: %s, %s, %s, %s",
                X_rest.shape, X_train.shape, y.shape, T2.shape, extra=d)

    return X_rest, X_train, y, T2


def BS_Read_CSV_Train_And_All_Data(Xy_fname, data_folder_path, start_yr=0, end_yr=-1, training_t_len_yr=2, PI_t_len_yr=1, pow=1):
    def Date_Int(delta_days, date):
        delta_days = int(delta_days)
        new_date = date + pd.to_timedelta(delta_days, unit='D')
        return 10000 * new_date.year + 100 * new_date.month + new_date.day
    
    df_bike_sharing = pd.read_csv(
        os.path.join(data_folder_path, Xy_fname),
        index_col=0)
    # Experiments with power of yearly mean when normalizing the bike rental count
    if 'cnt' in list(df_bike_sharing) and 'trailing_wind_mean_cnt' in list(df_bike_sharing):
        df_bike_sharing['trailing_norm_pow_cnt'] = df_bike_sharing[['cnt','trailing_wind_mean_cnt']].apply(lambda x: x[0]/(x[1]**pow), axis=1)
    # print("The difference of two cnts are: {}.".format(df_bike_sharing[['trailing_norm_cnt','trailing_norm_pow_cnt']].apply(lambda x:np.abs(x[0]-x[1]), axis=1).sum()))
    
    # df_bike_sharing = df_bike_sharing.query('yr>=2011').copy()
    df_bike_sharing.loc[:, 'dteday'] = pd.to_datetime(
        df_bike_sharing['dteday'])
    df_bike_sharing.loc[:, 'YrMn'] = df_bike_sharing['dteday'].map(
        lambda x: 100 * x.year + x.month)
    df_bike_sharing.loc[:, 'YrMnDay'] = df_bike_sharing['dteday'].map(
        lambda x: 10000 * x.year + 100 * x.month + x.day)
    # start_date = df_bike_sharing.loc[0, 'dteday'] # If the data start with index 1, this would report error.
    # start_date = df_bike_sharing.iloc[0]['dteday']
    train_start_date = max(df_bike_sharing.iloc[0]['dteday'], datetime(year=start_yr, month=1, day=1))
    df_bike_sharing = df_bike_sharing.query('YrMnDay >= ' + str(Date_Int(0, train_start_date)))
    days_of_year = 365
    print("The start date is {}.".format(train_start_date))

    # One year of data for model building
    # training_t_len_yr = 3
    date_model_int = Date_Int(training_t_len_yr * days_of_year, train_start_date) # All data including training and validation
    df_bike_sharing_train_val = df_bike_sharing.query('YrMnDay < ' + str(date_model_int) + ' & YrMnDay >= ' + str(Date_Int(0, train_start_date)))
    print("The training end date is {}.".format(date_model_int))
    arr_all_idx = np.arange(df_bike_sharing_train_val.shape[0])
    arr_val_idx = np.random.choice(arr_all_idx, size=int(df_bike_sharing_train_val.shape[0] // 3), replace=False)
    df_bike_sharing_train = df_bike_sharing_train_val.iloc[np.setdiff1d(arr_all_idx, arr_val_idx), ] # Still ordered
    df_bike_sharing_val = df_bike_sharing_train_val.iloc[arr_val_idx, ]

    time_stamp_train_val = np.array(df_bike_sharing.query('YrMnDay < ' + str(date_model_int) + ' & YrMnDay >= ' + str(Date_Int(0, train_start_date)))['yr'])

    # One year of data for Phase-I
    date_PI_int = Date_Int((training_t_len_yr+PI_t_len_yr) * days_of_year, train_start_date)
    df_bike_sharing_PI = df_bike_sharing.query('YrMnDay >= ' +
        str(date_model_int) +
        ' & YrMnDay < ' +
        str(date_PI_int))
    print("The PI end date is {}.".format(date_PI_int))

    # The rest data for Phase-II
    if end_yr <= 0:
        df_bike_sharing_PII = df_bike_sharing.query('YrMnDay >= ' + str(date_PI_int))
        time_stamp_PIPII = np.array(df_bike_sharing.query('YrMnDay >= ' + str(date_model_int))['yr'])
    else:
        PII_end_date = min(df_bike_sharing.iloc[-1]['dteday'], datetime(year=end_yr, month=12, day=31))
        date_PII_end_int = Date_Int(0, PII_end_date)
        df_bike_sharing_PII = df_bike_sharing.query('YrMnDay >= ' + str(date_PI_int) + ' & YrMnDay <= ' + str(date_PII_end_int))
        df_bike_sharing = df_bike_sharing.query('YrMnDay <= ' + str(date_PII_end_int)) 
        time_stamp_PIPII = np.array(df_bike_sharing.query('YrMnDay >= ' + str(date_model_int) + ' & YrMnDay <= ' + str(date_PII_end_int))['yr'])

    print("The Entire end date is {}.".format(df_bike_sharing.iloc[-1]['dteday']))

    logging.info(
        ("The size of data for building model: %s(training: %s; validation%s); "
         "Phase-I: %s; Phase-II: %s.\n"),
        df_bike_sharing_train_val.shape,
        df_bike_sharing_train.shape,
        df_bike_sharing_val.shape,
        df_bike_sharing_PI.shape,
        df_bike_sharing_PII.shape,
        extra=d)

    logging.info(
        ("The start date for building model: %s(training: %s; validation%s); "
         "Phase-I: %s; Phase-II: %s.\n"),
        df_bike_sharing_train_val.iloc[0]['dteday'],
        df_bike_sharing_train.iloc[0]['dteday'],
        df_bike_sharing_val.iloc[0]['dteday'],
        df_bike_sharing_PI.iloc[0]['dteday'],
        df_bike_sharing_PII.iloc[0]['dteday'],
        extra=d)

    return df_bike_sharing, df_bike_sharing_train_val, df_bike_sharing_train, df_bike_sharing_val, df_bike_sharing_PI, df_bike_sharing_PII, time_stamp_train_val, time_stamp_PIPII



def Calculate_R2(cov, sigma, coeffs, cov2, sigma2, coeffs2, transf_mat):
    cov_mat = np.matmul(np.matmul(np.transpose(transf_mat), cov), transf_mat)
    cov_mat2 = np.matmul(np.matmul(np.transpose(transf_mat), cov2), transf_mat)
    var = np.matmul(coeffs, np.matmul(cov_mat, coeffs))
    # This is due to special definition of Rs2 in report 20181226
    var2 = np.matmul(coeffs, np.matmul(cov_mat2, coeffs))
    var_cd = np.matmul(coeffs - coeffs2, np.matmul(cov_mat2, coeffs - coeffs2))

    Rs1 = 1 - sigma**2 / (var + sigma**2)
    Rs2 = 1 - (var_cd + sigma2**2) / (var2 + sigma2**2)
    return Rs1, Rs2


class Batch(object):
    def __init__(self, X, y, batch_size, wei=None):
        self.X = X
        self.y = y
        self.wei = wei
        self.batch_size = batch_size
        self.size = X.shape[0]

    def getBatch(self):
        # In the return values for this function, I cannot add the flag
        # "dtype=tf.float32" for y, because it would result in error for logistic
        # regression when response is label (integer type).
        # For poisson regression, the response type should be tf.float32, but
        # but it cannot be transfer here. It should depend on the data generating
        # function for poisson regression.
        indices = np.random.choice(list(range(self.size)), self.batch_size)
        return (tf.convert_to_tensor(self.X[indices, :], dtype=tf.float32),
                tf.convert_to_tensor(self.y[indices]),
                tf.convert_to_tensor(self.wei[indices]) if self.wei is not None else None)


def Build_Model(hidden_layer_sizes, activation, input_dim, output_dim, output_acti=False):
    """Build a neural network model. And the weights has been initialized in return."""
    model = tf.keras.Sequential()

    num_param = 0
    if not hidden_layer_sizes:
        model.add(tf.keras.layers.Dense(output_dim, input_dim=input_dim, autocast=False))
        num_param += output_dim * (input_dim + 1)
    else:
        num_hidnode_prev = input_dim
        for idx, num_hidnode in enumerate(hidden_layer_sizes):
            if idx == 0:
                model.add(
                    # Constrain the norm (square-root of sum-of-square) of
                    # weights into any hidden node and also bias in any layer.
                    # See doc: https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/constraints.py
                    # and: https://www.tensorflow.org/versions/r1.11/api_docs/python/tf/keras/constraints
                    # The reason to add constraints here is that we cannot auto-differentiate
                    # sparse_softmax_cross_entropy() or softmax_cross_entropy,
                    # so that we have to calculate ourselves. However, the gradient
                    # would over-flow if the weights or biases are too large. So
                    # adding constraints on them make the nnet more well-behaved.

                    # This has some propergating effects. Because I add constraints
                    # on bias, so that I have to scale the response y (ideally
                    # we can make it into range [-1, 1]).
                    tf.keras.layers.Dense(
                        num_hidnode,
                        activation=activation,
                        kernel_constraint=max_norm(np.sqrt(10*num_hidnode_prev)),
                        bias_constraint=max_norm(np.sqrt(10*num_hidnode)),
                        input_dim=input_dim,
                        autocast=False))
            else:
                model.add(
                    tf.keras.layers.Dense(
                        num_hidnode,
                        kernel_constraint=max_norm(np.sqrt(10*num_hidnode_prev)),
                        bias_constraint=max_norm(np.sqrt(10*num_hidnode)),
                        activation=activation,
                        autocast=False))
            num_param += num_hidnode * (num_hidnode_prev + 1)
            num_hidnode_prev = num_hidnode
        # The output has no activation because later we will add
        # the logistic or softmax function on the output layer.
        # However, if not added, one-hidden layer may be not nonlinear enough
        model.add(tf.keras.layers.Dense(output_dim,
                        kernel_constraint=max_norm(np.sqrt(10*num_hidnode_prev)),
                        bias_constraint=max_norm(np.sqrt(10*output_dim)),
                        # activation=activation if output_acti else None, # We don't need this, because we can add another hidden layer to output value without hard bound
                        autocast=False))
        num_param += output_dim * (num_hidnode_prev + 1)

    return model, num_param


def Add_Nug_Mat(sym_mat, nugget=0, rcond=RCOND_NUM):
    """ Add conditional number onto a symmetric matrix to ensure the conditional number. """
    # eig_large, _ = eigsh(sym_mat+np.eye(sym_mat.shape[0]), 10, which='LM', maxiter=10*100*sym_mat.shape[0])
    # eig_large -= 1
    eig_large, _ = eigsh(sym_mat, 1, which='LM')
    # When there are several small eigenvalues close to each other, using np.linalg.eigs converges very slowly.
    # eig_small, _ = eigs(sym_mat, 5, which='SM')
    iden_factor = eig_large[0]*rcond
    # logger.info("All the eigen-values are %s.\n", np.linalg.eigvalsh(sym_mat), extra=d)
    logger.info("The largest eigen-value is {}.\n".format(eig_large, ), extra=d)
    logger.info(("The identity matrix need to add: a nugget %s and the "
                 "factor to ensure condition number is %s."), nugget, iden_factor, extra=d)
    nugget = max(nugget, iden_factor)
    return sym_mat + nugget*np.identity(sym_mat.shape[0])


def Inv_Mat_Rcond(sym_mat, nugget=0, rcond=RCOND_NUM):
    """ Invert a matrix to ensure a condition number. """
    sym_mat = Add_Nug_Mat(sym_mat, nugget, rcond)
    return np.linalg.inv(sym_mat)


def Inv_Cov(score_vecs, nugget=0, rcond=RCOND_NUM, wei=None):
    """ Invert covariance matrix of the score vectors. """
    if wei is None:
        wei = np.ones((score_vecs.shape[0],1))
    score_mu = np.sum(score_vecs, axis = 0)/np.sum(wei)
    score_centered = score_vecs - score_mu * wei
    S = np.dot(np.transpose(score_centered), score_centered/wei) / np.sum(wei)
    return Inv_Mat_Rcond(S, nugget, rcond)

def Vert_Y_Cal_Wei(y, num_class, yweight):
    y = np.vstack(y)
    wei = np.ones_like(y)
    if num_class==2 and yweight>1:
        wei[y==1] = yweight
    return y, wei

# def InversedCov(score_mm, nugget=0, rcond=RCOND_NUM):
#     # S = np.dot(np.transpose(score_mm), score_mm) / score_mm.shape[0]
#     # # print("Covariance matrix-----------\n")
#     # # print S
#     # logger.info('Condition #: %s', np.linalg.cond(S), extra=d)
#     # S = S + nugget * np.diag(np.repeat(1, S.shape[0]), 0)
#     # # logger.info('Condition # (delta): %s', np.linalg.cond(S), extra=d)
#     # Sinv = np.linalg.inv(S)
#     # # Sinv = np.linalg.pinv(S, rcond)
#     # logger.info('Condition # (rcond): %s', np.linalg.cond(Sinv), extra=d)
#     score_mu = np.mean(score_mm, axis = 0)
#     score_centered = score_mm - score_mu
#     S = np.dot(np.transpose(score_centered), score_centered) / score_centered.shape[0]
#     # print("Covariance matrix-----------\n")
#     # print S
#     logger.info('Coveriance Matrix Condition #(before adding nugget): %s', np.linalg.cond(S), extra=d)
#     S_nugget_cnt = 0
#     while np.linalg.cond(S)>1/rcond:
#         S_nugget_cnt += 1
#         logger.info("The score covariance matrix is singular and add a diagonal matrix, %s times.", S_nugget_cnt, extra=d)
#         S += nugget * np.identity(S.shape[0])
#     # S = S + nugget * np.identity(S.shape[0])
#     # logger.info('Condition # (delta): %s', np.linalg.cond(S), extra=d)
#     Sinv = np.linalg.inv(S)
#     # Sinv = np.linalg.pinv(S, rcond)
#     logger.info('Coveriance Matrix Condition #(after adding nugget, %s times): %s', S_nugget_cnt, np.linalg.cond(Sinv), extra=d)
#     return Sinv

def Comp_Oper_Square(arr):
    return arr**2

def Gen_CV_Info(penal_param, stopping_lag, training_batch_size, learning_rate, FLAGS):
    """ Generate CV information for deploying cross-validation. """
    if FLAGS.cv_flag:
        (N_rep, K_fold, cv_param_ls, cv_tasks_ls, cv_rand_search, n_jobs) = (
            FLAGS.cv_N_rep, FLAGS.cv_K_fold, FLAGS.cv_param_ls, 
            FLAGS.cv_task_param_ls, FLAGS.cv_rand_search, FLAGS.cv_n_jobs)
        cv_param_ls = cv_param_ls.split('-')
        cv_tasks_ls = cv_tasks_ls.split('-')
        # The cv for detection speed simulations.
        # if 'penal_param' in cv_tasks_ls:
        #     penal_param_ls = [10**i for i in range(-3, 2)]
        # else:
        #     penal_param_ls = [penal_param]
        # if 'stopping_lag' in cv_tasks_ls:
        #     stopping_lag_ls = [100*i for i in range(4,11)]
        # else:
        #     stopping_lag_ls = [stopping_lag]
        # if 'training_batch_size' in cv_tasks_ls:
        #     training_batch_size_ls = [int(10**i) for i in range(1,4)]
        # else:
        #     training_batch_size_ls = [training_batch_size]
        # if 'learning_rate' in cv_tasks_ls:
        #     learning_rate_ls = [10**i for i in range(-3,1)]

        # The cv for materials detection.
        # Total combination number is 5*7*3*3=315
        if 'penal_param' in cv_tasks_ls:
            penal_param_ls = [10**i for i in range(-8, 3)]
        else:
            penal_param_ls = [penal_param]
        if 'stopping_lag' in cv_tasks_ls:
            stopping_lag_ls = [100*i for i in range(4,11)]
        else:
            stopping_lag_ls = [stopping_lag]
        if 'training_batch_size' in cv_tasks_ls:
            training_batch_size_ls = [int(10**i) for i in range(1,4)]
        else:
            training_batch_size_ls = [training_batch_size]
        if 'learning_rate' in cv_tasks_ls:
            learning_rate_ls = [10**i for i in range(-5,0)]
        else:
            learning_rate_ls = [learning_rate]
        cv_tasks = [[p1, p2, p3, p4] for (p1, p2, p3, p4) in itertools.product(
            penal_param_ls, stopping_lag_ls, training_batch_size_ls, learning_rate_ls)]
        cv_tasks_info = (N_rep, K_fold, cv_param_ls, cv_rand_search, cv_tasks, n_jobs)
    else:
        cv_tasks_info = None

    return cv_tasks_info


def CV_Shuffle_Index(num_sample, K_fold):
    """Generate index for cross validation using random shuffle.
    Args:
        num_sample: The number of samples in the data set.
        K_fold: The number of folds to do cross validation.

    Returns:
        index_array_list: A list of array of indices to do cross validation.
    """
    index_permu = np.random.choice(
        num_sample, num_sample, replace=False)
    step = math.floor(num_sample / K_fold)
    start_index = np.arange(0, num_sample, int(step))
    index_array_list = []
    for i in range(K_fold-1):
        index_array_list.append(index_permu[start_index[i]:start_index[i + 1]])
    if int(step) * K_fold < num_sample:
        index_array_list.append(index_permu[start_index[K_fold-1]:start_index[K_fold]])
        for i in range(int(num_sample - step * K_fold)):
            index_array_list[i] = np.append(
                index_array_list[i], index_permu[int(step * K_fold) + i])
    else:
        index_array_list.append(index_permu[start_index[K_fold-1]:])
    return index_array_list


def CV_Stratified_Shuffle_Index_Upsample(X, y, K_fold):
    skf = StratifiedKFold(n_splits=K_fold)
    index_array_list = []
    for _, test_idx in skf.split(X,y):
        index_array_list.append(test_idx)
    print(index_array_list)
    return index_array_list

def CV_Shuffle_Index_Upsample(X, y, K_fold):
    skf = KFold(n_splits=K_fold)
    index_array_list = []
    for _, test_idx in skf.split(X,y):
        index_array_list.append(test_idx)
    print(index_array_list)
    return index_array_list


def CV_Train_Nnet(gen_model_func, gen_model_func_param, initial_weights_file, Train_Nnet,
                  loss, pred, cv_log_folder, rep_idx, cv_idx, fold_idx, X, y, wei, train_idx_ls, val_idx_ls,
                  penal_param, stopping_lag, training_batch_size,
                  learning_rate, FLAGS):
    """ A wrapper for training nnet for CV. """
    logger.info("The cv job: rep_idx: %s, cv_idx: %s, fold_idx: %s.", rep_idx, cv_idx, fold_idx, extra=d)
    model_ckpt_name = '_'.join([str(rep_idx), str(cv_idx), str(fold_idx)]) + ".h5"
    trained_model, val_loss_value, best_index, val_metric = Train_Nnet(
                       gen_model_func, gen_model_func_param, initial_weights_file,
                       loss, pred, X, y, wei, train_idx_ls, val_idx_ls,
                       penal_param, stopping_lag, training_batch_size,
                       learning_rate, model_ckpt_name, FLAGS, cv_log_folder)

    return (rep_idx, cv_idx, fold_idx, penal_param, stopping_lag, training_batch_size,
            learning_rate, best_index, len(val_idx_ls) * val_loss_value, val_metric)


def CV_Nnet(gen_model_func, gen_model_func_param, initial_weights_file,
            Train_Nnet, loss, pred, X, y, wei, N_rep, K_fold, cv_param_ls, cv_rand_search, cv_tasks, n_jobs, FLAGS):
    """ Cross-validate the model using the tasks. """
    N_sample = X.shape[0]
    all_tasks = []
    # initial_model_name = "cv_initial_model.h5"
    # pickle.dump(model, open(os.path.join(FLAGS.training_res_folder, initial_model_name),'wb'))
    cv_folder = "cv_folder"
    cv_folder_path = os.path.join(FLAGS.training_res_folder, cv_folder)
    print("The cv folder is {}.".format(cv_folder_path))
    # Create a folder for cv files or delete previous files in it.
    if not os.path.exists(cv_folder_path):
        os.makedirs(cv_folder_path)
    else:
        for fname in os.listdir(cv_folder_path):
            os.remove(os.path.join(cv_folder_path, fname))

    # Use randomized search instead of grid-search.
    if cv_rand_search > 0:
        print("The number of tasks and cv random search: {}, {}.".format(len(cv_tasks), cv_rand_search))
        arr_rand_idx = np.random.choice(len(cv_tasks), cv_rand_search, replace=False)
        cv_tasks = [cv_tasks[i] for i in arr_rand_idx]

    # Else cross-validation on all possible parameters.
    for rep_idx in range(N_rep):
        # fold_index_ls = CV_Shuffle_Index(N_sample, K_fold)
        if FLAGS.reg_model.endswith('lin'):
            fold_index_ls = CV_Shuffle_Index_Upsample(X, y, K_fold)
        else:
            fold_index_ls = CV_Stratified_Shuffle_Index_Upsample(X, y, K_fold)
        for cv_idx, cv_task in enumerate(cv_tasks):
            for fold_idx, val_idx_ls in enumerate(fold_index_ls):
                # new_model = pickle.load(open(os.path.join(FLAGS.training_res_folder, initial_model_name),'rb'))
                train_idx_ls = [ind for ind in range(X.shape[0]) if ind not in val_idx_ls]
                all_tasks.append([gen_model_func, gen_model_func_param,
                    initial_weights_file, Train_Nnet, loss, pred, cv_folder, rep_idx, cv_idx,
                    fold_idx, X, y, wei, train_idx_ls, val_idx_ls] + cv_task + [FLAGS])
    
    ls_env_var_strs = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']
    logger.info("The dictionary of environment variables is %s.", os.environ, extra=d)
    for env_var in ls_env_var_strs:
        if env_var in os.environ:
            logger.info("The environment variable %s is %s.", env_var, os.environ[env_var], extra=d)

    cv_time_start = time.time()
    with parallel_backend('loky', n_jobs=n_jobs):
        cv_res = Parallel(verbose=10, pre_dispatch=FLAGS.cv_pre_dispatch)(
            delayed(CV_Train_Nnet)(*task) for task in all_tasks)
    logger.info(("The cross-validation of %s replication, %s folds, "
                 "and %s different settings takes time %s."),
                N_rep, K_fold, len(cv_tasks), time.time() - cv_time_start, extra=d)

    df_cv_res = pd.DataFrame(data=np.array(cv_res), columns=['rep_idx', 'cv_idx', 'fold_idx'] + cv_param_ls + ['best_index', 'val_loss', 'val_metric'])
    df_cv_res.to_csv(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), "cv_res.csv"), header=True, index=False)

    df_cv_res_mean = df_cv_res.groupby(['cv_idx']).mean()
    df_cv_res_mean.to_csv(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), "cv_res_mean.csv"), header=True, index=False)

    best_param_row_idx = df_cv_res_mean['val_loss'].idxmin()

    logger.info("The best cv parameter is %s.", df_cv_res_mean.loc[[best_param_row_idx], cv_param_ls], extra=d)

    df_cv_res_mean.loc[[best_param_row_idx], cv_param_ls].to_csv(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), "best_cv_res.csv"), header=True, index=False)

    return df_cv_res_mean.loc[best_param_row_idx, cv_param_ls]


def Agg_Gamma_Data(FLAGS):
    """ Aggregate gamma data from replicates. """
    output_file_path = os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'agg_gamma.csv')
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)
    with open(output_file_path, 'a') as out_f:
        header = True
        for i in range(FLAGS.N_rep_find_gamma+1):
            fname = '_'.join([str(i), 'nnet', str(FLAGS.nnet), 'score_gamma'])+'.csv'
            fname_path = os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), fname)
            print(fname_path)
            df_fdata = pd.read_csv(fname_path)
            df_fdata.to_csv(out_f, header=header, index=False)
            if header:
                header = False
    df_all_gamma_res = pd.read_csv(output_file_path)
    df_all_gamma_res_mean = df_all_gamma_res.groupby(["change_factor"]).mean().reset_index()
    ave_output_file_path = os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'ave_agg_gamma.csv')
    df_all_gamma_res_mean.to_csv(ave_output_file_path, header=True, index=False)

# Calculate gradient in batch
def grad_func_batch(
        X_batch,
        y_batch,
        gen_model_func,
        gen_model_func_param,
        grad_func,
        wei_batch=None,
        **kwargs):
    """Calculate gradient vectors for a batch of dataset."""
    model, _ = gen_model_func(*gen_model_func_param)
    model.set_weights(kwargs['model_weights'])
    hidden_layer_sizes = gen_model_func_param[0]
    grad_func_kwargs = {}
    for key, val in kwargs.items():
        if key=='model_weights':
            grad_func_kwargs['model'] = model
        else:
            grad_func_kwargs[key] = val
    grads = []
    y_batch = np.vstack(y_batch)
    for i in range(X_batch.shape[0]):
        x = tf.convert_to_tensor(X_batch[[i]], dtype=tf.float32)
        y = tf.convert_to_tensor(y_batch[[i]])
        grad_raw = grad_func(x, y, wei=wei_batch[[i]] if wei_batch is not None else wei_batch, **grad_func_kwargs)
        # print(grad_raw)
        grad_np_vec = np.array([])
        for j in range(len(hidden_layer_sizes) + 1):
            # Already checked the dimension of gradient.
            # For the i to (i+1)th layer (input layer is special layer,
            # let's call it layer 0), the weights are stored in 2*i
            # of grad_raw, and biases are stored in 2*i+1 of grad_raw.
            # The values are stored in grad_raw[2*i] and grad_raw[2*i+1]
            # as matrices and arrays. jth column in the matrics corresponds
            # to jth nodes of ith layer; jth element in the arrays corresponds
            # to jth nodes of ith layer.
            # There is no value of weights.
            #
            # gradient for weights and biases
            grad_np_vec = np.append(
                grad_np_vec, grad_raw[2 * j].numpy())
            grad_np_vec = np.append(
                grad_np_vec, grad_raw[2 * j + 1].numpy())
        grads.append(grad_np_vec)
    
    # # Parallized function to calculate gradients
    # def vectorize_grad(idx, x, y, hidden_layer_sizes, grad_func, kwargs):
    #     x = tf.convert_to_tensor(x, dtype=tf.float32)
    #     y = tf.convert_to_tensor(y)
    #     grad_raw = grad_func(x, y, **kwargs)
    #     # print(grad_raw)
    #     grad_np_vec = np.array([])
    #     for j in range(len(hidden_layer_sizes) + 1):
    #         grad_np_vec = np.append(
    #             grad_np_vec, grad_raw[2 * j].numpy())
    #         grad_np_vec = np.append(
    #             grad_np_vec, grad_raw[2 * j + 1].numpy())
    #     return idx, grad_np_vec
    # ls_tasks = [(idx, copy.deepcopy(x), copy.deepcopy(y), hidden_layer_sizes, grad_func, copy.deepcopy(kwargs)) for idx, (x, y) in enumerate(zip(X_batch, y_batch))]
    # print(ls_tasks[:3])

    # with parallel_backend('loky', n_jobs=5):
    #     res = Parallel(verbose=1, pre_dispatch="1.5*n_jobs")(delayed(vectorize_grad)(*task) for task in ls_tasks)
    # res.sort(key=lambda x: x[0]) # Sort incrementally inplace
    # _, grads = zip(*res)
    return np.array(grads)

# This version of grad_func_batch is designed for implicit_gradient.
# After I change to the tf.GradientTape, I need to change the way to
# assemble gradients.

# def cal_penal_bool_vec(
#         x,
#         y,
#         hidden_layer_sizes,
#         grad_func,
#         **kwargs):
#     x = tf.convert_to_tensor(x, dtype=tf.float32)
#     y = tf.convert_to_tensor(y)
#     grad_raw = grad_func(x, y, **kwargs)
#     print("The gradient:")
#     print(grad_raw)
#     time.sleep(1000)
#     penal_bool_vec = np.array([])
#     for i in xrange(len(hidden_layer_sizes) + 1):
#         # Already checked the dimension of gradient.
#         # For the i-1 to ith layer (input layer is special layer,
#         # let's call it layer -1), the weights are stored in 2*i
#         # of grad_raw, and biases are stored in 2*i+1 of grad_raw.
#         # The values are stored in grad_raw[2*i][0] and grad_raw[2*i+1][0]
#         # as matrices and arrays. jth column in the matrics corresponds
#         # to jth nodes of ith layer; jth element in the arrays corresponds
#         # to jth nodes of ith layer.
#         # grad_raw[2*i][0] stores gradients while grad_raw[2*i][1]
#         # stores corresponding weight or bias values.
#         #
#         # gradient for weights and biases
#         penal_bool_vec = np.append(
#             penal_bool_vec, np.ones_like(grad_raw[2 * i][0].numpy()))
#         penal_bool_vec = np.append(
#             penal_bool_vec, np.zeros_like(grad_raw[2 * i + 1][0].numpy()))
#         # print("The sample gradient {} with length {}".format(grads_PI[-1], grads_PI[-1].shape))
#     return penal_bool_vec

# def grad_func_batch(
#         X_batch,
#         y_batch,
#         hidden_layer_sizes,
#         grad_func,
#         **kwargs):
#     """Calculate gradient vectors for a batch of dataset."""
#     for i in xrange(X_batch.shape[0]):
#         x = tf.convert_to_tensor(X_batch[[i]], dtype=tf.float32)
#         y = tf.convert_to_tensor(y_batch[[i]])
#         grad_raw = grad_func(x, y, **kwargs)
#         # print(grad_raw)
#         grad_np_vec = np.array([])
#         for j in xrange(len(hidden_layer_sizes) + 1):
#     # Already checked the dimension of gradient.
#     # For the i to (i+1)th layer (input layer is special layer,
#     # let's call it layer 0), the weights are stored in 2*i
#     # of grad_raw, and biases are stored in 2*i+1 of grad_raw.
#     # The values are stored in grad_raw[2*i][0] and grad_raw[2*i+1][0]
#     # as matrices and arrays. jth column in the matrics corresponds
#     # to jth nodes of ith layer; jth element in the arrays corresponds
#     # to jth nodes of ith layer.
#     # grad_raw[2*i][0] stores gradients while grad_raw[2*i][1]
#     # stores corresponding weight or bias values.
#     #
#     # gradient for weights and biases
#             grad_np_vec = np.append(
#                 grad_np_vec, grad_raw[2 * j][0].numpy())
#             grad_np_vec = np.append(
#                 grad_np_vec, grad_raw[2 * j + 1][0].numpy())
#     return grad_np_vec

def grad_func_batch_paral(
        X_batch,
        y_batch,
        gen_model_func,
        gen_model_func_param,
        grad_func,
        wei_batch=None,
        num_blocks=UTIL_TASK_NJOBS,
        **kwargs):
    """Calculate gradient vectors for a batch of dataset."""
    # grads = []
    y_batch = np.vstack(y_batch)
    sub_size = int(math.ceil(X_batch.shape[0]/num_blocks))
    ls_X_batchs = [X_batch[i*sub_size:min((i+1)*sub_size, X_batch.shape[0])] for i in range(num_blocks)]
    ls_y_batchs = [y_batch[i*sub_size:min((i+1)*sub_size, y_batch.shape[0])] for i in range(num_blocks)]
    ls_wei_batchs = [wei_batch[i*sub_size:min((i+1)*sub_size, y_batch.shape[0])] if wei_batch is not None else None for i in range(num_blocks)]

    # Tensorflow model cannot be past as parameters to function for joblib parallelization.
    def grad_func_batch_helper(idx, X_batch, y_batch, gen_model_func, gen_model_func_param, grad_func, wei_batch=None, kwargs={}):
        return idx, grad_func_batch(X_batch, y_batch, gen_model_func, gen_model_func_param, grad_func, wei_batch=wei_batch, **kwargs)
    
    ls_tasks = [(idx, X, y, gen_model_func, gen_model_func_param, grad_func, wei, kwargs) for idx, (X, y, wei) in enumerate(zip(ls_X_batchs, ls_y_batchs, ls_wei_batchs))]

    with parallel_backend('loky', n_jobs=num_blocks):
        res = Parallel(verbose=1, pre_dispatch="1.5*n_jobs")(delayed(grad_func_batch_helper)(*task) for task in ls_tasks)
    res.sort(key=lambda x: x[0]) # Sort incrementally inplace
    _, ls_grads = zip(*res)
    return np.concatenate(ls_grads, axis=0)

# Calculate ewma Hotelling T2 statistics
def Scores_ewma(score, gamma, start):
    if gamma > 0:
        N = score.shape[0]
        npar = score.shape[1]
        out = np.zeros((N, npar))
        last_ewma = start
        start_time = time.time()
        for t in range(N):
        #   new ewma                  history             current score
            out[t, :] = (1 - gamma) * last_ewma + gamma * score[t, :]
            last_ewma = out[t, :]
        print("Calculate ewma score with shape {} takes {}s.".format(score.shape, time.time()-start_time))
        return out
    else:
        return score


def Cal_Multi_Chart(ls_arr_t2_sewma_PI, ls_arr_dev_PI, ls_arr_t2_sewma_PII, ls_arr_dev_PII, lbd_alarm_level, ubd_alarm_level, alarm_level, 
                    max_iter=30, tol=1e-6, multi_chart_scale_flag='mid'):
    def cal_out_of_control(ls_arr_PI, alarm_level, ucl=None, lcl=None, multi_chart_scale_flag='mid'):
        arr = np.array(ls_arr_PI)
        
        if ucl is None or lcl is None:
            if multi_chart_scale_flag == 'mid':
                ucl = np.percentile(arr, (100+alarm_level)/2)
                lcl = np.percentile(arr, (100-alarm_level)/2)
            else:
                ucl = np.percentile(arr, alarm_level)
                lcl = np.percentile(arr, 0)
        
        if multi_chart_scale_flag == 'mid':
            mid = (ucl+lcl)/2
            wid = (ucl-lcl)/2
            arr_scaled = (arr-mid)/wid
        elif multi_chart_scale_flag == 'ucl':
            arr_scaled = arr/(ucl/2)-1
        else:
            arr_scaled = np.logical_or(arr>ucl, arr<lcl)
        
        logger.info("The multichart function has been changed!!!!!!", extra=d)
        
        return ucl, lcl, arr_scaled

    def comb_control_chart(arr1, arr2, multi_chart_scale_flag='mid'):
        if multi_chart_scale_flag=='mid':
            arr_sign = np.sign(arr1+arr2)
            # num_zero = np.sum(arr_sign==0)
            # logger.info("The number and portion of sign being 0 %s, %s.", num_zero, num_zero/np.size(arr_sign), extra=d)
            arr_sign[arr_sign==0] = 1 # If two plots have the exact the same opposite sign, just use 1.
            arr_multi_chart = arr_sign.astype(np.float32)*np.maximum(np.abs(arr1), np.abs(arr2))
            comb_out_contr = np.sum(np.logical_or(arr_multi_chart>1, arr_multi_chart<-1))/np.size(arr_multi_chart)
        elif multi_chart_scale_flag=='ucl':
            arr_multi_chart = np.maximum(arr1, arr2)
            comb_out_contr = np.sum(arr_multi_chart>1)/np.size(arr_multi_chart)
        else:
            arr_multi_chart = np.logical_or(arr1, arr2).astype(np.int32)
            comb_out_contr = np.sum(arr_multi_chart)/np.size(arr_multi_chart)
        return arr_multi_chart, comb_out_contr

    cnt = 0
    old_comb_out_contr, comb_out_contr = 0, -1
    start_time = time.time()
    while True:
        mid_alarm_level = (lbd_alarm_level+ubd_alarm_level)/2
        ucl1, lcl1, arr_scale1_PI = cal_out_of_control(ls_arr_t2_sewma_PI, alarm_level=mid_alarm_level, multi_chart_scale_flag='ucl')
        ucl2, lcl2, arr_scale2_PI = cal_out_of_control(ls_arr_dev_PI, alarm_level=mid_alarm_level, multi_chart_scale_flag=multi_chart_scale_flag)
        arr_multi_chart_PI, comb_out_contr = comb_control_chart(arr_scale1_PI, arr_scale2_PI, multi_chart_scale_flag=multi_chart_scale_flag)

        if comb_out_contr<(100-alarm_level)/100: # alarm_level too high
            ubd_alarm_level = mid_alarm_level
        else:
            lbd_alarm_level = mid_alarm_level
        cnt += 1
        logger.info("Iteration %s (%s,%s,%s,%s,%s,%s).", cnt, mid_alarm_level, ucl1, lcl1, ucl2, lcl2, comb_out_contr, extra=d)
        if cnt>= max_iter or np.abs(comb_out_contr-old_comb_out_contr)<tol:
            logger.info("(The iterations end at %s with tolerance of %s; time %s s) Out-of-control ratio: %s; Alarm rate for individual control charts: %s.", cnt, np.abs(comb_out_contr-old_comb_out_contr), time.time()-start_time, comb_out_contr, mid_alarm_level, extra=d)
            break
        old_comb_out_contr = comb_out_contr

    _, _, arr_scale1_PII = cal_out_of_control(ls_arr_t2_sewma_PII, alarm_level=mid_alarm_level, ucl=ucl1, lcl=lcl1, multi_chart_scale_flag='ucl')
    _, _, arr_scale2_PII = cal_out_of_control(ls_arr_dev_PII, alarm_level=mid_alarm_level, ucl=ucl2, lcl=lcl2, multi_chart_scale_flag=multi_chart_scale_flag)
    arr_multi_chart_PII, _ = comb_control_chart(arr_scale1_PII, arr_scale2_PII, multi_chart_scale_flag=multi_chart_scale_flag)

    return arr_multi_chart_PI, arr_multi_chart_PII


def Cal_Out_of_Control(ls_arr_PI, ls_arr_PII, alarm_level, cl_flag='both'):
    if cl_flag == 'both':
        ucl = np.percentile(np.array(ls_arr_PI), (100+alarm_level)/2)
        lcl = np.percentile(np.array(ls_arr_PI), (100-alarm_level)/2)
    else:
        ucl = np.percentile(np.array(ls_arr_PI), alarm_level)
        lcl = np.percentile(np.array(ls_arr_PI), 0)

    def por_out_contr(arr, ucl, lcl, cl_flag='both'):
        if cl_flag == 'both':
            return np.sum(np.logical_or(arr>ucl, arr<lcl))/np.size(arr)
        else:
            return np.sum(arr>ucl)/np.size(arr)

    ls_out_contr_PI, ls_out_contr_PII = [], []
    for arr in ls_arr_PI:
        ls_out_contr_PI.append(por_out_contr(arr, ucl, lcl, cl_flag=cl_flag))
    for arr in ls_arr_PII:
        ls_out_contr_PII.append(por_out_contr(arr, ucl, lcl, cl_flag=cl_flag))
    return np.array(ls_out_contr_PI), np.array(ls_out_contr_PII)

def My_Split(astr):
    if len(astr)==0:
        return []
    else:
        res = astr.split(',')
        res = [int(ele) for ele in res]
        return res

# # Calculate ewma Hotelling T2 statistics in parallel
# def Scores_ewma_paral(score, gamma, start, cutoff_ratio=1e-6):
#     if gamma > 0:
#         trunc_wind_size = int(cutoff_ratio/gamma)
#         ewma_wei = [gamma]
#         for i in range(trunc_wind_size):
#             ewma_wei.append(ewma_wei[-1]*(1-gamma))
#         ewma_wei = np.array(ewma_wei[::-1])

def Weighted_Mean_Dev(vec, wei=None):
    if wei is None:
        wei = np.ones((vec.shape[0],1))
    weighted_mean = np.sum(vec*wei, axis=0)/np.sum(wei)
    weighted_std = np.sqrt(np.sum((vec-weighted_mean)**2*wei, axis=0)/(np.sum(wei)-1))
    return weighted_mean, weighted_std


def HotellingT2(x, mu, Sinv, dtype=np.float32):
    tmp = x - mu
    return np.sqrt(np.dot(np.dot(tmp, Sinv), np.transpose(tmp)), dtype=dtype)

def HotellingT2Helper(t, x, mu, Sinv, dtype=np.float32):
    tmp = x - mu
    return t, np.sqrt(np.dot(np.dot(tmp, Sinv), np.transpose(tmp)), dtype=dtype)

# Require too much memory because Sinv2 is a square matrix and for a large number of predictors
# this would blow the memory.
# def HotellingT2Paral(score_ewma, mu_train, Sinv2, gamma, dtype=np.float32, flag_PI=True):
#     if flag_PI:
#         ls_tasks = [(t, score_ewma[t,:], mu_train, Sinv2/(1 - (1 - gamma)**(2 * (1 + t))), dtype)
#                     for t in range(score_ewma.shape[0])]
#     else:
#         ls_tasks = [(t, score_ewma[t,:], mu_train, Sinv2, dtype)
#                     for t in range(score_ewma.shape[0])]
#     with parallel_backend('loky', n_jobs=UTIL_TASK_NJOBS):
#         res = Parallel(verbose=1, pre_dispatch="1.5*n_jobs")(delayed(HotellingT2Helper)(*task) for task in ls_tasks)
#     res.sort(key=lambda  x: x[0]) # Sort incrementally inplace
#     # print(res[:10])
#     _, t2ewma = zip(*res)
#     return np.array(t2ewma)


def MonIdx2Date(time_stamp, date_index_flag, to_decode=False):
    if date_index_flag and to_decode:
        # The following three lines are for credit risk data showing year index
        time_stamp = int(time_stamp)
        month = (time_stamp - 1) % 12 + 1
        date_int = 300 + (time_stamp - month) / 12 * 100 + month 
        return str(int(date_int//100))+'-'+str(int(date_int%100)) # Data starts from 2003.
        # return(300 + (time_stamp - month) / 12 * 100 + month) # Data starts from 2003.
    else:
        # For bike sharing date showing year index or other simulated dataset
        return str(int(time_stamp))


def calPenalBoolVec(
            x,
            y,
            hidden_layer_sizes,
            grad_func,
            **kwargs):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y)
        grad_raw = grad_func(x, y, **kwargs)
        penal_bool_vec = np.array([])
        for i in range(len(hidden_layer_sizes) + 1):
            # Already checked the dimension of gradient.
            # For the i to (i+1)th layer (input layer is special layer,
            # let's call it layer 0), the weights are stored in 2*i
            # of grad_raw, and biases are stored in 2*i+1 of grad_raw.
            # The values are stored in grad_raw[2*i] and grad_raw[2*i+1]
            # as matrices and arrays. jth column in the matrics corresponds
            # to jth nodes of ith layer; jth element in the arrays corresponds
            # to jth nodes of ith layer.
            # There is no value of weights.
            #
            # gradient for weights and biases
            penal_bool_vec = np.append(
                penal_bool_vec, np.ones_like(grad_raw[2 * i].numpy()))
            penal_bool_vec = np.append(
                penal_bool_vec, np.zeros_like(grad_raw[2 * i + 1].numpy()))
            # print("The sample gradient {} with length {}".format(grads_PI[-1], grads_PI[-1].shape))
        return penal_bool_vec


# Those following functions are for calculating EWMA of statistics.
def calEwmaStatisticsHelper(fp_score_mark):
    """
        fp_score_mark: a list of markers to subset dataset.
    """
    fp_score_idx = np.where(fp_score_mark == 1)[0]
    fp_score = 1.0 * np.sum(fp_score_mark) / len(fp_score_mark)
    if len(fp_score_idx)==0:
        rl_fp_score = len(fp_score_mark) # Run length
        len_after_detect=0
        signal_ratio = 0 # Signal ratio after the first detection.
        flag_rl = False
    else:
        rl_fp_score = fp_score_idx[0]
        len_after_detect = len(fp_score_mark)-fp_score_idx[0]
        signal_ratio = 1.0*np.sum(fp_score_mark[fp_score_idx[0]:])/len_after_detect
        flag_rl = True
    return (fp_score, rl_fp_score, len_after_detect, signal_ratio, flag_rl)


def EwmaT2PI(score_PI, mu_train, Sinv_train, gamma,
        eff_wind_len_factor, nugget, start_PI, dtype=np.float32):
    # Phase I data
    # N_PI = score_PI.shape[0]
    # Sinv = InversedCov(score_PI, nugget)  # nnet needs 0.15
    # mu = np.mean(score_PI, axis=0)

    ext_len = int(eff_wind_len_factor/gamma)
    if ext_len>0:
        score_PI_ext = np.vstack((score_PI[-ext_len:,:], score_PI)).astype(dtype) # corner case when ext_len=0
    else:
        score_PI_ext = score_PI
    N_PI_ext = score_PI_ext.shape[0]

    score_ewma = Scores_ewma(score_PI_ext, gamma, start_PI).astype(dtype)  # starting point
    # score_ewma = Scores_ewma(score, gamma, np.zeros(Sinv.shape[0])) #
    # starting point
    Sinv2 = Sinv_train * (2 - gamma) / gamma
    start_time = time.time()
    # t2ewmaI = HotellingT2Paral(score_ewma, mu_train, Sinv2, gamma, dtype=dtype, flag_PI=True)
    t2ewmaI = np.zeros((N_PI_ext, ))
    for t in range(N_PI_ext):
        # Statistical quality control-(11.32)
        t2ewmaI[t] = HotellingT2(
            score_ewma[t, :], mu_train, Sinv2 / (1 - (1 - gamma)**(2 * (1 + t))))
    print("Calculate Hotelling T2 of score with shape {} takes {}s.".format(score_PI.shape, time.time()-start_time))
    
    start_PII = score_ewma[-1, :]

    return ext_len, start_PII, score_ewma, t2ewmaI


def EwmaT2PII(score_PII, mu_train, Sinv_train, gamma, start_PII, dtype=np.float32):
    # Phase II data:
    N_PII = score_PII.shape[0]
    # scoreII = score(model, X_test, y_test, reg_val)
    # score_ewmaII = Scores_ewma(score_PII, gamma, score_ewma[-1, :])
    score_ewmaII = Scores_ewma(score_PII, gamma, start_PII).astype(dtype)
    Sinv2 = Sinv_train * (2 - gamma) / gamma
    start_time = time.time()
    # t2ewmaII = HotellingT2Paral(score_ewmaII, mu_train, Sinv2, gamma, dtype=dtype, flag_PI=False)
    t2ewmaII = np.zeros((N_PII,))
    for t in range(N_PII):
        t2ewmaII[t] = HotellingT2(score_ewmaII[t, :], mu_train, Sinv2)
    print("Calculate Hotelling T2 of score with shape {} takes {}s.".format(score_PII.shape, time.time()-start_time))

    return score_ewmaII, t2ewmaII


def EwmaPI(comp_PI, gamma, eff_wind_len_factor, start_PI):
    ext_len = int(eff_wind_len_factor/gamma)
    if ext_len>0:
        comp_PI_ext = np.hstack((comp_PI[-ext_len:], comp_PI))
    else:
        comp_PI_ext = comp_PI

    ewma_comp_PI = Scores_ewma(np.reshape(comp_PI_ext, (-1, 1)), gamma, start_PI)
    ewma_comp_PI = np.reshape(ewma_comp_PI, (-1,))

    start_PII = ewma_comp_PI[-1]

    return ext_len, start_PII, ewma_comp_PI


def EwmaPII(comp_PII, gamma, start_PII):
    # Phase II data:
    N_PII = comp_PII.shape[0]
    ewma_comp_PII = Scores_ewma(np.reshape(comp_PII, (-1, 1)), gamma, start_PII)
    ewma_comp_PII = np.reshape(ewma_comp_PII, (-1,))

    return ewma_comp_PII


def calEwmaT2StatisticsPI(offset, alarm_level, score_PI,
        mu_train, Sinv_train, gamma, eff_wind_len_factor, nugget, start_PI):
    """Calculate statistics for Phase-I for calculation for Phase-II."""
    #
    ext_len, start_PII, _, t2ewmaI = EwmaT2PI(score_PI, mu_train, Sinv_train,
        gamma, eff_wind_len_factor, nugget, start_PI)

    ucl_score = np.percentile(t2ewmaI[offset+ext_len:], alarm_level)

    fp_score_mark = t2ewmaI[offset+ext_len:] > ucl_score
    (fp_score, rl_fp_score, len_after_detect,
     signal_ratio, flag_rl) = calEwmaStatisticsHelper(fp_score_mark)

    return (t2ewmaI[offset+ext_len:], start_PII, ucl_score, fp_score, rl_fp_score,
            len_after_detect, signal_ratio, flag_rl)


def calEwmaT2StatisticsPII(score_PII, mu_train, Sinv_train, gamma, start_PII, ucl_score):
    """Calculate statistics for Phase-II."""
    _, t2ewmaII = EwmaT2PII(score_PII, mu_train, Sinv_train, gamma, start_PII)

    fp_score_mark = t2ewmaII > ucl_score
    (fp_score, rl_fp_score, len_after_detect,
     signal_ratio, flag_rl) = calEwmaStatisticsHelper(fp_score_mark)
    return (t2ewmaII, fp_score, rl_fp_score, len_after_detect, signal_ratio, flag_rl)


def calEwmaStatisticsPI(offset, alarm_level, comp_PI, gamma, eff_wind_len_factor, start_PI):
    """Calculate statistics for Phase-I for calculation for Phase-II."""
    ext_len, start_PII, ewma_comp_PI = EwmaPI(comp_PI, gamma, eff_wind_len_factor, start_PI)

    ucl_comp = np.percentile(
        ewma_comp_PI[offset+ext_len:], (100.0 + alarm_level) / 2)
    lcl_comp = np.percentile(
        ewma_comp_PI[offset+ext_len:], (100.0 - alarm_level) / 2)
    fp_comp_mark = np.array([v>ucl_comp or v<lcl_comp for v in ewma_comp_PI[offset+ext_len:]])
    (fp_comp, rl_fp_comp, len_after_detect,
     signal_ratio, flag_rl) = calEwmaStatisticsHelper(fp_comp_mark)
    fp_comp_u_mark = np.array([v>ucl_comp for v in ewma_comp_PI[offset+ext_len:]])
    fp_comp_l_mark = np.array([v<lcl_comp for v in ewma_comp_PI[offset+ext_len:]])
    fp_comp_u, _, _, _, _ = calEwmaStatisticsHelper(fp_comp_u_mark)
    fp_comp_l, _, _, _, _ = calEwmaStatisticsHelper(fp_comp_l_mark)
    return (ewma_comp_PI[offset+ext_len:], start_PII, lcl_comp, ucl_comp,
            fp_comp, fp_comp_l, fp_comp_u, rl_fp_comp,
            len_after_detect, signal_ratio, flag_rl)


def calEwmaStatisticsPII(comp_PII, gamma, start_PII, lcl_comp, ucl_comp):
    """Calculate statistics for Phase-II."""
    ewma_comp_PII = EwmaPII(comp_PII, gamma, start_PII)

    fp_comp_mark = np.array([v>ucl_comp or v<lcl_comp for v in ewma_comp_PII])
    (fp_comp, rl_fp_comp, len_after_detect,
     signal_ratio, flag_rl) = calEwmaStatisticsHelper(fp_comp_mark)
    fp_comp_u_mark = np.array([v>ucl_comp for v in ewma_comp_PII])
    fp_comp_l_mark = np.array([v<lcl_comp for v in ewma_comp_PII])
    fp_comp_u, _, _, _, _ = calEwmaStatisticsHelper(fp_comp_u_mark)
    fp_comp_l, _, _, _, _ = calEwmaStatisticsHelper(fp_comp_l_mark)
    return (ewma_comp_PII, fp_comp, fp_comp_l, fp_comp_u, rl_fp_comp,
            len_after_detect, signal_ratio, flag_rl)



# Determine parameter of EWMA by setting some positive-signal-ratio.
def Find_EWMA_Gamma_Vec(
        metric_PI,
        metric_PII,
        alarm_level,
        gamma_ll,
        gamma_ul,
        nugget,
        positive_rate_cd,
        offset_PI,
        mu_train, Sinv_train,
        eff_wind_len_factor, start_PI,
        max_iter=40):
    """ Find EWMA gamma to have meet target positive_rate_cd for vector (score). """
    # # Need to calculate covariance matrix
    # # Phase I data:
    # N_PI = metric_PI.shape[0]
    # Sinv = InversedCov(metric_PI, nugget)  # nnet needs 0.15
    # mu = np.mean(metric_PI, axis=0)

    N_PI = metric_PI.shape[0]
    N_PII = metric_PII.shape[0]

    iter_cnt = 0
    while True:
        # Prevent ext_len longer than N_PI
        # gamma = max(1.0*eff_wind_len_factor/N_PI, 10.0**((np.log10(gamma_ll) + np.log10(gamma_ul)) / 2))
        gamma = max(1.0*eff_wind_len_factor/N_PI, ((gamma_ll + gamma_ul) / 2))
        # print gamma, eff_wind_len_factor, N_PI, ((gamma_ll + gamma_ul) / 2)
        # Phase I data:
        ext_len, start_PII, _, metric_t2_ewma = EwmaT2PI(metric_PI, mu_train,
                Sinv_train, gamma, eff_wind_len_factor, nugget, start_PI)

        # print t2ewmaI.shape, offset_PI, ext_len, gamma, eff_wind_len_factor/N_PI # debug
        # Control Chart
        # alarm_level = 99.99
        ucl = np.percentile(metric_t2_ewma[offset_PI+ext_len:], alarm_level)

        # Phase II data:
        (_, _, _, _, sig_ratio, _) = calEwmaT2StatisticsPII(metric_PII, mu_train, Sinv_train, gamma, start_PII, ucl)

        if sig_ratio <= positive_rate_cd:
            gamma_ul = gamma
        else:
            gamma_ll = gamma
        logger.info(("The iteration %s, upper control limit %s, "
                     "sig-ratio rate %s, gamma %s (%s, %s)"),
                     iter_cnt, ucl, sig_ratio, gamma, gamma_ll, gamma_ul, extra=d)
        iter_cnt += 1
        if iter_cnt >= max_iter or abs(sig_ratio - positive_rate_cd) / positive_rate_cd <= 0.001:
            break

    # print("From utility function--------\n")
    # print t2ewmaII

    # label_size = 16
    # # line_width = 1.5
    # # marker_size = 5
    # plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    # plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.5)
    # ax1 = plt.subplot(211)
    # ax1.set_xlabel('Phase-I Observation Index',size=label_size)
    # ax1.set_ylabel('MEWMA Multivariate\nControl Chart', color='k',size=label_size)
    # ax1.get_yaxis().set_label_coords(-0.08,0.5)
    # ax1.xaxis.set_tick_params(labelsize=label_size*0.5)
    # ax1.yaxis.set_tick_params(labelsize=label_size*0.5)
    #
    # ax1.plot(np.arange(len(t2ewmaI)), t2ewmaI)
    # ax1.plot(np.arange(len(t2ewmaI)), np.repeat(ucl, len(t2ewmaI)))
    # ax1.plot(np.arange(len(t2ewmaI)), np.repeat(0, len(t2ewmaI)), 'k', lw=0.5)
    # # tmp = np.min(t2ewmaI)
    # # for i in np.arange(time_stamp[0], time_stamp[idx_PII-1]+1, time_step):
    # #     j = np.argmax(time_stamp >= i)
    # #     ax1.text(j, ax1.get_ylim()[0]+0.1*(ax1.get_ylim()[1]-ax1.get_ylim()[0]), MonIdx2Date(time_stamp[j], date_index_flag),rotation=45)
    #
    # ax1 = plt.subplot(212)
    # ax1.set_xlabel('Phase-II Observation Index',size=label_size)
    # ax1.set_ylabel('MEWMA Multivariate\nControl Chart', color='k',size=label_size)
    # ax1.get_yaxis().set_label_coords(-0.08,0.5)
    # ax1.xaxis.set_tick_params(labelsize=label_size*0.5)
    # ax1.yaxis.set_tick_params(labelsize=label_size*0.5)
    #
    # # ax1.scatter(np.arange(len(t2ewmaII)), t2ewmaII, s=0.01)
    # ax1.plot(np.arange(len(t2ewmaII)), t2ewmaII)
    # ax1.plot(np.arange(len(t2ewmaII)), np.repeat(ucl, len(t2ewmaII)), 'k')
    # ax1.plot(np.arange(len(t2ewmaII)), np.repeat(0, len(t2ewmaII)), 'k', lw=0.5)
    # # tmp = np.min(t2ewmaII)
    # # for i in np.arange(time_stamp[idx_PII], max(time_stamp)+1, time_step):
    # #     j = np.argmax(time_stamp >= i)
    # #     ax1.text(j - idx_PII, ax1.get_ylim()[0]+0.1*(ax1.get_ylim()[1]-ax1.get_ylim()[0]), MonIdx2Date(time_stamp[j], date_index_flag),rotation=45)
    #
    # # if not os.path.exists(folder_name):
    # #     os.makedirs(folder_name)
    # # plt.savefig(folder_name + fig_name)
    # plt.show()
    # # plt.close()

    return gamma, gamma_ll, gamma_ul


def Find_EWMA_Gamma_Sca(
        metric_PI,
        metric_PII,
        alarm_level,
        gamma_ll,
        gamma_ul,
        positive_rate_cd,
        offset_PI,
        eff_wind_len_factor, start_PI,
        max_iter=40):
    """ Find EWMA gamma to have meet target positive_rate_cd for scalar (other metrics). """
    # No need to calculate covariance matrix
    N_PI = metric_PI.shape[0]
    N_PII = metric_PII.shape[0]

    iter_cnt = 0
    while True:
        # Prevent ext_len longer than N_PI
        # gamma = max(1.0*eff_wind_len_factor/N_PI, 10.0**((np.log10(gamma_ll) + np.log10(gamma_ul)) / 2))
        # print gamma, eff_wind_len_factor, N_PI, 10.0**((np.log10(gamma_ll) + np.log10(gamma_ul)) / 2)
        gamma = max(1.0*eff_wind_len_factor/N_PI, ((gamma_ll + gamma_ul) / 2))
        # print gamma, eff_wind_len_factor, N_PI, ((gamma_ll + gamma_ul) / 2)

        # Phase I data:
        ext_len, start_PII, metric_ewma = EwmaPI(metric_PI, gamma, eff_wind_len_factor, start_PI)

        # print metric_ewma.shape, offset_PI, ext_len, gamma, 1.0*eff_wind_len_factor/N_PI # debug
        # Control Chart
        # alarm_level = 99.99
        lcl = np.percentile(
            metric_ewma[offset_PI+ext_len:], (100.0 - alarm_level) / 2)
        ucl = np.percentile(
            metric_ewma[offset_PI+ext_len:], (100.0 + alarm_level) / 2)

        # Phase-II:
        (_, _, _, _, _, _, sig_ratio, _) = calEwmaStatisticsPII(metric_PII, gamma, start_PII, lcl, ucl)

        if sig_ratio <= positive_rate_cd:
            gamma_ul = gamma
        else:
            gamma_ll = gamma
        logger.info(("The iteration %s, lower control limit %s, upper control "
                     "limit %s, sig-ratio rate %s, gamma %s (%s, %s)"),
                     iter_cnt, lcl, ucl, sig_ratio, gamma, gamma_ll, gamma_ul, extra=d)
        # print iter_cnt, lcl, ucl, fp, gamma
        iter_cnt += 1
        if iter_cnt >= max_iter or abs(sig_ratio - positive_rate_cd) / positive_rate_cd <= 0.001:
            break
    # print("From utility function--------\n")
    # print metric_ewma, metric_ewmaII

    return gamma, gamma_ll, gamma_ul

def Save_EWMA_Data(ewma, tag, id_num, ch_idx, ga_idx, ch_f, sim_data_path):
    """Save all the important simulation parameters for record."""
    if not os.path.exists(sim_data_path):
        os.makedirs(sim_data_path)
    np.savetxt(os.path.join(sim_data_path,
                           '_'.join([str(id_num), str(ch_idx), str(ga_idx),
                                     str(ch_f).replace(".", "_"), tag]) + '.csv'),
                            ewma,
                            delimiter=',')


def Agg_Data(FLAGS):
    score_header_name = ['rnd_seed', 'ch_f_idx', 'ga_idx', 'ch_f', 'gamma', 'ucl', 'fpI', 'rlI', 'rlenI', 'srI', 'flagrlI', 'fpII', 'rlII', 'rlenII', 'srII', 'flagrlII']
    other_header_name = ['rnd_seed', 'ch_f_idx', 'ga_idx', 'ch_f', 'gamma', 'lcl', 'ucl', 'fpI', 'fplI', 'fpuI', 'rlI', 'rlenI', 'srI', 'flagrlI', 'fpII', 'fplII', 'fpuII', 'rlII', 'rlenII', 'srII', 'flagrlII']

    def agg_data_one_kind(tag, header_name, FLAGS):
        out_fname = '.'.join(['_'.join(['res',tag]),'csv'])
        out_folder = os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder),'agg_res')
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        out_file = os.path.join(out_folder, out_fname)
        # if os.path.isfile(out_file):
        #     logger.info("The previous data files are already processed.", extra=d)
        #     return
        with open(out_file, 'a') as out_f:
            header = True
            read_fname_ls = []
            for fname in os.listdir(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder)):
                if re.search(tag, fname):
                    # print fname
                    abs_fpath = os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), fname)
                    fdata = np.transpose(np.genfromtxt(abs_fpath, delimiter=','))
                    df_data = pd.DataFrame(data=fdata, columns=header_name)
                    df_data.to_csv(out_f, header=header, index=False)
                    read_fname_ls.append(fname)
                    if header:
                        # Only need to add header once
                        header = False

        for fname in read_fname_ls:
            abs_fpath = os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), fname)
            os.remove(abs_fpath) # Remove data after integration.

    FLAGS.tag_ls = FLAGS.tag_ls.split("-")
    header_name_ls = [other_header_name] * len(FLAGS.tag_ls)
    header_name_ls[0] = score_header_name

    agg_tasks = [(tag, header_name_ls[idx], FLAGS) for idx, tag in enumerate(FLAGS.tag_ls)]
    time_start = time.time()
    with parallel_backend('loky', n_jobs=FLAGS.num_processors):
        Parallel(verbose=100, pre_dispatch='2*n_jobs')(delayed(agg_data_one_kind)(*task) for task in agg_tasks)
    logger.info(("The time took to aggregate ewma statistics parallely"
                 "for default alarm rate for all tags is %s."),
                 time.time()-time_start, extra=d)

    # for idx, tag in enumerate(FLAGS.tag_ls):
    #     time_start = time.time()
    #     agg_data_one_kind(tag, header_name_ls[idx], FLAGS)
    #     logger.info(("The time took to aggregate ewma statistics"
    #                  "for default alarm rate for tag %s is %s."),
    #                  tag, time.time()-time_start, extra=d)


def Find_Ctrl_Lmt_Same_MRL0(raw_data_folder, raw_data_size, base_tag, comp_tag,
                            base_no_cd_stats, comp_no_cd_stats,
                            FLAGS, tol, iter_max):
    """ For each gamma, match the median in-control run-length of the comparing tag with
        that of the base tag, and determine the corresponding alarm level. Then, use
        that alarm level to go over the raw data and calculate the median
        out-of-control run-length and store the alarm level and run-length.
    """
    offset = OFFSET
    eff_wind_len_factor = FLAGS.eff_wind_len_factor
    sub_str = "0_0_1_ewma_" + comp_tag
    num_gammas = len(FLAGS.score_gamma_ls)
    num_change_factors = len(FLAGS.change_factor_ls)
    df_stat_same_mrl0 = pd.DataFrame(columns=[
        "gamma", "change_factor", "target_alarm_level",
        "median_run_len_PII", "median_signal_ratio_PII",
        "mean_run_len_PII", "mean_signal_ratio_PII"])
    for ga_idx in range(num_gammas-1):
        gamma = FLAGS.score_gamma_ls[ga_idx+1]
        al_lo, al_hi = 95.0, 100.0 # Alarm level is in 100-scale
        print((base_no_cd_stats["rlII"][ga_idx], comp_no_cd_stats["rlII"][ga_idx]))
        if base_no_cd_stats["rlII"][ga_idx] < comp_no_cd_stats["rlII"][ga_idx]:
            al_hi = FLAGS.alarm_level
        else:
            al_lo = FLAGS.alarm_level
        al_try = (al_lo + al_hi)/2
        # Calculate the Median Run Length until it matches that of
        # base statistics, like score
        iter_cnt = 0
        time_start_process = time.time()
        while True:
            logger.info("Binary search for gamma %s at iteration %s",
                FLAGS.score_gamma_ls[ga_idx+1], iter_cnt, extra=d)
            # Obtain EWMA and statistics in Phase-I
            runlen0_comp_PIIs = np.zeros(raw_data_size)
            sigratio0_comp_PIIs = np.zeros(raw_data_size)
            start_comp_PIIs = np.zeros(raw_data_size)
            lcl_comps = np.zeros(raw_data_size)
            ucl_comps = np.zeros(raw_data_size)
            for row_idx in range(raw_data_size):
                logger.info("Read file %s at iteration %s",
                    row_idx, iter_cnt, extra=d)
                # Obtain Phase-I EWMA and statistics
                PI_fname = "_".join([str(row_idx), sub_str, "PI.csv"])
                comp_PI = np.genfromtxt(os.path.join(raw_data_folder, PI_fname))
                (_, start_comp_PIIs[row_idx], lcl_comps[row_idx], ucl_comps[row_idx],
                 _, _, _, _, _, _, _) = calEwmaStatisticsPI(
                    offset, al_try, comp_PI, gamma, eff_wind_len_factor, np.mean(comp_PI))
                # Obtain Phase-II EWMA and statistics
                PII_fname = "_".join([str(row_idx), sub_str, "PII.csv"])
                comp_PII = np.genfromtxt(os.path.join(raw_data_folder, PII_fname))
                (_, _, _, _, runlen0_comp_PIIs[row_idx],
                 _, sigratio0_comp_PIIs[row_idx], _) = calEwmaStatisticsPII(
                    comp_PII, gamma, start_comp_PIIs[row_idx],
                    lcl_comps[row_idx], ucl_comps[row_idx])
            runlen0_comp_PII_median = np.median(runlen0_comp_PIIs)
            sigratio0_comp_PII_median = np.median(sigratio0_comp_PIIs)
            iter_cnt += 1
            if runlen0_comp_PII_median < base_no_cd_stats["rlII"][ga_idx]:
                al_lo = al_try
            else:
                al_hi = al_try
            al_try = (al_lo + al_hi)/2
            if np.abs(al_hi-al_lo) < tol or iter_cnt >= iter_max:
                targ_al = al_try # Targeted alarm level
                df_stat_same_mrl0.at[df_stat_same_mrl0.shape[0]] = [
                    gamma, 1, targ_al,
                    runlen0_comp_PII_median, sigratio0_comp_PII_median,
                    np.mean(runlen0_comp_PIIs), np.mean(sigratio0_comp_PIIs)]
                break
        logger.info(
            'Time spent in %s iteration with actual tolerance %s binary search is: %s.',
            iter_cnt, np.abs(al_hi-al_lo), time.time() - time_start_process,
            extra=d)

        # Obtain run length in out-of-control cases
        for ch_idx in range(num_change_factors - 1):
            ch_f = FLAGS.change_factor_ls[ch_idx + 1]
            ch_sub_str = "_".join([str(ch_idx+1), str(0),
                str(ch_f).replace(".", "_"), "ewma", comp_tag])
            runlen1_comp_PIIs = np.zeros(raw_data_size)
            sigratio1_comp_PIIs = np.zeros(raw_data_size)
            for row_idx in range(raw_data_size):
                # Obtain Phase-I EWMA and statistics
                # We don't need to recalculate start_comp_PII and control limits
                # PI_fname = "_".join([row_idx, ch_sub_str, "PI.csv"])
                # comp_PI = np.genfromtxt(os.path.join(raw_data_folder, PI_fname))
                # (_, start_comp_PII, lcl_comp, ucl_comp,
                #  _, _, _, _, _, _, _) = calEwmaStatisticsPI(
                #     offset, targ_al, comp_PI, gamma, np.mean(comp_PI))
                # Obtain Phase-II EWMA and statistics
                PII_fname = "_".join([str(row_idx), ch_sub_str, "PII.csv"])
                comp_PII = np.genfromtxt(os.path.join(raw_data_folder, PII_fname))
                (_, _, _, _, runlen1_comp_PIIs[row_idx],
                 _, sigratio1_comp_PIIs[row_idx], _) = calEwmaStatisticsPII(
                    comp_PII, gamma, start_comp_PIIs[row_idx],
                    lcl_comps[row_idx], ucl_comps[row_idx])
            # Integrate those results from the out-of-control cases into the df_stat_same_mrl0
            df_stat_same_mrl0.at[df_stat_same_mrl0.shape[0]] = [
                gamma, FLAGS.change_factor_ls[ch_idx+1], targ_al,
                np.median(runlen1_comp_PIIs),
                np.median(sigratio1_comp_PIIs),
                np.mean(runlen1_comp_PIIs),
                np.mean(sigratio1_comp_PIIs)]

    return df_stat_same_mrl0


def Agg_Save_EWMA_Sim_One_File(raw_data_folder, raw_data_size, tag, ph, ch_idx, ga_idx, ch_f, FLAGS):
    """Aggregate simulation files into one file."""
    logger.info("File processing for tag %s gamma index %s and change factor index %s.",
        tag, ga_idx, ch_idx, extra=d)
    agg_fname = '_'.join([tag, ph, 'ewma', str(ch_idx), str(ga_idx),
        str(ch_f).replace(".", "_")]) + '.csv'
    fname = '_'.join([str(0), str(ch_idx), str(ga_idx),
        str(ch_f).replace(".", "_"), 'ewma', tag, ph]) + '.csv'
    if not os.path.isfile(os.path.join(raw_data_folder, fname)):
        # No such ewma file anymore
        logger.info("No EWMA file %s for tag %s with gamma index %s and change factor index %s. ",
            fname, tag, ga_idx, ch_idx, extra=d)
        return
    read_file_ls = []
    if os.path.isfile(os.path.join(raw_data_folder, agg_fname)):
        logger.info("File processing for tag %s with gamma index %s and change factor index %s already existed.",
            fname, ga_idx, ch_idx, extra=d)
        return
    # read_file_ls = []
    # if os.path.isfile(os.path.join(raw_data_folder, agg_fname)):
    #     logger.info("File processing for tag %s with gamma index %s and change factor index %s already existed.",
    #         tag, ga_idx, ch_idx, extra=d)
    #     return
    time_start = time.time()
    with open(os.path.join(raw_data_folder, agg_fname), 'a') as out_f:
        for id_num in range(raw_data_size):
            ch_f = FLAGS.change_factor_ls[ch_idx]
            fname = '_'.join([str(id_num), str(ch_idx), str(ga_idx),
                str(ch_f).replace(".", "_"), 'ewma', tag, ph]) + '.csv'
            read_file_ls.append(fname)
            fpath = os.path.join(raw_data_folder, fname)
            fdata = np.reshape(np.genfromtxt(fpath, delimiter=','), (1,-1))
            df_data = pd.DataFrame(data=fdata, )
            df_data.to_csv(out_f, header=False, index=False)
    for fname in read_file_ls:
        os.remove(os.path.join(raw_data_folder, fname))
    logger.info("File processing for %s with gamma index %s and change factor index %s in Phase %s takes time %s.",
        tag, ga_idx, ch_idx, ph, time.time()-time_start, extra=d)


def Agg_Save_EWMA_Sim(raw_data_folder, raw_data_size, tag, FLAGS):
    """ Go over all .csv files and aggregate them in terms of change_idx,
        gamma_idx, and Phase (Phase-I and Phase-II).
    """
    num_gammas = len(FLAGS.score_gamma_ls)
    num_change_factors = len(FLAGS.change_factor_ls)
    Phase_ls = ["PI", "PII"]
    for ga_idx in range(num_gammas):
        for ch_idx in range(num_change_factors):
            ch_f = FLAGS.change_factor_ls[ch_idx]
            for ph in Phase_ls:
                Agg_Save_EWMA_Sim_One_File(raw_data_folder, raw_data_size, tag, ph, ch_idx, ga_idx, ch_f, FLAGS)


def Parallel_Agg_Save_EWMA_Sim(raw_data_folder, raw_data_size, tag_ls, n_jobs, FLAGS):
    """ Go over all .csv files and aggregate them in terms of change_idx,
        gamma_idx, and Phase (Phase-I and Phase-II).
    """
    num_gammas = len(FLAGS.score_gamma_ls)
    num_change_factors = len(FLAGS.change_factor_ls)
    Phase_ls = ["PI", "PII"]

    tasks = [(raw_data_folder, raw_data_size, tag, ph, ch_idx, ga_idx, FLAGS.change_factor_ls[ch_idx], FLAGS) for
        (tag, ph, ch_idx, ga_idx) in itertools.product(
            tag_ls, Phase_ls, list(range(num_change_factors)),
            list(range(num_gammas)))]
    with parallel_backend('loky', n_jobs=n_jobs):
        res = Parallel(verbose=100, pre_dispatch='2*n_jobs')(
            delayed(Agg_Save_EWMA_Sim_One_File)(*task) for task in tasks)


def Parallel_Find_Ctrl_Lmt_Same_RL0(raw_data_folder, raw_data_size, base_tag, tag_ls,
        func_ls, func_param_ls, func_name_ls,
        base_no_cd_stats, al_lo, al_hi, n_jobs,
        no_cd_stats_folder, output_file,
        FLAGS, tol, iter_max):
    """Parallelize the process of finding the control limits."""
    tasks = [(raw_data_folder, raw_data_size, base_tag, tag,
            func_ls, func_param_ls, func_name_ls,
            base_no_cd_stats, al_lo, al_hi,
            no_cd_stats_folder, output_file,
            FLAGS, tol, iter_max) for tag in tag_ls]
    with parallel_backend('loky', n_jobs=n_jobs):
        res = Parallel(verbose=100, pre_dispatch='2*n_jobs')(
            delayed(Find_Ctrl_Lmt_Same_RL0)(*task) for task in tasks)


def Find_Ctrl_Lmt_Same_RL0(raw_data_folder, raw_data_size, base_tag, tag,
                func_ls, func_param_ls, func_name_ls,
                base_no_cd_stats, al_lo, al_hi,
                no_cd_stats_folder, output_file,
                FLAGS, tol, iter_max):
    """ Find control limit to match the in control run length.
    """
    offset = OFFSET
    eff_wind_len_factor = FLAGS.eff_wind_len_factor
    num_gammas = len(FLAGS.score_gamma_ls)
    num_change_factors = len(FLAGS.change_factor_ls)
    df_stat_same_rl0 = pd.DataFrame(columns=[
        "gamma", "change_factor", "target_alarm_level"] + func_name_ls)

    def Cal_T2_CLmt_Run_Len(base_PI, base_PII, al_try, offset):
        ucl_score = np.percentile(base_PI[offset:], al_try)
        fp_score_mark = base_PII > ucl_score
        (_, runlen_T2_base_PII, _, sigratio_T2_base_PII, _) = calEwmaStatisticsHelper(
            fp_score_mark)
        return ucl_score, runlen_T2_base_PII, sigratio_T2_base_PII

    def Cal_Clmt_Run_Len0(comp_PI, comp_PII, al_try, gamma, eff_wind_len_factor, offset):
        # Obtain Phase-I EWMA and statistics
        (ewma_comp_PI, start_comp_PII, lcl_comp, ucl_comp,
         _, _, _, _, _, _, _) = calEwmaStatisticsPI(
            offset, al_try, comp_PI, gamma, eff_wind_len_factor, np.mean(comp_PI))
        # Obtain Phase-II EWMA and statistics
        (ewma_comp_PII, _, _, _, runlen_comp_PII,
         _, sigratio_comp_PII, _) = calEwmaStatisticsPII(
            comp_PII, gamma, start_comp_PII, lcl_comp, ucl_comp)
        return start_comp_PII, ewma_comp_PI, ewma_comp_PII, lcl_comp, ucl_comp, runlen_comp_PII, sigratio_comp_PII

    def Cal_Clmt_Run_Len1(comp_PII, start_comp_PII, lcl_comp, ucl_comp, gamma):
        # Obtain Phase-II EWMA and statistics
        (ewma_comp_PII, _, _, _, runlen_comp_PII,
         _, sigratio_comp_PII, _) = calEwmaStatisticsPII(
            comp_PII, gamma, start_comp_PII, lcl_comp, ucl_comp)
        return ewma_comp_PII, runlen_comp_PII, sigratio_comp_PII

    for ga_idx in range(1, num_gammas):
        gamma = FLAGS.score_gamma_ls[ga_idx]
        al_try = (al_lo + al_hi)/2
        iter_cnt = 0
        if tag == base_tag:
            # If the ewma is from score.
            ch_f0 = FLAGS.change_factor_ls[0]
            agg_fname_PI = '_'.join([tag, "PI", 'ewma', str(0),
                str(ga_idx), str(ch_f0).replace(".", "_")]) + '.csv'
            agg_fname_PII = '_'.join([tag, "PII", 'ewma', str(0),
                str(ga_idx), str(ch_f0).replace(".", "_")]) + '.csv'
            ewma_no_cd_PI = np.genfromtxt(
                os.path.join(raw_data_folder, agg_fname_PI), delimiter=',')
            ewma_no_cd_PII = np.genfromtxt(
                os.path.join(raw_data_folder, agg_fname_PII), delimiter=',')
            param = {"al_try":al_try, "offset":offset}
            time_start_process = time.time()
            while True:
                ewma_no_cd_stats = np.zeros((ewma_no_cd_PII.shape[0],3))
                for row_idx in range(ewma_no_cd_PII.shape[0]):
                    ewma_no_cd_stats[row_idx,:] = np.array(
                        Cal_T2_CLmt_Run_Len(ewma_no_cd_PI[row_idx,:],
                            ewma_no_cd_PII[row_idx,:], **param))
                ucl_base = np.reshape(ewma_no_cd_stats[:,0], (-1,))
                runlen0_T2_base_PII = np.reshape(ewma_no_cd_stats[:,1], (-1,))
                func_res_ls = [np.apply_along_axis(func, 0, runlen0_T2_base_PII, **func_param) for func, func_param in zip(func_ls, func_param_ls)]
                runlen0_T2_base_PII_stat = func_res_ls[0]
                logger.info("Binary search for tag %s with gamma %s with alarm rate %s (cur: %s, targ: %s) at iteration %s takes %s seconds.",
                    tag, gamma, al_try, runlen0_T2_base_PII_stat, base_no_cd_stats["rlII"][ga_idx-1],
                    iter_cnt, time.time() - time_start_process, extra=d)
                if runlen0_T2_base_PII_stat < base_no_cd_stats["rlII"][ga_idx]:
                    al_lo = al_try
                else:
                    al_hi = al_try
                al_try = (al_lo + al_hi)/2
                if np.abs(al_hi-al_lo) < tol or iter_cnt >= iter_max:
                    targ_al = al_try # Targeted alarm level
                    df_stat_same_rl0.at[df_stat_same_rl0.shape[0]] = [gamma, 1, targ_al] + func_res_ls
                    break
                iter_cnt += 1
            logger.info(
                'Time spent in %s iteration with actual tolerance %s binary search is: %s.',
                iter_cnt, np.abs(al_hi-al_lo), time.time() - time_start_process,
                extra=d)

            for ch_idx in range(1,num_change_factors):
                ch_f = FLAGS.change_factor_ls[ch_idx]
                agg_fname_PII = '_'.join([tag, "PII", 'ewma', str(ch_idx),
                    str(ga_idx), str(ch_f).replace(".", "_")]) + '.csv'
                ewma_PII = np.genfromtxt(
                    os.path.join(raw_data_folder, agg_fname_PII), delimiter=',')
                runlen1_T2_base_PII = np.zeros(ewma_PII.shape[0])
                for row_idx in range(ewma_PII.shape[0]):
                    fp_score_mark = ewma_PII[row_idx,:] > ucl_base[row_idx]
                    (_, runlen1_T2_base_PII[row_idx], _, sigratio1_T2_base_PII, _) = calEwmaStatisticsHelper(
                        fp_score_mark)
                func_res_ls = [np.apply_along_axis(func, 0, runlen1_T2_base_PII, **func_param) for func, func_param in zip(func_ls, func_param_ls)]
                df_stat_same_rl0.at[df_stat_same_rl0.shape[0]] = [gamma, FLAGS.change_factor_ls[ch_idx], targ_al] + func_res_ls
        else:
            # If ewma comes is from other metrics.
            ch_f0 = FLAGS.change_factor_ls[0]
            agg_fname_PI = '_'.join([tag, "PI", 'ewma', str(0),
                str(0), str(ch_f0).replace(".", "_")]) + '.csv'
            agg_fname_PII = '_'.join([tag, "PII", 'ewma', str(0),
                str(0), str(ch_f0).replace(".", "_")]) + '.csv'
            raw_no_cd_PI = np.genfromtxt(
                os.path.join(raw_data_folder, agg_fname_PI), delimiter=',')
            raw_no_cd_PII = np.genfromtxt(
                os.path.join(raw_data_folder, agg_fname_PII), delimiter=',')
            param = {"al_try":al_try, "gamma":gamma,
                     "eff_wind_len_factor":eff_wind_len_factor,
                     "offset":offset}
            time_start_process = time.time()
            while True:
                ewma_no_cd_stats = np.zeros((raw_no_cd_PII.shape[0],4))
                for row_idx in range(raw_no_cd_PII.shape[0]):
                    (ewma_no_cd_stats[row_idx,0], _, _, ewma_no_cd_stats[row_idx,1],
                     ewma_no_cd_stats[row_idx,2], ewma_no_cd_stats[row_idx,3],
                     _) = Cal_Clmt_Run_Len0(raw_no_cd_PI[row_idx,:], raw_no_cd_PII[row_idx,:], **param)
                start_comp_PII = np.reshape(ewma_no_cd_stats[:,0], (-1,))
                ucl_comp = np.reshape(ewma_no_cd_stats[:,1], (-1,))
                lcl_comp = np.reshape(ewma_no_cd_stats[:,2], (-1,))
                runlen0_comp_PII = np.reshape(ewma_no_cd_stats[:,3], (-1,))
                func_res_ls = [np.apply_along_axis(func, 0, runlen0_comp_PII, **func_param) for func, func_param in zip(func_ls, func_param_ls)]
                runlen0_comp_PII_stat = func_res_ls[0]
                logger.info("Binary search for tag %s with gamma %s with alarm rate %s (cur: %s, targ: %s) at iteration %s takes %s seconds.",
                    tag, gamma, al_try, runlen0_comp_PII_stat, base_no_cd_stats["rlII"][ga_idx],
                    iter_cnt, time.time() - time_start_process, extra=d)
                if runlen0_comp_PII_stat < base_no_cd_stats["rlII"][ga_idx]:
                    al_lo = al_try
                else:
                    al_hi = al_try
                al_try = (al_lo + al_hi)/2
                if np.abs(al_hi-al_lo) < tol or iter_cnt >= iter_max:
                    targ_al = al_try # Targeted alarm level
                    df_stat_same_rl0.at[df_stat_same_rl0.shape[0]] = [gamma, 1, targ_al] + func_res_ls
                    break
                iter_cnt += 1
            logger.info(
                'Time spent in %s iteration with actual tolerance %s binary search is: %s.',
                iter_cnt, np.abs(al_hi-al_lo), time.time() - time_start_process,
                extra=d)

            for ch_idx in range(1,num_change_factors):
                ch_f = FLAGS.change_factor_ls[ch_idx]
                agg_fname_PII = '_'.join([tag, "PII", 'ewma', str(ch_idx),
                    str(0), str(ch_f).replace(".", "_")]) + '.csv'
                raw_PII = np.genfromtxt(
                    os.path.join(raw_data_folder, agg_fname_PII), delimiter=',')
                runlen1_comp_PII = np.zeros(raw_PII.shape[0])
                for row_idx in range(raw_PII.shape[0]):
                    _, runlen1_comp_PII[row_idx], _ = Cal_Clmt_Run_Len1(
                        raw_PII[row_idx,:], start_comp_PII[row_idx],
                        lcl_comp[row_idx], ucl_comp[row_idx], gamma)
                func_res_ls = [np.apply_along_axis(func, 0, runlen1_comp_PII, **func_param) for func, func_param in zip(func_ls, func_param_ls)]
                df_stat_same_rl0.at[df_stat_same_rl0.shape[0]] = [gamma, ch_f, targ_al] + func_res_ls
    print(df_stat_same_rl0)
    out_file_path = os.path.join(no_cd_stats_folder, "_".join([tag, output_file]))
    # if not os.path.isfile(out_file_path):
    df_stat_same_rl0.to_csv(out_file_path)
    return df_stat_same_rl0


def Parallel_Read_Find_Save_Ctrl_Lmt_Same_RL0(raw_data_folder, raw_data_size, base_tag, tag_ls,
                                    func_ls, func_param_ls, func_name_ls,
                                    base_no_cd_stats, al_lo_0, al_hi_0, n_jobs,
                                    no_cd_stats_folder, output_file,
                                    FLAGS, tol, iter_max):
    """ For each gamma, match the median in-control run-length of the base and comparing tag with
        a given value, and determine the corresponding alarm level. Then, use
        that alarm level to go over the raw data and calculate the median
        out-of-control run-length and store the alarm level and run-length.
    """
    offset = OFFSET
    eff_wind_len_factor = FLAGS.eff_wind_len_factor
    num_gammas = len(FLAGS.score_gamma_ls)
    num_change_factors = len(FLAGS.change_factor_ls)

    for tag in tag_ls:
        df_stat_same_rl0 = pd.DataFrame(columns=[
            "gamma", "change_factor", "target_alarm_level"] + func_name_ls)
        tag_fnames = []
        time_tag_process = time.time()
        for ga_idx in range(1, num_gammas):
            # Test if this combination of tag, gamma exists or not.
            logger.info("The combination is (%s, %s)", tag, ga_idx, extra=d)
            row_idx0 = 0
            ch_f = FLAGS.change_factor_ls[0]
            if tag == base_tag:
                PI_fname = "_".join([str(row_idx0), str(0), str(ga_idx),
                    str(ch_f).replace(".", "_"), "ewma", tag, "PI.csv"])
            else:
                PI_fname = "_".join([str(row_idx0), str(0), str(0),
                    str(ch_f).replace(".", "_"), "ewma", tag, "PI.csv"])
            if not os.path.isfile(os.path.join(raw_data_folder, PI_fname)):
                logger.info("The combination (%s, %s) has been processed (%s)", tag, ga_idx, PI_fname, extra=d)
                continue
            # If this combination of tag, gamma exisits, go on processing.
            gamma = FLAGS.score_gamma_ls[ga_idx]
            # al_lo, al_hi = 95, 100 # Alarm level is in 100-scale
            al_lo, al_hi = al_lo_0, al_hi_0
            al_try = (al_lo + al_hi)/2
            # Calculate the Median Run Length until it matches that of
            # base statistics, like score
            iter_cnt = 0
            time_start_process = time.time()
            while True:
                # Obtain EWMA and statistics in Phase-I
                # Calculate in-control median run-length for the trial alarm level
                (start_PIIs, lcls, ucls,
                 runlen0_PIIs, sigratio0_PIIs, PI_fnames, PII_fnames) = Parallel_Cal_Run_Len0(
                    raw_data_size, raw_data_folder, base_tag, tag, offset, al_try, ga_idx,
                    eff_wind_len_factor, n_jobs, FLAGS)
                logger.info("The return results of run length of tag %s is %s.", tag, runlen0_PIIs, extra=d)

                # sigratio0_PII_median = np.median(sigratio0_PIIs)
                func_res_ls = [np.apply_along_axis(func, 0, runlen0_PIIs, **func_param) for func, func_param in zip(func_ls, func_param_ls)]
                runlen0_PII_stat = func_res_ls[0]

                logger.info("Binary search for tag %s with gamma %s with alarm rate %s (cur: %s, targ: %s) at iteration %s takes %s seconds.",
                    tag, gamma, al_try,
                    runlen0_PII_stat, base_no_cd_stats["rlII"][ga_idx],
                    iter_cnt, time.time() - time_start_process, extra=d)
                iter_cnt += 1
                if runlen0_PII_stat < base_no_cd_stats["rlII"][ga_idx]:
                    al_lo = al_try
                else:
                    al_hi = al_try
                al_try = (al_lo + al_hi)/2
                if np.abs(al_hi-al_lo) < tol or iter_cnt >= iter_max:
                    targ_al = al_try # Targeted alarm level
                    tag_fnames.extend(PI_fnames)
                    tag_fnames.extend(PII_fnames)
                    df_stat_same_rl0.at[df_stat_same_rl0.shape[0]] = [gamma, 1, targ_al] + func_res_ls
                    np.savetxt(os.path.join(no_cd_stats_folder, '_'.join([tag, str(0), str(ga_idx), "run_len"])+'.csv'),
                               runlen0_PIIs, delimiter=',')
                    break
            logger.info(
                'Time spent in %s iteration with actual tolerance %s binary search for tag %s is: %s.',
                iter_cnt, np.abs(al_hi-al_lo), tag, time.time() - time_start_process,
                extra=d)

            # Obtain run length in out-of-control cases
            for ch_idx in range(1, num_change_factors):
                runlen1_PIIs, sigratio1_PIIs, PII_fnames = Parallel_Cal_Run_Len1(raw_data_size, raw_data_folder,
                                base_tag, tag, start_PIIs, lcls, ucls, ch_idx, ga_idx, n_jobs, FLAGS)
                tag_fnames.extend(PII_fnames)
                func_res_ls = [np.apply_along_axis(func, 0, runlen1_PIIs, **func_param) for func, func_param in zip(func_ls, func_param_ls)]
                # Integrate those results from the out-of-control cases into the df_stat_same_mrl0
                df_stat_same_rl0.at[df_stat_same_rl0.shape[0]] = [gamma, FLAGS.change_factor_ls[ch_idx], targ_al] + func_res_ls
                np.savetxt(os.path.join(no_cd_stats_folder,
                                '_'.join([tag, str(ch_idx), str(ga_idx), "run_len"])+'.csv'),
                           runlen1_PIIs, delimiter=',')

        logger.info('Time spent in binary search for tag %s with different gamma\'s is: %s.',
            tag, time.time() - time_tag_process, extra=d)
        logger.info("The statistics after matching in-control run length related to tag %s is %s.", tag, df_stat_same_rl0, extra=d)
        out_file_path = os.path.join(no_cd_stats_folder, "_".join([tag, output_file]))
        # if not os.path.isfile(out_file_path):
        df_stat_same_rl0.to_csv(out_file_path)

        for fname in tag_fnames:
            # To save space, remove those ewma simulation pathes.
            abs_fpath = os.path.join(raw_data_folder, fname)
            if os.path.isfile(abs_fpath):
                os.remove(abs_fpath)


# def Parallel_Find_Ctrl_Lmt_Same_MRL0(raw_data_folder, raw_data_size, base_tag, comp_tag,
#                             base_no_cd_stats, comp_no_cd_stats, n_jobs,
#                             FLAGS, tol, iter_max):
#     """ For each gamma, match the median in-control run-length of the comparing tag with
#         that of the base tag, and determine the corresponding alarm level. Then, use
#         that alarm level to go over the raw data and calculate the median
#         out-of-control run-length and store the alarm level and run-length.
#
#         Deprecated: Previously, we need to choose match the run-length of
#         comparing tag with that of base tag (score). The run-length of base tag
#         is given based on alarm level, so that different change factors will give
#         different run-length. However, after discussion, we decide to fix run-length
#         for the base tag, so that no matter what change factor it is, the
#         run-length should be the same. So I also need to match in-control run-length
#         of base tag. This function is designed only to match in-control run-length
#         of comparing tag, not base tag. So this function is deprecated.
#     """
#     offset = OFFSET
#     eff_wind_len_factor = FLAGS.eff_wind_len_factor
#     sub_str = "0_0_1_ewma_" + comp_tag
#     num_gammas = len(FLAGS.score_gamma_ls)
#     num_change_factors = len(FLAGS.change_factor_ls)
#     df_stat_same_mrl0 = pd.DataFrame(columns=[
#         "gamma", "change_factor", "target_alarm_level",
#         "median_run_len_PII", "median_signal_ratio_PII",
#         "mean_run_len_PII", "mean_signal_ratio_PII"])
#
#     for ga_idx in xrange(num_gammas-1):
#         gamma = FLAGS.score_gamma_ls[ga_idx+1]
#         al_lo, al_hi = 95, 100 # Alarm level is in 100-scale
#         print base_no_cd_stats["rlII"][ga_idx], comp_no_cd_stats["rlII"][ga_idx]
#         if base_no_cd_stats["rlII"][ga_idx] < comp_no_cd_stats["rlII"][ga_idx]:
#             al_hi = FLAGS.alarm_level
#         else:
#             al_lo = FLAGS.alarm_level
#         al_try = (al_lo + al_hi)/2
#         # Calculate the Median Run Length until it matches that of
#         # base statistics, like score
#         iter_cnt = 0
#         time_start_process = time.time()
#         while True:
#             # Obtain EWMA and statistics in Phase-I
#             res_array = Parallel_Cal_Run_Len0(
#                 raw_data_size, raw_data_folder, sub_str, offset, al_try, gamma,
#                 eff_wind_len_factor, n_jobs, FLAGS)
#
#             start_comp_PIIs = res_array[0,:]
#             lcl_comps = res_array[1,:]
#             ucl_comps = res_array[2,:]
#             runlen0_comp_PIIs = res_array[3,:]
#             sigratio0_comp_PIIs = res_array[4,:]
#
#             runlen0_comp_PII_median = np.median(runlen0_comp_PIIs)
#             sigratio0_comp_PII_median = np.median(sigratio0_comp_PIIs)
#             logger.info("Binary search for gamma %s at iteration %s takes %s seconds.",
#                 FLAGS.score_gamma_ls[ga_idx+1], iter_cnt,
#                 time.time() - time_start_process, extra=d)
#             iter_cnt += 1
#             if runlen0_comp_PII_median < base_no_cd_stats["rlII"][ga_idx]:
#                 al_lo = al_try
#             else:
#                 al_hi = al_try
#             al_try = (al_lo + al_hi)/2
#             if np.abs(al_hi-al_lo) < tol or iter_cnt >= iter_max:
#                 targ_al = al_try # Targeted alarm level
#                 df_stat_same_mrl0.at[df_stat_same_mrl0.shape[0]] = [
#                     gamma, 1, targ_al,
#                     runlen0_comp_PII_median, sigratio0_comp_PII_median,
#                     np.mean(runlen0_comp_PIIs), np.mean(sigratio0_comp_PIIs)]
#                 break
#         logger.info(
#             'Time spent in %s iteration with actual tolerance %s binary search is: %s.',
#             iter_cnt, np.abs(al_hi-al_lo), time.time() - time_start_process,
#             extra=d)
#
#         # Obtain run length in out-of-control cases
#         for ch_idx in xrange(num_change_factors - 1):
#             ch_f = FLAGS.change_factor_ls[ch_idx + 1]
#             ch_sub_str = "_".join([str(ch_idx+1), str(0),
#                 str(ch_f).replace(".", "_"), "ewma", comp_tag])
#             res_array = Parallel_Cal_Run_Len1(raw_data_size, raw_data_folder, ch_sub_str,
#                                       start_comp_PIIs, lcl_comps, ucl_comps, gamma, n_jobs, FLAGS)
#
#             runlen1_comp_PIIs = res_array[0,:]
#             sigratio1_comp_PIIs = res_array[1,:]
#             # Integrate those results from the out-of-control cases into the df_stat_same_mrl0
#             df_stat_same_mrl0.at[df_stat_same_mrl0.shape[0]] = [
#                 gamma, FLAGS.change_factor_ls[ch_idx+1], targ_al,
#                 np.median(runlen1_comp_PIIs),
#                 np.median(sigratio1_comp_PIIs),
#                 np.mean(runlen1_comp_PIIs),
#                 np.mean(sigratio1_comp_PIIs)]
#
#     return df_stat_same_mrl0


def Check_Monotonic(np_arr):
    return (all(np_arr[i] <= np_arr[i + 1] for i in range(len(np_arr) - 1)) or
            all(np_arr[i] >= np_arr[i + 1] for i in range(len(np_arr) - 1)))


def Parallel_Cal_Run_Len0(raw_data_size, raw_data_folder, base_tag, tag,
                          offset, al_try, ga_idx, eff_wind_len_factor, n_jobs, FLAGS):
    temp_folder_name = os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'temp_folder')

    def Cal_T2_Run_Len(row_idx, raw_data_folder, tag, offset, al_try, ga_idx, eff_wind_len_factor):
        ch_f = FLAGS.change_factor_ls[0]
        sub_str = "_".join([str(0), str(ga_idx),
            str(ch_f).replace(".", "_"), "ewma", tag])
        logger.info("Read file %s", row_idx, extra=d)
        # Obtain Phase-I EWMA and statistics
        PI_fname = "_".join([str(row_idx), sub_str, "PI.csv"])
        base_PI = np.genfromtxt(os.path.join(raw_data_folder, PI_fname))
        ucl_base = np.percentile(base_PI, al_try)
        # Obtain Phase-II EWMA and statistics
        PII_fname = "_".join([str(row_idx), sub_str, "PII.csv"])
        base_PII = np.genfromtxt(os.path.join(raw_data_folder, PII_fname))
        fp_score_mark = base_PII > ucl_base
        (_, runlen0_base_PII, _,
         sigratio0_base_PII, _) = calEwmaStatisticsHelper(fp_score_mark)
        return row_idx, ucl_base, runlen0_base_PII, sigratio0_base_PII, PI_fname, PII_fname

    def Cal_Run_Len(row_idx, raw_data_folder, tag, offset, al_try, ga_idx, eff_wind_len_factor):
        gamma = FLAGS.score_gamma_ls[ga_idx]
        ch_f = FLAGS.change_factor_ls[0]
        sub_str = "_".join([str(0), str(0),
            str(ch_f).replace(".", "_"), "ewma", tag])
        logger.info("Read file %s", row_idx, extra=d)
        # Obtain Phase-I EWMA and statistics
        PI_fname = "_".join([str(row_idx), sub_str, "PI.csv"])
        comp_PI = np.genfromtxt(os.path.join(raw_data_folder, PI_fname))
        (_, start_comp_PII, lcl_comp, ucl_comp,
         _, _, _, _, _, _, _) = calEwmaStatisticsPI(
            offset, al_try, comp_PI, gamma, eff_wind_len_factor, np.mean(comp_PI))
        # Obtain Phase-II EWMA and statistics
        PII_fname = "_".join([str(row_idx), sub_str, "PII.csv"])
        comp_PII = np.genfromtxt(os.path.join(raw_data_folder, PII_fname))
        (_, _, _, _, runlen0_comp_PII,
         _, sigratio0_comp_PII, _) = calEwmaStatisticsPII(
            comp_PII, gamma, start_comp_PII,
            lcl_comp, ucl_comp)
        return row_idx, start_comp_PII, lcl_comp, ucl_comp, runlen0_comp_PII, sigratio0_comp_PII, PI_fname, PII_fname

    tasks = [(row_idx, raw_data_folder, tag, offset, al_try, ga_idx, eff_wind_len_factor) for row_idx in range(raw_data_size)]
    if tag == base_tag:
        with parallel_backend('loky', n_jobs=n_jobs):
            res = Parallel(verbose=100, temp_folder=temp_folder_name, pre_dispatch='2*n_jobs')(delayed(Cal_T2_Run_Len)(*task) for task in tasks)
    else:
        with parallel_backend('loky', n_jobs=n_jobs):
            res = Parallel(verbose=100, temp_folder=temp_folder_name, pre_dispatch='2*n_jobs')(delayed(Cal_Run_Len)(*task) for task in tasks)
    res_trans = (list(zip(*res)))
    row_order = np.array(res_trans[0])
    if not Check_Monotonic(row_order):
        logger.info("The in-control run length of tag %s from parallel computing is not in row order.", tag, extra=d)

    if tag == base_tag:
        start_PIIs = None
        lcls = None
        ucls = np.array(res_trans[1])
        runlen0_PIIs = np.array(res_trans[2])
        sigratio0_PIIs = np.array(res_trans[3])
        PI_fnames = list(res_trans[4])
        PII_fnames = list(res_trans[5])
    else:
        start_PIIs = np.array(res_trans[1])
        lcls = np.array(res_trans[2])
        ucls = np.array(res_trans[3])
        runlen0_PIIs = np.array(res_trans[4])
        sigratio0_PIIs = np.array(res_trans[5])
        PI_fnames = list(res_trans[6])
        PII_fnames = list(res_trans[7])

    return start_PIIs, lcls, ucls, runlen0_PIIs, sigratio0_PIIs, PI_fnames, PII_fnames


def Parallel_Cal_Run_Len1(raw_data_size, raw_data_folder, base_tag, tag,
                          start_PIIs, lcls, ucls, ch_idx, ga_idx, n_jobs, FLAGS):
    temp_folder_name = os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'temp_folder')

    def Cal_T2_Run_Len(row_idx, raw_data_folder, tag, ch_idx, ga_idx, ucl_base):
        gamma = FLAGS.score_gamma_ls[ga_idx]
        ch_f = FLAGS.change_factor_ls[ch_idx]
        ch_sub_str = "_".join([str(ch_idx), str(ga_idx),
            str(ch_f).replace(".", "_"), "ewma", tag])
        logger.info("Read file %s", row_idx, extra=d)
        PII_fname = "_".join([str(row_idx), ch_sub_str, "PII.csv"])
        base_PII = np.genfromtxt(os.path.join(raw_data_folder, PII_fname))
        fp_score_mark = base_PII > ucl_base
        (_, runlen1_base_PII, _,
         sigratio1_base_PII, _) = calEwmaStatisticsHelper(fp_score_mark)
        return row_idx, runlen1_base_PII, sigratio1_base_PII, PII_fname

    def Cal_Run_Len(row_idx, raw_data_folder, tag, ch_idx, ga_idx, start_comp_PII, lcl_comp, ucl_comp):
        gamma = FLAGS.score_gamma_ls[ga_idx]
        ch_f = FLAGS.change_factor_ls[ch_idx]
        ch_sub_str = "_".join([str(ch_idx), str(0),
            str(ch_f).replace(".", "_"), "ewma", tag])
        logger.info("Read file %s", row_idx, extra=d)
        PII_fname = "_".join([str(row_idx), ch_sub_str, "PII.csv"])
        comp_PII = np.genfromtxt(os.path.join(raw_data_folder, PII_fname))
        (_, _, _, _, runlen1_comp_PII,
         _, sigratio1_comp_PII, _) = calEwmaStatisticsPII(
            comp_PII, gamma, start_comp_PII, lcl_comp, ucl_comp)
        return row_idx, runlen1_comp_PII, sigratio1_comp_PII, PII_fname

    if tag == base_tag:
        tasks = [(row_idx, raw_data_folder, tag, ch_idx, ga_idx,
                  ucls[row_idx]) for row_idx in range(raw_data_size)]
        with parallel_backend('loky', n_jobs=n_jobs):
            res = Parallel(verbose=100, temp_folder=temp_folder_name, pre_dispatch='2*n_jobs')(delayed(Cal_T2_Run_Len)(*task) for task in tasks)
    else:
        tasks = [(row_idx, raw_data_folder, tag, ch_idx, ga_idx,
                  start_PIIs[row_idx], lcls[row_idx], ucls[row_idx]) for row_idx in range(raw_data_size)]
        with parallel_backend('loky', n_jobs=n_jobs):
            res = Parallel(verbose=100, temp_folder=temp_folder_name, pre_dispatch='2*n_jobs')(delayed(Cal_Run_Len)(*task) for task in tasks)
    res_trans = (list(zip(*res)))
    row_order = np.array(res_trans[0])
    if not Check_Monotonic(row_order):
        logger.info("The out-control run length of tag %s from parallel computing is not in row order.", tag, extra=d)

    runlen1_PIIs = np.array(res_trans[1])
    sigratio1_PIIs = np.array(res_trans[2])
    PII_fnames = list(res_trans[3])

    return runlen1_PIIs, sigratio1_PIIs, PII_fnames


def Visual_Score_PCA(score_PI, score_PII, N_PIIs, folder_name, dim=2):
    """Calculate and plot PCA for scores in PI and PII."""
    # Plot the trace of singular values.
    _, sing_val, vh = np.linalg.svd(score_PI, full_matrices=False)
    print("--------------Singular Value-----------\n")
    print(sing_val)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Number of Singular Values', fontsize=15)
    ax.set_ylabel('Singular Values', fontsize=15)
    ax.set_title("Singular Value Trace Plots", fontsize=20)
    ax.plot(np.arange(sing_val.shape[0]), sing_val)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(folder_name + "sing_val.png")
    plt.close()

    pca_PI = PCA(n_components=dim)
    pca_score_PI = pca_PI.fit_transform(score_PI)
    print("\n PCA explained Variance Ratio in PI:\n")
    print((pca_PI.explained_variance_ratio_))
    # pca_PII = PCA(n_components=dim)
    pca_score_PII = np.dot(score_PII, vh[:2, ].T)
    # print("\n PCA explained Variance Ratio in PII:\n")
    # print(pca_PII.explained_variance_ratio_)
    # Plot PCA in 2D
    if dim == 2:
        title = "PI_score_pca"
        Plot_2D_PCA(pca_score_PI, [], title, folder_name, title + ".png")
        title = "PII_score_pca"
        Plot_2D_PCA(pca_score_PII, N_PIIs, title, folder_name, title + ".png")


def Plot_2D_PCA(pca_scores, N_PIIs, title, folder_name, fig_name):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title(title, fontsize=20)

    if len(N_PIIs) > 0:
        temp_N_PIIs = np.cumsum([0] + N_PIIs)
        colors = ['r', 'g', 'b', 'k', 'y']
        for idx in range(len(N_PIIs)):
            ax.scatter(pca_scores[temp_N_PIIs[idx]:temp_N_PIIs[idx + 1], 0],
                       pca_scores[temp_N_PIIs[idx]:temp_N_PIIs[idx + 1], 1],
                       c=colors[idx], s=0.2)
    else:
        ax.scatter(pca_scores[:, 0], pca_scores[:, 1], s=0.2)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(folder_name + fig_name)
    plt.close()
