import tensorflow as tf
# import tensorflow.contrib.eager as tf
import numpy as np
import pandas as pd
import math
import logging
import argparse
import matplotlib.pyplot as plt
# plt.switch_backend('agg') # Needed for running on quest
import pickle
import sys
import os
import re
import time
import copy
import statsmodels.api as sm
from joblib import delayed, Parallel, parallel_backend

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, r2_score

from control_chart.utils import *
from control_chart.hotelling import *

# Cross-validation:
# 1 factors 8 combinations, 5 replication, 10 folds, 24 cores, max_steps=50000. Took 40 mins.
# 3 factors 72 combinations, 3 replication, 5 folds, 24 cores, max_steps=5000. Took 30 mins.

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s(%(funcName)s)[%(lineno)d]: %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'zkg'}
logger = logging.getLogger('regressors')
logging.getLogger('regressors').setLevel(logging.INFO)

# class Batch(object):
#     def __init__(self, X, y, batch_size):
#         self.batch_size = batch_size
#         self.X = X
#         self.y = y
#         self.size = X.shape[0]
#
#     def getBatch(self):
#         indices = np.random.choice(range(self.size), self.batch_size)
#         return (tf.convert_to_tensor(
#             self.X[indices, :], dtype=tf.float32),
#             tf.convert_to_tensor(self.y[indices]))
#
#
# def Build_Model(hidden_layer_sizes, input_dim, output_dim):
#     """Build a neural network model."""
#     model = tf.keras.Sequential()
#
#     if not hidden_layer_sizes:
#         model.add(tf.keras.layers.Dense(output_dim, input_dim=input_dim))
#     else:
#         for idx, num_hidnode in enumerate(hidden_layer_sizes):
#             if idx == 0:
#                 model.add(
#                     tf.keras.layers.Dense(
#                         num_hidnode,
#                         activation="relu",
#                         input_dim=input_dim))
#             else:
#                 model.add(
#                     tf.keras.layers.Dense(
#                         num_hidnode,
#                         activation="relu"))
#         # The output has no activation because later we will add
#         # the logistic or softmax function on the output layer.
#         model.add(tf.keras.layers.Dense(output_dim))
#
#     return model


class Nnet_Reg(object):
    def __init__(self, X_train, y_train, X_val, y_val, hidden_layer_sizes, penal_type, penal_param, stopping_lag, training_batch_size,
                 learning_rate, train_PI_flag, model_time_stamp, FLAGS, normal_flag=True, cv_tasks_info=None, plot_trace_flag=False):
        """ Train a nnet model using those training and validation data sets. """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.penal_type = penal_type
        self.penal_param = penal_param
        self.stopping_lag = stopping_lag
        self.training_batch_size = training_batch_size
        self.learning_rate = learning_rate
        self.train_PI_flag = train_PI_flag
        self.model_time_stamp = model_time_stamp
        self.FLAGS = FLAGS
        self.normal_flag = normal_flag
        self.cv_tasks_info = cv_tasks_info
        self.plot_trace_flag = plot_trace_flag

        # # Define those function in __init__ so that they can be past into joblib function for parallelization.
        # # Inside those functions, call other functions directly without adding 'self.' prior to that.
        # def loss_reg(inputs, targets, model, wei=None):
        #     predictions = model(inputs)
        #     return tf.reduce_mean(tf.losses.mean_squared_error(
        #         y_true=targets, y_pred=predictions))

        # def loss_pois(inputs, targets, model, wei=None):
        #     log_inputs = model(inputs)
        #     return tf.reduce_mean(tf.nn.log_poisson_loss(
        #         targets=targets, log_input=log_inputs, compute_full_loss=True))

        # # Use the self-written loss function as below is ~5x slower than built-in log_poisson_loss. 20190602.
        # # def loss_pois(inputs, targets, model):
        # #     log_inputs = model(inputs)
        # #     const = tf.reduce_mean(np.sum([np.sum(np.log(np.arange(1, v+1))) for v in targets], dtype=np.float32))
        # #     # logger.info("The target and log_factorial are %s, %s.", targets, [np.sum(np.log(np.arange(1, v+1))) for v in targets], extra=d)
        # #     return tf.reduce_mean(-targets*log_inputs+tf.exp(log_inputs)) + const

        # def loss_grad(inputs, targets, model, loss, wei=None):
        #     with tf.GradientTape() as tape:
        #         loss_value = loss(inputs, targets, model)
        #     return tape.gradient(loss_value, model.variables)

        # def pred_reg(inputs, model):
        #     return np.reshape(model(inputs), (-1,))

        # def pred_pois(inputs, model):
        #     return np.exp(np.reshape(model(inputs), (-1,)))

        # def residual_reg(inputs, targets, model):
        #     predictions = pred_reg(inputs, model)
        #     # print "Predictions from the model."
        #     # print predictions
        #     targets, predictions = targets.reshape((-1,)), predictions.reshape((-1,))
        #     return targets-predictions, predictions

        # def residual_pois(inputs, targets, model):
        #     predictions = pred_pois(inputs, model)
        #     # print "Predictions from the model."
        #     # print predictions
        #     targets, predictions = targets.reshape((-1,)), predictions.reshape((-1,))
        #     return targets-predictions, predictions

        # def obj_func(inputs, targets, model, loss, penal_param, wei=None):
        #     if self.penal_type.lower() == 'none':
        #         obj = loss(inputs, targets, model)
        #     else:
        #         if self.penal_type.lower() == 'l2':
        #             # Only penalize weights not biases. Has tested that the
        #             # gradient has penalization part.
        #             obj = loss(inputs, targets, model) + 0.5 * penal_param * tf.reduce_sum(
        #                 [tf.nn.l2_loss(var) for var in model.variables if re.findall(r'kernel', var.name)])
        #         if self.penal_type.lower() == 'l1':
        #             obj = loss(inputs, targets, model) + penal_param * tf.reduce_sum(
        #                 [tf.reduce_sum(tf.abs(var)) for var in model.variables if re.findall(r'kernel', var.name)])
        #     return obj

        # def obj_grad(inputs, targets, model, loss, penal_param, wei=None):
        #     with tf.GradientTape() as tape:
        #         obj_value = obj_func(inputs, targets, model, loss, penal_param)
        #     return tape.gradient(obj_value, model.variables)

        # def plot_trace(trace_train, trace_val, best_index, pos, name, prefix='reg'):
        #     fig, ax = plt.subplots()
        #     print((len(trace_train), self.training_batch_size))
        #     ax.plot(
        #         np.arange(len(trace_train)) * self.training_batch_size,
        #         trace_train, 'b-', label='Training ' + name)
        #     ax.plot(
        #         np.arange(len(trace_val)) * self.training_batch_size,
        #         trace_val, 'r-', label='Validation ' + name)
        #     ax.axvline(x=best_index * self.training_batch_size)
        #     legend = ax.legend(loc=pos, shadow=False, fontsize='x-large', fancybox=True, facecolor='white', framealpha=0.5)
        #     # Put a nicer background color on the legend.
        #     # legend.get_frame().set_facecolor('#00FFCC')
        #     plt.savefig(
        #         os.path.join(self.FLAGS.training_res_folder,
        #         '_'.join([prefix, str(self.penal_param).replace(".", "_"), name])+'.png'))
        #     plt.show()
        #     plt.close()

        # def append_logging_to_file(dict_logging_files, model_ckpt_fname, log_path, purge_flag=True):
        #     for name, log_ls in dict_logging_files.items():
        #         log_fname = name+'_'+model_ckpt_fname.split('.')[0]+'.txt'
        #         with open(os.path.join(log_path, log_fname), "a+") as log_f:
        #             log_f.write('\n'.join(["{0:.3f}".format(ele,) for ele in log_ls]))
        #         if purge_flag:
        #             dict_logging_files[name]=[]

        # def Train_Nnet_Reg(gen_model_func, gen_model_func_param, initial_weights_file,
        #                 loss, pred, X, y, wei, train_idx_ls, val_idx_ls,
        #                 penal_param, stopping_lag, training_batch_size,
        #                 learning_rate, model_ckpt_fname, FLAGS, log_folder, plot_trace_flag=False):
        #     """ Train neural network using the training and validation datasets. """
        #     # Create a new model
        #     # Looks like I cannot pass model as parameter to this function. Otherwise, it cannot use joblib to parallelize jobs.
        #     model, _ = gen_model_func(*gen_model_func_param)
        #     model.load_weights(os.path.join(FLAGS.training_res_folder, initial_weights_file))
        #     logger.info("The new model has weights %s.", model.variables, extra=d)

        #     X_train, y_train = X[train_idx_ls,:], y[train_idx_ls]
        #     X_val, y_val = X[val_idx_ls,:], y[val_idx_ls]

        #     optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        #     batch = Batch(X_train, y_train, training_batch_size, wei=wei)

        #     log_path = os.path.join(FLAGS.training_res_folder, log_folder)
        #     if not os.path.exists(log_path):
        #         os.mkdir(log_path)
        #     dict_logging_files = {"loss_train": [],
        #                         "r2_train": [],
        #                         "loss_val": [],
        #                         "r2_val":[]}

        #     # for name in dict_logging_files.keys():
        #     #     log_file_path = os.path.join(FLAGS.training_res_folder, name+'.txt')
        #     #     if os.path.exists(log_file_path):
        #     #         os.remove(log_file_path)

        #     # if loss is self.loss_reg:
        #     #     pred = self.pred_reg
        #     # elif loss is self.loss_pois:
        #     #     pred = self.pred_pois

        #     round_cnt = 0 # Each round the learning rate is divided by 2

        #     step_per_epoch = X_train.shape[0] // training_batch_size

        #     while round_cnt < FLAGS.training_rounds:
        #         min_validate_loss = float("inf")
        #         best_counter = 0
        #         step = 0  # Total count of training times.
        #         best_index = 0
        #         while (best_counter < stopping_lag and step < FLAGS.max_steps):
        #             step += 1

        #             # print("print batch size: %s." % (batch.batch_size,))
        #             batch_X_train, batch_y_train, _ = batch.getBatch()

        #             # optimizer.minimize(lambda: obj_func(model, batch_X_train , batch_y_train),
        #             # global_step = tf.train.get_or_create_global_step())

        #             # Calculate derivatives of the input function with respect to its
        #             # parameters.
        #             grads = obj_grad(batch_X_train, batch_y_train, model, loss, penal_param)
        #             # Diminishing step for updating.
        #             grads = [grad * (np.floor(step/FLAGS.decay_steps) + 1)**(-0.5) for grad in grads]
        #             # Robbins Monro conditions for convergence of SGD
        #             # grads = [grad * (step/FLAGS.decay_steps + 1)**(-1) for grad in grads]
        #             # Apply the gradient to the model
        #             # optimizer.apply_gradients(list(zip(grads, model.variables)),
        #             #                           global_step=tf.train.get_or_create_global_step())
        #             optimizer.apply_gradients(zip(grads, model.variables))

        #             if plot_trace_flag:
        #                 dict_logging_files['loss_train'].append(loss(X_train, y_train, model))
        #                 dict_logging_files['r2_train'].append(r2_score(y_train, pred(X_train, model)))

        #             if step % step_per_epoch == 0:
        #                 # Validate on dataset every epoch
        #                 dict_logging_files['loss_val'].append(loss(X_val, y_val, model))
        #                 dict_logging_files['r2_val'].append(r2_score(y_val, pred(X_val, model)))            

        #                 if (dict_logging_files['loss_val'][-1] < min_validate_loss):
        #                     model.save_weights(os.path.join(log_path, model_ckpt_fname))
        #                     # logger.info('The variables now is {}.'.format(model.variables,), extra=d)
        #                     # print('The variables at step {} is {}.'.format(step, model.variables))

        #                     FLAGS.best_r2_val = min_validate_loss = dict_logging_files['loss_val'][-1]
        #                     best_counter = 0
        #                     best_index = step
        #                 else:
        #                     best_counter += step_per_epoch

        #                 logger.info(("Validation squared loss and r2"
        #                     " at step (round {}, {}): ({},{},{},{}) {}, {}, {}").format(
        #                                                             round_cnt,
        #                                                             step,
        #                                                             penal_param,
        #                                                             stopping_lag,
        #                                                             training_batch_size,
        #                                                             learning_rate,
        #                                                             dict_logging_files['loss_val'][-1],
        #                                                             dict_logging_files['r2_val'][-1],
        #                                                             r2_score(y_train, pred(X_train, model))), extra=d)
                        
        #                 if len(dict_logging_files['loss_val'])==10000:
        #                     # Without doing this, the memory will explode.
        #                     append_logging_to_file(dict_logging_files, model_ckpt_fname, log_path, purge_flag=False)

        #         learning_rate /= 2
        #         round_cnt += 1
                
        #     # logger.info(
        #     #     'The variables at the end of step %s is %s.',
        #     #     step,
        #     #     model.variables,
        #     #     extra=d)

        #     append_logging_to_file(dict_logging_files, model_ckpt_fname, log_path, purge_flag=False)

        #     if plot_trace_flag:
        #         model_postfix = model_ckpt_fname.split('.')[0]
        #         arr_loss_train = np.genfromtxt(os.path.join(log_path, 'loss_train_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
        #         arr_loss_val = np.genfromtxt(os.path.join(log_path, 'loss_val_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
        #         arr_r2_train = np.genfromtxt(os.path.join(log_path, 'r2_train_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
        #         arr_r2_val = np.genfromtxt(os.path.join(log_path, 'r2_val_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
        #         plot_trace(arr_loss_train, arr_loss_val, best_index, 'upper right', 'loss(mean squared or poisson loss)-{}'.format(step_per_epoch))
        #         plot_trace(arr_r2_train, arr_r2_val, best_index, 'upper right', 'R-squared-{}'.format(step_per_epoch))

        #     # Calculate Gradient for Phase-I and Phase-II
        #     model.load_weights(os.path.join(log_path, model_ckpt_fname))
        #     # logger.info("The best_index %d.", best_index, extra=d)
        #     logger.info("The best_index {} with training R-square as {} and training&validating R-squared as {}.".format(
        #         best_index, r2_score(y_train, pred(X_train, model)), 
        #         r2_score(np.vstack((y_train, y_val)), pred(np.vstack((X_train, X_val)), model))), extra=d)
        #     logger.info('The variables for the best model is {}.'.format(model.variables), extra=d)
        #     # logger.info('The variables now is {}.'.format(model.variables,), extra=d)

        #     val_loss_value = loss(X_val, y_val, model)

        #     val_r2 = r2_score(y_val, pred(X_val, model))

        #     return model, val_loss_value, best_index, val_r2

        # def dev_reg(pred, y):
        #     pred, y = pred.reshape((-1,)), y.reshape((-1,))
        #     dev = (y - pred)**2 # Defined as -ln(Likelihood)
        #     return dev

        # def dev_pois(pred, y):
        #     pred, y = pred.reshape((-1,)), y.reshape((-1,))
        #     lg_factor_y = np.array([np.sum(np.log(np.arange(1, v+1))) for v in y])
        #     dev = pred-y*np.log(pred)+lg_factor_y
        #     return dev

        # def cum_abs_resi(abs_resi_PI, resi_PII):
        #     abs_resi_PII = np.absolute(resi_PII)
        #     abs_resi_PI = np.reshape(abs_resi_PI, (-1,))
        #     abs_resi_PII = np.reshape(abs_resi_PII, (-1,))
        #     abs_resi_PI_PII = np.hstack((abs_resi_PI, abs_resi_PII))
        #     cum_abs_resi_PI_PII = 1.0 * np.cumsum(abs_resi_PI_PII) / (1.0 + np.arange(len(abs_resi_PI_PII)))
        #     cum_abs_resi_PI = cum_abs_resi_PI_PII[:len(abs_resi_PI)]
        #     cum_abs_resi_PII = cum_abs_resi_PI_PII[len(abs_resi_PI):]
        #     return cum_abs_resi_PI, cum_abs_resi_PII, abs_resi_PII, abs_resi_PI_PII

        # def fisher_mat_score_cov(grads, fisher_nugget=0, penal_matrix=None):
        #     # The following coefficient 1.0 is critical in calculating fisher
        #     # information matrix when we have l2 regularization parameter.
        #     grads_mu = np.mean(grads, axis=0)
        #     grads_centered = grads - grads_mu
        #     fisher_info_mat = 1.0*np.matmul(grads_centered.T, grads_centered)/grads.shape[0]

        #     if fisher_nugget > 0:
        #         # Be careful the sign of Fisher Information Matrix.
        #         logger.info("The penalization parameter is %s\n", fisher_nugget, extra=d)
        #         logger.info('Condition # for Fisher Info Mat before including penalization: %s',
        #                     np.linalg.cond(fisher_info_mat), extra=d)
        #         fisher_info_mat = (fisher_info_mat + fisher_nugget * penal_matrix)
        #         logger.info('Condition # for Fisher Info Mat after including penalization: %s',
        #                     np.linalg.cond(fisher_info_mat), extra=d)
        #     return fisher_info_mat

        # def fisher_mat(X_batch, y_batch, model, loss, num_param, fisher_nugget=0, penal_matrix=None):
        #     """ This is to calculate the fisher information matrix using the hessian
        #         matrix of marginal distribution.
        #     """
        #     def gather_tensor(grads, shape):
        #         flatten_tensor = tf.zeros(0)
        #         for g in grads:
        #     #         print g
        #             flatten_tensor=tf.concat([flatten_tensor, tf.reshape(g, (-1,))], axis=0)
        #         return tf.reshape(flatten_tensor, shape)

        #     def hessian_fn(x, y, model, loss):
        #         with tf.GradientTape(persistent=True) as tape:
        #             loss_val = loss(x, y, model)
        #             dy_dx = tape.gradient(loss_val, model.variables)
        #             grads_tensor = gather_tensor(dy_dx, (-1,))
        #             # print("The first order gradient list is: %s.\n" % grads_tensor)
        #             hess_tensor=tf.ones([0, grads_tensor.shape[0]])
        #             for grad in grads_tensor:
        #                 grad_grads = tape.gradient(grad, model.variables)
        #                 # print grad_grads
        #                 hess_tensor=tf.concat([hess_tensor, gather_tensor(grad_grads, (1,-1))], axis=0)
        #                 # print("The second order gradient is: %s.\n" % hess_tensor)
        #         return hess_tensor.numpy()

        #     fisher_mat_est = np.zeros([num_param, num_param])
        #     for i in range(X_batch.shape[0]):
        #         x = tf.convert_to_tensor(X_batch[[i]], dtype=tf.float32)
        #         y = tf.convert_to_tensor(y_batch[[i]], dtype=tf.float32)
        #         fisher_mat_est += hessian_fn(x, y, model, loss)
        #     fisher_mat_est /= X_batch.shape[0]
        #     if fisher_nugget > 0:
        #         logger.info('Condition # for Fisher Info Mat before including penalization: %s',
        #                     np.linalg.cond(fisher_mat_est), extra=d)
        #         fisher_mat_est += fisher_nugget * penal_matrix
        #         logger.info('Condition # for Fisher Info Mat after including penalization: %s',
        #                     np.linalg.cond(fisher_mat_est), extra=d)
        #     return fisher_mat_est

        # # Define those functions as member objects so that other class method can call them and they can be past as parameter to joblib for parallelization.
        # self.loss_reg = loss_reg
        # self.loss_pois = loss_pois
        # self.loss_grad = loss_grad
        # self.pred_reg = pred_reg
        # self.pred_pois = pred_pois
        # self.residual_reg = residual_reg
        # self.residual_pois = residual_pois
        # self.obj_func = obj_func
        # self.obj_grad = obj_grad
        # self.plot_trace = plot_trace
        # self.append_logging_to_file = append_logging_to_file
        # self.Train_Nnet_Reg = Train_Nnet_Reg
        # self.dev_reg = dev_reg
        # self.dev_pois = dev_pois
        # self.cum_abs_resi = cum_abs_resi
        # self.fisher_mat_score_cov = fisher_mat_score_cov
        # self.fisher_mat = fisher_mat

        # The training workflow starts
        self.build_model_param = (hidden_layer_sizes, self.FLAGS.activation, X_train.shape[1], 1, self.FLAGS.output_acti)
        self.model, self.num_param = Build_Model(*self.build_model_param)  # 1D response
        self.initial_weights_file = "initial_weights.h5"
        self.initial_weights_file_path = os.path.join(self.FLAGS.training_res_folder, self.initial_weights_file)
        loading_success = False
        while not loading_success:
            try: # Just prevent the error when change nnet structure
                self.model.load_weights(self.initial_weights_file_path)
                print("Load weights from other replicates!")
                loading_success = True
            except (ValueError, OSError) as e:
                self.model.save_weights(self.initial_weights_file_path)
                print("Initial weights is saved and will be loaded in the future!")
                loading_success = False
                print(e)
            time.sleep(5)

        if self.FLAGS.cv_flag:
            self.model.save_weights(self.initial_weights_file_path)
        elif self.train_PI_flag and not os.path.isfile(self.initial_weights_file_path):
            self.model.save_weights(self.initial_weights_file_path)
        # logger.info("The original model has weights %s.", model.variables, extra=d)

        if y_train is not None and y_train.shape[0]:
            y_train = np.vstack(y_train)
        if y_val is not None and y_val.shape[0]:
            y_val = np.vstack(y_val)
        # if y_train_val is not None and y_train_val.shape[0]:
        #     y_train_val = np.vstack(y_train_val)
        
        # (y_train, y_val) = (np.vstack(y_train), np.vstack(y_val))

        # Standardize dataset
        # The reason to standardize data is that if not when we put penalization on
        # weight matrix, we are actually also put penalization on bias.
        if self.normal_flag:
            self.FLAGS.model_scaler_name = '_'.join([self.model_time_stamp, "scaler.h5"])
            self.FLAGS.model_resp_scaler_name = '_'.join([self.model_time_stamp, "resp_scaler.h5"])
            if self.train_PI_flag:
                # Features
                self.scaler = StandardScaler()
                self.scaler.fit(X_train)
                (X_train, X_val) = (self.scaler.transform(X_train), self.scaler.transform(X_val))
                if X_train_val is not None and X_train_val.shape[0]>0:
                    X_train_val = self.scaler.transform(X_train_val)
                pickle.dump(self.scaler, open(os.path.join(self.FLAGS.training_res_folder,
                                                           self.FLAGS.model_scaler_name), 'wb'))
                print("The mean and std are: {}, {}.".format(self.scaler.mean_, self.scaler.scale_))
                
                # Responses
                self.resp_scaler = MinMaxScaler()
                self.resp_scaler.fit(y_train)
                (y_train, y_val) = (
                    self.resp_scaler.transform(y_train), self.resp_scaler.transform(y_val))
                if y_train_val is not None and y_train_val.shape[0]>0:
                    y_train_val = self.resp_scaler.transform(y_train_val)
                pickle.dump(self.resp_scaler, open(os.path.join(self.FLAGS.training_res_folder,
                                                    self.FLAGS.model_resp_scaler_name), 'wb'))
            else:
                self.scaler = pickle.load(open(os.path.join(self.FLAGS.training_res_folder,
                                                    self.FLAGS.model_scaler_name), 'rb'))
                # logger.info('The mean and std for X_PII for before scaling is (%s, %s).', np.mean(X_PII, axis=0), np.std(X_PII, axis=0), extra=d)
                # X_PII = self.scaler.transform(X_PII)
                # logger.info('The mean and std for X_PII for reloaded scaler is (%s, %s).', np.mean(X_PII, axis=0), np.std(X_PII, axis=0), extra=d)
                self.resp_scaler = pickle.load(open(os.path.join(self.FLAGS.training_res_folder,
                                        self.FLAGS.model_resp_scaler_name), 'rb')) 
                # y_PII = self.resp_scaler.transform(y_PII)

        # if self.FLAGS.reg_model == 'lin' or self.FLAGS.reg_model == 'nnet_lin':
        #     loss = self.loss = self.loss_reg
        #     self.residual = self.residual_reg
        #     self.pred = self.pred_reg
        #     self.dev = self.dev_reg
        # elif self.FLAGS.reg_model == 'pois' or self.FLAGS.reg_model == 'nnet_pois':
        #     loss = self.loss = self.loss_pois
        #     self.residual = self.residual_pois
        #     self.pred = self.pred_pois
        #     self.dev = self.dev_pois

        from regression.regressors_nnet_utils import Train_Nnet_Reg, loss_reg, loss_pois, pred_reg, pred_pois, obj_grad
        if self.train_PI_flag and (self.cv_tasks_info is not None):
            # Cross-validation.
            self.N_rep, self.K_fold, self.cv_param_ls, self.cv_rand_search, self.cv_tasks, self.n_jobs = self.cv_tasks_info
            if self.FLAGS.reg_model == 'lin' or self.FLAGS.reg_model == 'nnet_lin':
                self.best_cv_param = CV_Nnet(Build_Model, self.build_model_param, self.initial_weights_file,
                    Train_Nnet_Reg, loss_reg, pred_reg, np.vstack((X_train, X_val)), np.vstack((y_train, y_val)), None,
                    self.N_rep, self.K_fold, self.cv_param_ls, self.cv_rand_search, self.cv_tasks, self.n_jobs, self.FLAGS)
            elif self.FLAGS.reg_model == 'pois' or self.FLAGS.reg_model == 'nnet_pois':
                self.best_cv_param = CV_Nnet(Build_Model, self.build_model_param, self.initial_weights_file,
                    Train_Nnet_Reg, loss_pois, pred_pois, np.vstack((X_train, X_val)), np.vstack((y_train, y_val)), None,
                    self.N_rep, self.K_fold, self.cv_param_ls, self.cv_rand_search, self.cv_tasks, self.n_jobs, self.FLAGS)
            self.cv_penal_param, self.cv_stopping_lag, self.cv_training_batch_size, self.cv_learning_rate = self.best_cv_param
            self.penal_param, self.stopping_lag, self.training_batch_size, self.learning_rate = self.best_cv_param
            logger.info('The best hyper-parameter after cross-validation is %s.', self.best_cv_param, extra=d) 
            self.stopping_lag = self.cv_stopping_lag = int(self.cv_stopping_lag)
            self.training_batch_size = self.cv_training_batch_size = int(self.cv_training_batch_size) # Important!!!
        else:
            if os.path.isfile(os.path.join(os.path.join(self.FLAGS.res_root_dir, self.FLAGS.model_file_folder), "best_cv_res.csv")):
                self.best_cv_param = pd.read_csv(os.path.join(os.path.join(self.FLAGS.res_root_dir, self.FLAGS.model_file_folder), "best_cv_res.csv"))
                self.best_cv_param = (np.array(self.best_cv_param))
                self.best_cv_param = tuple(np.reshape(self.best_cv_param, (-1,)))
                # print best_cv_param
                self.cv_penal_param, self.cv_stopping_lag, self.cv_training_batch_size, self.cv_learning_rate = self.best_cv_param
                self.penal_param, self.stopping_lag, self.training_batch_size, self.learning_rate = self.best_cv_param
                self.stopping_lag = self.cv_stopping_lag = int(self.cv_stopping_lag)
                self.stopping_lag = self.cv_training_batch_size = int(self.cv_training_batch_size) # Important!!!
                
            else:
                self.best_cv_param = (self.penal_param, self.stopping_lag, self.training_batch_size, self.learning_rate)

        self.model_ckpt_fname = '_'.join([self.model_time_stamp, self.FLAGS.best_model]) # In this way, the model can be deleted in large-scale simulations.


        log_folder = 'log_folder'

        # if train_PI_flag:
        # if not os.path.isfile(os.path.join(self.FLAGS.training_res_folder, model_ckpt_fname)):
        if train_PI_flag:
            if self.FLAGS.reg_model == 'lin' or self.FLAGS.reg_model == 'nnet_lin':
                self.model, _, self.best_index, _= Train_Nnet_Reg(Build_Model, self.build_model_param, self.initial_weights_file,
                            loss_reg, pred_reg, np.vstack((X_train, X_val)), np.vstack((y_train, y_val)), None,
                            list(np.arange(X_train.shape[0])),
                            list(X_train.shape[0] + np.arange(X_val.shape[0])),
                            self.penal_param, self.stopping_lag, self.training_batch_size,
                            self.learning_rate, self.model_ckpt_fname, self.FLAGS, log_folder, plot_trace_flag=self.plot_trace_flag)
            elif self.FLAGS.reg_model == 'pois' or self.FLAGS.reg_model == 'nnet_pois':
                self.model, _, self.best_index, _= Train_Nnet_Reg(Build_Model, self.build_model_param, self.initial_weights_file,
                            loss_pois, pred_pois, np.vstack((X_train, X_val)), np.vstack((y_train, y_val)), None,
                            list(np.arange(X_train.shape[0])),
                            list(X_train.shape[0] + np.arange(X_val.shape[0])),
                            self.penal_param, self.stopping_lag, self.training_batch_size,
                            self.learning_rate, self.model_ckpt_fname, self.FLAGS, log_folder, plot_trace_flag=self.plot_trace_flag)
            print(("The final model checkpoint file name is {}.\n".format(self.model_ckpt_fname)))
        else:
            # Calculate Gradient for Phase-I and Phase-II
            log_path = os.path.join(self.FLAGS.training_res_folder, log_folder)
            self.model.load_weights(os.path.join(log_path, self.model_ckpt_fname))
            logger.info('The variables for the best model (reloaded) is %s.', self.model.variables, extra=d)

        # self.grad_func = self.obj_grad
        # self.grad_func_kwargs = {'model':self.model, 'loss':self.loss, 'penal_param':self.penal_param}
        # self.grad_func_paral_kwargs = {'model_weights':self.model.get_weights(), 'loss':self.loss, 'penal_param':self.penal_param}

        grad_func = obj_grad
        grad_func_kwargs = {'model':self.model, 'loss':loss_reg, 'penal_param':self.penal_param}
        grad_func_paral_kwargs = {'model_weights':self.model.get_weights(), 'loss':loss_reg, 'penal_param':self.penal_param}
        if self.FLAGS.reg_model == 'lin' or self.FLAGS.reg_model == 'nnet_lin':
            grad_func_paral_kwargs['loss']=grad_func_kwargs['loss']=loss_reg
        elif self.FLAGS.reg_model == 'pois' or self.FLAGS.reg_model == 'nnet_pois':
            grad_func_paral_kwargs['loss']=grad_func_kwargs['loss']=loss_pois

        self.penal_vec = calPenalBoolVec(
                X_train[[0]],
                y_train[[0]],
                self.hidden_layer_sizes,
                grad_func,
                **grad_func_kwargs)
        self.penal_matrix = np.diag(self.penal_vec)

    def cal_metrics(self, X, y, data_info='train'):
        """ 
            X: Predictor matrix
            y: In shape (-1, 1)
        
        """
        print("Calculate scores for {}...".format(data_info))
        start_time = time.time()
        # Cannot directly pass member function into a function that use joblib and that function as parameter.
        # https://stackoverflow.com/a/50704372/4307919
        from regression.regressors_nnet_utils import obj_grad, loss_reg, loss_pois, residual_reg, residual_pois, dev_reg, dev_pois, fisher_mat_score_cov
        grad_func = obj_grad
        grad_func_kwargs = {'model':self.model, 'loss':loss_reg, 'penal_param':self.penal_param}
        grad_func_paral_kwargs = {'model_weights':self.model.get_weights(), 'loss':loss_reg, 'penal_param':self.penal_param}
        
        if self.FLAGS.reg_model == 'lin' or self.FLAGS.reg_model == 'nnet_lin':
            grad_func_paral_kwargs['loss']=grad_func_kwargs['loss']=loss_reg
        elif self.FLAGS.reg_model == 'pois' or self.FLAGS.reg_model == 'nnet_pois':
            grad_func_paral_kwargs['loss']=grad_func_kwargs['loss']=loss_pois

        grads = grad_func_batch_paral(X, y, Build_Model, self.build_model_param, grad_func, **grad_func_paral_kwargs)
        print("The calculation for {} takes {}s.".format(data_info, time.time()-start_time))
        fisher_info_mat = fisher_mat_score_cov(grads, fisher_nugget=0, penal_matrix=None)
        
        # Score mean and inverse of variance matrix for later ewma calculation.
        mu = np.mean((-1.0)*grads, axis=0)
        Sinv = Inv_Cov((-1.0)*grads, self.FLAGS.nugget)

        if self.FLAGS.reg_model == 'lin' or self.FLAGS.reg_model == 'nnet_lin':
            resi, pred = residual_reg(X, y, self.model)
            dev = dev_reg(pred, y)
        elif self.FLAGS.reg_model == 'pois' or self.FLAGS.reg_model == 'nnet_pois':
            resi, pred = residual_pois(X, y, self.model)
            dev = dev_pois(pred, y)
        abs_resi = np.absolute(resi)

        if self.normal_flag:
            pred = self.resp_scaler.inverse_transform(pred.reshape((-1,1)))

        if data_info == 'train':
            self.FLAGS.mu_train, self.FLAGS.Sinv_train, self.FLAGS.fisher_info_mat_train = mu, Sinv, fisher_info_mat
            logger.info("The training score mean is {}.".format(self.FLAGS.mu_train), extra=d)
            logger.info("The training score Sinv is {}.".format(self.FLAGS.Sinv_train), extra=d)
            self.FLAGS.best_r2_train = r2_score(y, pred)
        elif data_info == 'PI':
            self.FLAGS.mu_PI, self.FLAGS.Sinv_PI, self.FLAGS.fisher_info_mat_PI = mu, Sinv, fisher_info_mat
            # self.FLAGS.mu_train = self.FLAGS.mu_PI
            logger.info("The PI score mean is {}.".format(self.FLAGS.mu_PI), extra=d)
            logger.info("The PI score Sinv is {}.".format(self.FLAGS.Sinv_PI), extra=d)

        logger.info("The R-squared for {} data set: {}\n".format(data_info, r2_score(y, pred)), extra=d)

        return fisher_info_mat, pred.reshape((-1,)), grads, resi.reshape((-1,)), abs_resi.reshape((-1,)), dev.reshape((-1,))


def Neural_Network_Reg(X_train,
                       y_train,
                       X_val,
                       y_val,
                       X_PI,
                       y_PI,
                       X_PII,
                       y_PII,
                       N_PIIs,
                       gamma,  # EWMA parameter, currently it is always 0.
                       hidden_layer_sizes,
                       penal_type,
                       penal_param,
                       stopping_lag,
                       training_batch_size,
                       learning_rate,
                       train_PI_flag,
                       model_time_stamp,
                       FLAGS,
                       X_train_val=None,
                       y_train_val=None,
                       normal_flag=True, # Whether the data will be normalized in this function.
                       cv_tasks_info = None,
                       plot_trace_flag=False):
    """ Fit a neural network for classification problem.

    Args:
        X_train: The training variables as nparray.
        y_train: The training targets as nparray.
        X_val: The validation variables as nparray.
        y_val: The validation targets as nparray.
        N_PIIs: Not useful here.
        hidden_layer_sizes: A list of number of hidden nodes for each hidden layer.
        penal_type: L1, L2 or None.
        penal_param: Penalization parameter.
        stopping_lag: The stopping lag before training stop.
        training_batch_size: The training batch size.
        learning_rate: The learning rate.
        FLAGS: Some parameters.
        cv_tasks_info: A tuple containing cv_tasks, N_rep, K_fold.

    Returns:

    """
    activation = FLAGS.activation

    build_model_param = (hidden_layer_sizes, activation, X_train.shape[1], 1, FLAGS.output_acti)
    model, num_param = Build_Model(*build_model_param)  # 1D response
    initial_weights_file = "initial_weights.h5"
    initial_weights_file_path = os.path.join(FLAGS.training_res_folder, initial_weights_file)
    try: # Just prevent the error when change nnet structure
        model.load_weights(initial_weights_file_path)
    except (ValueError, OSError) as e:
        model.save_weights(initial_weights_file_path)

    if FLAGS.cv_flag:
        model.save_weights(initial_weights_file_path)
    elif train_PI_flag and not os.path.isfile(initial_weights_file_path):
        model.save_weights(initial_weights_file_path)
    # logger.info("The original model has weights %s.", model.variables, extra=d)

    if y_train is not None and y_train.shape[0]:
        y_train = np.vstack(y_train)
    if y_val is not None and y_val.shape[0]:
        y_val = np.vstack(y_val)
    if y_train_val is not None and y_train_val.shape[0]:
        y_train_val = np.vstack(y_train_val)
    if y_PI is not None and y_PI.shape[0]:
        y_PI = np.vstack(y_PI)
    if y_PII is not None and y_PII.shape[0]:
        y_PII = np.vstack(y_PII)
    
    # (y_train, y_val) = (np.vstack(y_train), np.vstack(y_val))

    # Standardize dataset
    # The reason to standardize data is that if not when we put penalization on
    # weight matrix, we are actually also put penalization on bias.
    if normal_flag:
        FLAGS.model_scaler_name = '_'.join([model_time_stamp, "scaler.h5"])
        FLAGS.model_resp_scaler_name = '_'.join([model_time_stamp, "resp_scaler.h5"])
        if train_PI_flag:
            # Features
            scaler = StandardScaler()
            scaler.fit(X_train)
            (X_train, X_val) = (scaler.transform(X_train), scaler.transform(X_val))
            if X_PI is not None and X_PI.shape[0]>0:
                X_PI = scaler.transform(X_PI)
            if X_PII is not None and X_PII.shape[0]>0:
                X_PII = scaler.transform(X_PII)
            if X_train_val is not None and X_train_val.shape[0]>0:
                X_train_val = scaler.transform(X_train_val)
            pickle.dump(scaler, open(os.path.join(FLAGS.training_res_folder,
                                                FLAGS.model_scaler_name), 'wb'))
            print("The mean and std are: {}, {}.".format(scaler.mean_, scaler.scale_))
            
            # Responses
            resp_scaler = MinMaxScaler()
            resp_scaler.fit(y_train)
            (y_train, y_val) = (
                resp_scaler.transform(y_train), resp_scaler.transform(y_val))
            if y_PI is not None and y_PI.shape[0]>0:
                y_PI = resp_scaler.transform(y_PI)
            if y_PII is not None and y_PII.shape[0]>0:
                y_PII = resp_scaler.transform(y_PII)
            if y_train_val is not None and y_train_val.shape[0]>0:
                y_train_val = resp_scaler.transform(y_train_val)
            pickle.dump(resp_scaler, open(os.path.join(FLAGS.training_res_folder,
                                                FLAGS.model_resp_scaler_name), 'wb'))
        else:
            scaler = pickle.load(open(os.path.join(FLAGS.training_res_folder,
                                                FLAGS.model_scaler_name), 'rb'))
            # logger.info('The mean and std for X_PII for before scaling is (%s, %s).', np.mean(X_PII, axis=0), np.std(X_PII, axis=0), extra=d)
            X_PII = scaler.transform(X_PII)
            # logger.info('The mean and std for X_PII for reloaded scaler is (%s, %s).', np.mean(X_PII, axis=0), np.std(X_PII, axis=0), extra=d)
            resp_scaler = pickle.load(open(os.path.join(FLAGS.training_res_folder,
                                    FLAGS.model_resp_scaler_name), 'rb')) 
            y_PII = resp_scaler.transform(y_PII)


    def loss_reg(inputs, targets, model, wei=None):
        predictions = model(inputs)
        return tf.reduce_mean(tf.losses.mean_squared_error(
            y_true=targets, y_pred=predictions))

    def loss_pois(inputs, targets, model, wei=None):
        log_inputs = model(inputs)
        return tf.reduce_mean(tf.nn.log_poisson_loss(
            targets=targets, log_input=log_inputs, compute_full_loss=True))

    # Use the self-written loss function as below is ~5x slower than built-in log_poisson_loss. 20190602.
    # def loss_pois(inputs, targets, model):
    #     log_inputs = model(inputs)
    #     const = tf.reduce_mean(np.sum([np.sum(np.log(np.arange(1, v+1))) for v in targets], dtype=np.float32))
    #     # logger.info("The target and log_factorial are %s, %s.", targets, [np.sum(np.log(np.arange(1, v+1))) for v in targets], extra=d)
    #     return tf.reduce_mean(-targets*log_inputs+tf.exp(log_inputs)) + const

    def loss_grad(inputs, targets, model, loss, wei=None):
        with tf.GradientTape() as tape:
            loss_value = loss(inputs, targets, model)
        return tape.gradient(loss_value, model.variables)

    def pred_reg(inputs, model):
        return np.reshape(model(inputs), (-1,))

    def pred_pois(inputs, model):
        return np.exp(np.reshape(model(inputs), (-1,)))

    def residual_reg(inputs, targets, model):
        predictions = pred_reg(inputs, model)
        # print "Predictions from the model."
        # print predictions
        targets, predictions = targets.reshape((-1,)), predictions.reshape((-1,))
        return targets-predictions, predictions

    def residual_pois(inputs, targets, model):
        predictions = pred_pois(inputs, model)
        # print "Predictions from the model."
        # print predictions
        targets, predictions = targets.reshape((-1,)), predictions.reshape((-1,))
        return targets-predictions, predictions

    def obj_func(inputs, targets, model, loss, penal_param, wei=None):
        if penal_type.lower() == 'none':
            obj = loss(inputs, targets, model)
        else:
            if penal_type.lower() == 'l2':
                # Only penalize weights not biases. Has tested that the
                # gradient has penalization part.
                obj = loss(inputs, targets, model) + 0.5 * penal_param * tf.reduce_sum(
                    [tf.nn.l2_loss(var) for var in model.variables if re.findall(r'kernel', var.name)])
            if penal_type.lower() == 'l1':
                obj = loss(inputs, targets, model) + penal_param * tf.reduce_sum(
                    [tf.reduce_sum(tf.abs(var)) for var in model.variables if re.findall(r'kernel', var.name)])
        return obj

    def obj_grad(inputs, targets, model, loss, penal_param, wei=None):
        with tf.GradientTape() as tape:
            obj_value = obj_func(inputs, targets, model, loss, penal_param)
        return tape.gradient(obj_value, model.variables)

    # def cal_penal_bool_vec(
    #         x,
    #         y,
    #         hidden_layer_sizes,
    #         grad_func,
    #         **kwargs):
    #     x = tf.convert_to_tensor(x, dtype=tf.float32)
    #     y = tf.convert_to_tensor(y)
    #     grad_raw = grad_func(x, y, **kwargs)
    #     penal_bool_vec = np.array([])
    #     for i in xrange(len(hidden_layer_sizes) + 1):
    #         # Already checked the dimension of gradient.
    #         # For the i to (i+1)th layer (input layer is special layer,
    #         # let's call it layer 0), the weights are stored in 2*i
    #         # of grad_raw, and biases are stored in 2*i+1 of grad_raw.
    #         # The values are stored in grad_raw[2*i] and grad_raw[2*i+1]
    #         # as matrices and arrays. jth column in the matrics corresponds
    #         # to jth nodes of ith layer; jth element in the arrays corresponds
    #         # to jth nodes of ith layer.
    #         # There is no value of weights.
    #         #
    #         # gradient for weights and biases
    #         penal_bool_vec = np.append(
    #             penal_bool_vec, np.ones_like(grad_raw[2 * i].numpy()))
    #         penal_bool_vec = np.append(
    #             penal_bool_vec, np.zeros_like(grad_raw[2 * i + 1].numpy()))
    #         # print("The sample gradient {} with length {}".format(grads_PI[-1], grads_PI[-1].shape))
    #     return penal_bool_vec

    def plot_trace(trace_train, trace_val, best_index, pos, name, prefix='reg'):
        fig, ax = plt.subplots()
        print((len(trace_train), training_batch_size))
        ax.plot(
            np.arange(len(trace_train)) * training_batch_size,
            trace_train,
            'b-',
            label='Training ' + name)
        ax.plot(
            np.arange(len(trace_val)) * training_batch_size,
            trace_val,
            'r-',
            label='Validation ' + name)
        ax.axvline(x=best_index * training_batch_size)
        legend = ax.legend(loc=pos, shadow=False, fontsize='x-large', fancybox=True, facecolor='white', framealpha=0.5)
        # Put a nicer background color on the legend.
        # legend.get_frame().set_facecolor('#00FFCC')
        plt.savefig(
            os.path.join(FLAGS.training_res_folder,
            '_'.join([prefix, str(penal_param).replace(".", "_"), name])+'.png'))
        plt.show()
        plt.close()

    def append_logging_to_file(dict_logging_files, model_ckpt_fname, log_path, purge_flag=True):
        for name, log_ls in dict_logging_files.items():
            log_fname = name+'_'+model_ckpt_fname.split('.')[0]+'.txt'
            with open(os.path.join(log_path, log_fname), "a+") as log_f:
                log_f.write('\n'.join(["{0:.3f}".format(ele,) for ele in log_ls]))
            if purge_flag:
                dict_logging_files[name]=[]

    def Train_Nnet_Reg(gen_model_func, gen_model_func_param, initial_weights_file,
                       loss, pred, X, y, wei, train_idx_ls, val_idx_ls,
                       penal_param, stopping_lag, training_batch_size,
                       learning_rate, model_ckpt_fname, FLAGS, log_folder, plot_trace_flag=False):
        """ Train neural network using the training and validation datasets. """
        # Create a new model
        # Looks like I cannot pass model as parameter to this function. Otherwise, it cannot use joblib to parallelize jobs.
        model, _ = gen_model_func(*gen_model_func_param)
        model.load_weights(os.path.join(FLAGS.training_res_folder, initial_weights_file))
        logger.info("The new model has weights %s.", model.variables, extra=d)

        X_train, y_train = X[train_idx_ls,:], y[train_idx_ls]
        X_val, y_val = X[val_idx_ls,:], y[val_idx_ls]

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        batch = Batch(X_train, y_train, training_batch_size, wei=wei)

        log_path = os.path.join(FLAGS.training_res_folder, log_folder)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        dict_logging_files = {"loss_train": [],
                              "r2_train": [],
                              "loss_val": [],
                              "r2_val":[]}

        # for name in dict_logging_files.keys():
        #     log_file_path = os.path.join(FLAGS.training_res_folder, name+'.txt')
        #     if os.path.exists(log_file_path):
        #         os.remove(log_file_path)

        if loss is loss_reg:
            pred = pred_reg
        elif loss is loss_pois:
            pred = pred_pois

        round_cnt = 0 # Each round the leanring rate is divided by 2

        step_per_epoch = X_train.shape[0] // training_batch_size

        while round_cnt < FLAGS.training_rounds:
            min_validate_loss = float("inf")
            best_counter = 0
            step = 0  # Total count of training times.
            best_index = 0
            while (best_counter < stopping_lag and step < FLAGS.max_steps):
                step += 1

                # print("print batch size: %s." % (batch.batch_size,))
                batch_X_train, batch_y_train, _ = batch.getBatch()

                # optimizer.minimize(lambda: obj_func(model, batch_X_train , batch_y_train),
                # global_step = tf.train.get_or_create_global_step())

                # Calculate derivatives of the input function with respect to its
                # parameters.
                grads = obj_grad(batch_X_train, batch_y_train, model, loss, penal_param)
                # Diminishing step for updating.
                grads = [grad * (np.floor(step/FLAGS.decay_steps) + 1)**(-0.5) for grad in grads]
                # Robbins Monro conditions for convergence of SGD
                # grads = [grad * (step/FLAGS.decay_steps + 1)**(-1) for grad in grads]
                # Apply the gradient to the model
                # optimizer.apply_gradients(list(zip(grads, model.variables)),
                #                           global_step=tf.train.get_or_create_global_step())
                optimizer.apply_gradients(zip(grads, model.variables))

                if plot_trace_flag:
                    dict_logging_files['loss_train'].append(loss(X_train, y_train, model))
                    dict_logging_files['r2_train'].append(r2_score(y_train, pred(X_train, model)))

                if step % step_per_epoch == 0:
                    # Validate on dataset every epoch
                    dict_logging_files['loss_val'].append(loss(X_val, y_val, model))
                    dict_logging_files['r2_val'].append(r2_score(y_val, pred(X_val, model)))            

                    if (dict_logging_files['loss_val'][-1] < min_validate_loss):
                        model.save_weights(os.path.join(log_path, model_ckpt_fname))
                        # logger.info('The variables now is {}.'.format(model.variables,), extra=d)
                        # print('The variables at step {} is {}.'.format(step, model.variables))

                        FLAGS.best_r2_val = min_validate_loss = dict_logging_files['loss_val'][-1]
                        best_counter = 0
                        best_index = step
                    else:
                        best_counter += step_per_epoch

                    logger.info(("Validation squared loss and r2"
                        " at step (round {}, {}): ({},{},{},{}) {}, {}, {}").format(
                                                                round_cnt,
                                                                step,
                                                                penal_param,
                                                                stopping_lag,
                                                                training_batch_size,
                                                                learning_rate,
                                                                dict_logging_files['loss_val'][-1],
                                                                dict_logging_files['r2_val'][-1],
                                                                r2_score(y_train, pred(X_train, model))), extra=d)
                    
                    if len(dict_logging_files['loss_val'])==10000:
                        # Without doing this, the memory will explode.
                        append_logging_to_file(dict_logging_files, model_ckpt_fname, log_path, purge_flag=False)

            learning_rate /= 2
            round_cnt += 1
            
        # logger.info(
        #     'The variables at the end of step %s is %s.',
        #     step,
        #     model.variables,
        #     extra=d)

        append_logging_to_file(dict_logging_files, model_ckpt_fname, log_path, purge_flag=False)

        if plot_trace_flag:
            model_postfix = model_ckpt_fname.split('.')[0]
            arr_loss_train = np.genfromtxt(os.path.join(log_path, 'loss_train_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
            arr_loss_val = np.genfromtxt(os.path.join(log_path, 'loss_val_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
            arr_r2_train = np.genfromtxt(os.path.join(log_path, 'r2_train_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
            arr_r2_val = np.genfromtxt(os.path.join(log_path, 'r2_val_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
            plot_trace(arr_loss_train, arr_loss_val, best_index, 'upper right', 'loss(mean squared or poisson loss)-{}'.format(step_per_epoch))
            plot_trace(arr_r2_train, arr_r2_val, best_index, 'upper right', 'R-squared-{}'.format(step_per_epoch))

        # Calculate Gradient for Phase-I and Phase-II
        model.load_weights(os.path.join(log_path, model_ckpt_fname))
        # logger.info("The best_index %d.", best_index, extra=d)
        logger.info("The best_index {} with training R-square as {} and training&validating R-squared as {}.".format(
            best_index, r2_score(y_train, pred(X_train, model)), 
            r2_score(np.vstack((y_train, y_val)), pred(np.vstack((X_train, X_val)), model))), extra=d)
        logger.info('The variables for the best model is {}.'.format(model.variables), extra=d)
        # logger.info('The variables now is {}.'.format(model.variables,), extra=d)

        val_loss_value = loss(X_val, y_val, model)

        val_r2 = r2_score(y_val, pred(X_val, model))

        return model, val_loss_value, best_index, val_r2

    if FLAGS.reg_model == 'lin' or FLAGS.reg_model == 'nnet_lin':
        loss = loss_reg
        residual = residual_reg
        pred = pred_reg
    elif FLAGS.reg_model == 'pois' or FLAGS.reg_model == 'nnet_pois':
        loss = loss_pois
        residual = residual_pois
        pred = pred_pois

    if train_PI_flag and (cv_tasks_info is not None):
        # Cross-validation.
        N_rep, K_fold, cv_param_ls, cv_rand_search, cv_tasks, n_jobs = cv_tasks_info
        best_cv_param = CV_Nnet(Build_Model, build_model_param, initial_weights_file,
            Train_Nnet_Reg, loss, pred, np.vstack((X_train, X_val)), np.vstack((y_train, y_val)), None,
            N_rep, K_fold, cv_param_ls, cv_rand_search, cv_tasks, n_jobs, FLAGS)
        penal_param, stopping_lag, training_batch_size, learning_rate = best_cv_param
        logger.info('The best hyper-parameter after cross-validation is %s.', best_cv_param, extra=d) 
        stopping_lag = int(stopping_lag)
        training_batch_size = int(training_batch_size) # Important!!!
    else:
        if os.path.isfile(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), "best_cv_res.csv")):
            best_cv_param = pd.read_csv(os.path.join(os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), "best_cv_res.csv"))
            best_cv_param = (np.array(best_cv_param))
            best_cv_param = tuple(np.reshape(best_cv_param, (-1,)))
            # print best_cv_param
            penal_param, stopping_lag, training_batch_size, learning_rate = best_cv_param
            stopping_lag = int(stopping_lag)
            training_batch_size = int(training_batch_size) # Important!!!
        else:
            best_cv_param = (penal_param, stopping_lag, training_batch_size, learning_rate)

    model_ckpt_fname = '_'.join([model_time_stamp, FLAGS.best_model]) # In this way, the model can be deleted in large-scale simulations.


    log_folder = 'log_folder'

    # if train_PI_flag:
    # if not os.path.isfile(os.path.join(FLAGS.training_res_folder, model_ckpt_fname)):
    if train_PI_flag:
        model, _, best_index, _= Train_Nnet_Reg(Build_Model, build_model_param, initial_weights_file,
                           loss, pred, np.vstack((X_train, X_val)), np.vstack((y_train, y_val)), None,
                           list(np.arange(X_train.shape[0])),
                           list(X_train.shape[0] + np.arange(X_val.shape[0])),
                           penal_param, stopping_lag, training_batch_size,
                           learning_rate, model_ckpt_fname, FLAGS, log_folder, plot_trace_flag=plot_trace_flag)
        print(("The final model checkpoint file name is {}.\n".format(model_ckpt_fname)))
    else:
        # Calculate Gradient for Phase-I and Phase-II
        log_path = os.path.join(FLAGS.training_res_folder, log_folder)
        model.load_weights(os.path.join(log_path, model_ckpt_fname))
        logger.info('The variables for the best model (reloaded) is %s.', model.variables, extra=d)

    # grad_loss = tf.implicit_gradients(loss)
    # obj_grad(model, loss, inputs, targets)
    # grad_func = tf.implicit_gradients(obj_func)

    def dev_reg(pred, y):
        pred, y = pred.reshape((-1,)), y.reshape((-1,))
        dev = (y - pred)**2 # Defined as -ln(Likelihood)
        return dev

    def dev_pois(pred, y):
        pred, y = pred.reshape((-1,)), y.reshape((-1,))
        lg_factor_y = np.array([np.sum(np.log(np.arange(1, v+1))) for v in y])
        dev = pred-y*np.log(pred)+lg_factor_y
        return dev

    if FLAGS.reg_model == 'lin' or FLAGS.reg_model == 'nnet_lin':
        dev = dev_reg
    elif FLAGS.reg_model == 'pois' or FLAGS.reg_model == 'nnet_pois':
        dev = dev_pois

    def cum_abs_resi(abs_resi_PI, resi_PII):
        abs_resi_PII = np.absolute(resi_PII)
        abs_resi_PI = np.reshape(abs_resi_PI, (-1,))
        abs_resi_PII = np.reshape(abs_resi_PII, (-1,))
        abs_resi_PI_PII = np.hstack((abs_resi_PI, abs_resi_PII))
        cum_abs_resi_PI_PII = 1.0 * \
            np.cumsum(abs_resi_PI_PII) / (1.0 + np.arange(len(abs_resi_PI_PII)))
        cum_abs_resi_PI = cum_abs_resi_PI_PII[:len(abs_resi_PI)]
        cum_abs_resi_PII = cum_abs_resi_PI_PII[len(abs_resi_PI):]
        return cum_abs_resi_PI, cum_abs_resi_PII, abs_resi_PII, abs_resi_PI_PII

    def fisher_mat_score_cov(grads, fisher_nugget=0, penal_matrix=None):
        # The following coefficient 1.0 is critical in calculating fisher
        # information matrix when we have l2 regularization parameter.
        grads_mu = np.mean(grads, axis=0)
        grads_centered = grads - grads_mu
        fisher_info_mat = 1.0*np.matmul(grads_centered.T, grads_centered)/grads.shape[0]

        if fisher_nugget > 0:
            # Be careful the sign of Fisher Information Matrix.
            logger.info("The penalization parameter is %s\n", fisher_nugget, extra=d)
            logger.info('Condition # for Fisher Info Mat before including penalization: %s',
                        np.linalg.cond(fisher_info_mat), extra=d)
            fisher_info_mat = (fisher_info_mat + fisher_nugget * penal_matrix)
            logger.info('Condition # for Fisher Info Mat after including penalization: %s',
                        np.linalg.cond(fisher_info_mat), extra=d)
        return fisher_info_mat

    def fisher_mat(X_batch, y_batch, model, loss, num_param, fisher_nugget=0, penal_matrix=None):
        """ This is to calculate the fisher information matrix using the hessian
            matrix of marginal distribution.
        """
        def gather_tensor(grads, shape):
            flatten_tensor = tf.zeros(0)
            for g in grads:
        #         print g
                flatten_tensor=tf.concat([flatten_tensor, tf.reshape(g, (-1,))], axis=0)
            return tf.reshape(flatten_tensor, shape)

        def hessian_fn(x, y, model, loss):
            with tf.GradientTape(persistent=True) as tape:
                loss_val = loss(x, y, model)
                dy_dx = tape.gradient(loss_val, model.variables)
                grads_tensor = gather_tensor(dy_dx, (-1,))
                # print("The first order gradient list is: %s.\n" % grads_tensor)
                hess_tensor=tf.ones([0, grads_tensor.shape[0]])
                for grad in grads_tensor:
                    grad_grads = tape.gradient(grad, model.variables)
                    # print grad_grads
                    hess_tensor=tf.concat([hess_tensor, gather_tensor(grad_grads, (1,-1))], axis=0)
                    # print("The second order gradient is: %s.\n" % hess_tensor)
            return hess_tensor.numpy()

        fisher_mat_est = np.zeros([num_param, num_param])
        for i in range(X_batch.shape[0]):
            x = tf.convert_to_tensor(X_batch[[i]], dtype=tf.float32)
            y = tf.convert_to_tensor(y_batch[[i]], dtype=tf.float32)
            fisher_mat_est += hessian_fn(x, y, model, loss)
        fisher_mat_est /= X_batch.shape[0]
        if fisher_nugget > 0:
            logger.info('Condition # for Fisher Info Mat before including penalization: %s',
                        np.linalg.cond(fisher_mat_est), extra=d)
            fisher_mat_est += fisher_nugget * penal_matrix
            logger.info('Condition # for Fisher Info Mat after including penalization: %s',
                        np.linalg.cond(fisher_mat_est), extra=d)
        return fisher_mat_est

    # Use score from loss to monitor
    # grad_func = loss_grad
    # Use penalized score to monitor
    grad_func = obj_grad
    grad_func_kwargs = {'model':model, 'loss':loss, 'penal_param':penal_param}
    grad_func_paral_kwargs = {'model_weights':model.get_weights(), 'loss':loss, 'penal_param':penal_param}

    # Phase I
    if train_PI_flag:
        # Calculate mean and covariance matrix of the score function for
        # training data.
        # Calculate score mean and variance, also the fisher information matrix
        # Here, we cannot use analytic form of fisher information matrix, because
        # there is no way to calculate the second order derivative analytically. So
        # we have to calculate the Fisher information matrix by sample mean of
        # gradient.T * gradient matrix.
        print("Calculate scores for train...")
        grads_train = grad_func_batch_paral(
            X_train,
            y_train,
            Build_Model, 
            build_model_param,
            grad_func,
            **grad_func_paral_kwargs)

        # No simple analytic form of second order derivative matrix.
        # fisher_info_mat_train = fisher_mat(grads_train, FLAGS.nugget)
        # fisher_info_mat_train = fisher_mat(grads_train, penal_param)
        penal_vec = calPenalBoolVec(
                X_train[[0]],
                y_train[[0]],
                hidden_layer_sizes,
                grad_func,
                **grad_func_kwargs)
        penal_matrix = np.diag(penal_vec)
        fisher_info_mat_train = fisher_mat_score_cov(grads_train, fisher_nugget=0, penal_matrix=None)
        # fisher_mat(X_train, y_train, model, loss, num_param, fisher_nugget=penal_param, penal_matrix=penal_matrix)

        # Score mean and inverse of variance matrix for later ewma calculation.
        mu_train = np.mean((-1.0)*grads_train, axis=0)
        Sinv_train = Inv_Cov((-1.0)*grads_train, FLAGS.nugget)

        resi_train, pred_train = residual(X_train, y_train, model)
        abs_resi_train = np.absolute(resi_train)
        dev_train = dev(pred_train, y_train)

        # train_val
        if X_train_val is not None and X_train_val.shape[0]>0:
            print("Calculate scores for train_val...")
            grads_train_val = grad_func_batch_paral(
                X_train_val,
                y_train_val,
                Build_Model, 
                build_model_param,
                grad_func,
                **grad_func_paral_kwargs)
            resi_train_val, pred_train_val = residual(X_train_val, y_train_val, model)
            abs_resi_train_val = np.absolute(resi_train_val)
            dev_train_val = dev(pred_train_val, y_train_val)
        else:
            (grads_train_val, pred_train_val, resi_train_val,
             abs_resi_train_val, dev_train_val) = (
                np.array([]), np.array([]),
                np.array([]), np.array([]),
                np.array([]))

        # Phase-I
        t0 = time.time()
        print("Calculate scores for Phase-I...")
        grads_PI = grad_func_batch_paral(
            X_PI,
            y_PI,
            Build_Model, 
            build_model_param,
            grad_func,
            **grad_func_paral_kwargs)
        # grads_PI = []
        # for i in range(X_PI.shape[0]):
        #     # Have to be a float tensor
        #     x = tf.convert_to_tensor(X_PI[[i]], dtype=tf.float32)
        #     y = tf.convert_to_tensor(y_PI[[i]])
        #     grad_raw = grad_loss(model, x, y)
        #     if hidden_layer_sizes:
        #         grad_np_vec = np.array([])
        #         for j in range(len(hidden_layer_sizes) + 1):
        #             # Already checked the dimension of gradient.
        #             grad_np_vec = np.append(
        #                 grad_np_vec, grad_raw[2 * j][0].numpy())
        #             grad_np_vec = np.append(
        #                 grad_np_vec, grad_raw[2 * j + 1][0].numpy())
        #         grads_PI.append(grad_np_vec)
        #         # print("The sample gradient {} with length {}".format(grads_PI[-1], grads_PI[-1].shape))
        #     else:
        #         grads_PI.append(
        #             np.append(grad_raw[0][0].numpy(),
        #                       grad_raw[1][0].numpy()))
        logger.info(
            "The calculation for Phase-I takes %s seconds.",
            (time.time() - t0),
            extra=d)

        fisher_info_mat_PI = fisher_mat_score_cov(grads_PI, fisher_nugget=penal_param, penal_matrix=penal_matrix)
        mu_PI = np.mean((-1) * grads_PI, axis=0)
        Sinv_PI = Inv_Cov((-1) * grads_PI, FLAGS.nugget)

        resi_PI, pred_PI = residual(X_PI, y_PI, model)
        abs_resi_PI = np.absolute(resi_PI)
        dev_PI = dev(pred_PI, y_PI)
    else:
        (mu_train, Sinv_train, fisher_info_mat_train, 
         mu_PI, Sinv_PI, fish_info_mat_PI, 
         pred_train, pred_train_val, pred_PI, 
         grads_train, grads_train_val, grads_PI, 
         abs_train, abs_resi_train_val, abs_resi_PI, 
         dev_train, dev_train_val, dev_PI) = (
                                 np.array([]), np.array([]),
                                 np.array([]), np.array([]),
                                 np.array([]), np.array([]),
                                 np.array([]), np.array([]),
                                 np.array([]), np.array([]),
                                 np.array([]), np.array([]),
                                 np.array([]), np.array([]),
                                 np.array([]), np.array([]),
                                 np.array([]), np.array([]))

    # Phase II
    t0 = time.time()
    if X_PII.shape[0]>0:
        print("Calculate scores for Phase-II...")
        grads_PII = grad_func_batch_paral(
            X_PII,
            y_PII,
            Build_Model, 
            build_model_param,
            grad_func,
            **grad_func_paral_kwargs)
        # grads_PII = []
        # for i in range(X_PII.shape[0]):
        #     x = tf.convert_to_tensor(X_PII[[i]], dtype=tf.float32)
        #     y = tf.convert_to_tensor(y_PII[[i]])
        #     grad_raw = grad_loss(model, x, y)
        #     if hidden_layer_sizes:
        #         grad_np_vec = np.array([])
        #         for j in range(len(hidden_layer_sizes) + 1):
        #             grad_np_vec = np.append(
        #                 grad_np_vec, grad_raw[2 * j][0].numpy())
        #             grad_np_vec = np.append(
        #                 grad_np_vec, grad_raw[2 * j + 1][0].numpy())
        #         grads_PII.append(grad_np_vec)
        #         # print("The sample gradient {} with length {}".format(grads_PI[-1], grads_PI[-1].shape))
        #     else:
        #         grads_PII.append(
        #             np.append(
        #                 grad_raw[0][0].numpy(),
        #                 grad_raw[1][0].numpy()))
        logger.info("The calculation for Phase-II takes %s seconds.", (time.time() - t0), extra=d)

        # Absolute residual
        resi_PII, pred_PII = residual(X_PII, y_PII, model)
        abs_resi_PII = np.absolute(resi_PII)
        # Deviance
        dev_PII = dev(pred_PII, y_PII)

        # Calculate cumulative absolute residual
        # Currently the cumulative statistics is not really useful. The reason
        # to keep it here is just to make the api easier to adapte. The returned
        # field for cumulative statistics should not be used to plot figure.
        # In one round case (not multiple generation of Phase-II with a common
        # Phase-I data, which mainly appear in simulation), the cumulative
        # statistics is correct.
        # (cum_abs_resi_PI, cum_abs_resi_PII,
        #  abs_resi_PII, abs_resi_PI_PII) = cum_abs_resi(
        #     abs_resi_PI, resi_PII)
        cum_abs_resi_PI, cum_abs_resi_PII = np.array([]), np.array([])
    else:
        grads_PII, pred_PII, resi_PII, abs_resi_PII, dev_PII, cum_abs_resi_PI, cum_abs_resi_PII = (
            np.zeros((0,grads_train_val.shape[1])), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

    # if gamma > 0:
    #     # Ewma absolute residual
    #     # ewma_err_PI_PII = Scores_ewma(np.reshape(pred_err_PI_PII, (-1,1)), gamma, np.mean(pred_err_PI_PII))
    #     ewma_abs_resi_PI_PII = Scores_ewma(np.reshape(
    #         abs_resi_PI_PII, (-1, 1)), gamma, np.mean(abs_resi_PI))
    #     ewma_abs_resi_PI_PII = np.reshape(ewma_abs_resi_PI_PII, (-1,))
    #     ewma_abs_resi_PI = ewma_abs_resi_PI_PII[:len(abs_resi_PI)]
    #     ewma_abs_resi_PII = ewma_abs_resi_PI_PII[len(abs_resi_PI):]

    #     # Ewma deviance
    #     # ewma_dev_PI_PII = Scores_ewma(np.reshape(dev_PI_PII, (-1,1)), gamma, np.mean(dev_PI_PII))
    #     ewma_dev_PI_PII = Scores_ewma(np.reshape(
    #         dev_PI_PII, (-1, 1)), gamma, np.mean(dev_PI))
    #     ewma_dev_PI_PII = np.reshape(ewma_dev_PI_PII, (-1,))
    #     ewma_dev_PI = ewma_dev_PI_PII[:len(dev_PI)]
    #     ewma_dev_PII = ewma_dev_PI_PII[len(dev_PI):]

    #     return (model, mu_train, Sinv_train, pred_PI, pred_PII,
    #             grads_PI, grads_PII, fisher_info_mat_train,
    #             cum_abs_resi_PI, cum_abs_resi_PII,
    #             ewma_abs_resi_PI, ewma_abs_resi_PII,
    #             ewma_dev_PI, ewma_dev_PII, best_cv_param)
    # else:
    if normal_flag:
        pred_PI, pred_PII = resp_scaler.inverse_transform(pred_PI.reshape((-1,1))).reshape((-1,)), resp_scaler.inverse_transform(pred_PII.reshape((-1,1))).reshape((-1,))
    return (model, model_ckpt_fname, mu_train, Sinv_train, mu_PI, Sinv_PI, 
            pred_train.reshape((-1,)), pred_train_val.reshape((-1,)), pred_PI.reshape((-1,)), pred_PII.reshape((-1,)), 
            grads_train, grads_train_val, grads_PI, grads_PII, fisher_info_mat_train,
            fisher_info_mat_PI, cum_abs_resi_PI.reshape((-1,)), cum_abs_resi_PII.reshape((-1,)),
            abs_resi_train.reshape((-1,)), abs_resi_train_val.reshape((-1,)), abs_resi_PI.reshape((-1,)), abs_resi_PII.reshape((-1,)), 
            dev_train.reshape((-1,)), dev_train_val.reshape((-1,)), dev_PI.reshape((-1,)), dev_PII.reshape((-1,)), best_cv_param)


# def Linear_Model(
#         X_train,
#         y_train,
#         X_PI,
#         y_PI,
#         X_PII,
#         y_PII,
#         penal_param,
#         nugget=0):
#     alphas = 10.0**np.arange(-4, 1)  # The penalization parameter to try.
#     # The cross-validation gives the best Ridge parameter as 0.1.
#     reg_ridge_cv = RidgeCV(alphas=alphas)
#     reg_ridge_cv.fit(X_train, y_train)
#     logger.info(
#         "The Ridge cv coef: {};\nThe intercept: {};\nThe penalization param: {}.\n".format(
#             reg_ridge_cv.coef_,
#             reg_ridge_cv.intercept_,
#             reg_ridge_cv.alpha_),
#         extra=d)
#     logger.info("The score for Ridge cv: {}\n".format(
#         reg_ridge_cv.score(X_train, y_train),), extra=d)

#     # The cross-validation gives the best Lasso parameter as 1e-6.
#     reg_lasso_cv = LassoCV(alphas=alphas, max_iter=200000)
#     reg_lasso_cv.fit(X_train, y_train)
#     logger.info(
#         "The Lasso cv coef: {};\nThe intercept: {};\nThe penalization param: {}.\n".format(
#             reg_lasso_cv.coef_,
#             reg_lasso_cv.intercept_,
#             reg_lasso_cv.alpha_),
#         extra=d)
#     logger.info("The score for Lasso cv: {}\n".format(
#         reg_lasso_cv.score(X_train, y_train),), extra=d)

#     # The cross-validation gives the best ElasticNet parameter as 1e-4.
#     reg_elasticnet_cv = ElasticNetCV(alphas=alphas, max_iter=200000)
#     reg_elasticnet_cv.fit(X_train, y_train)
#     logger.info(
#         "The ElasticNet cv coef: {};\nThe intercept: {};\nThe penalization param: {}.\n".format(
#             reg_elasticnet_cv.coef_,
#             reg_elasticnet_cv.intercept_,
#             reg_elasticnet_cv.alpha_),
#         extra=d)
#     logger.info("The score for ElassticNet cv: {}\n".format(
#         reg_elasticnet_cv.score(X_train, y_train),), extra=d)

#     # Model actually used.
#     # reg_model = Lasso(alpha=10**(-6), max_iter=20000)
#     reg_model = Ridge(alpha=penal_param)
#     reg_model.fit(X_train, y_train)
#     logger.info("The final coef: {};\nThe intercept: {}.\n".format(
#         reg_model.coef_, reg_model.intercept_), extra=d)
#     pred_train = reg_model.predict(X_train)

#     # Calculate prediction for Phase I and Phase II
#     param = np.array(np.append(reg_model.coef_, reg_model.intercept_), ndmin=2)
#     # Phase I
#     pred_y_PI = np.vstack(reg_model.predict(X_PI))
#     design_mat = np.append(X_PI, np.ones([X_PI.shape[0], 1]), axis=1)
#     # print design_mat
#     grads_PI = np.vstack(pred_y_PI - y_PI) * design_mat
#     fisher_info_mat_PI = - \
#         np.matmul(design_mat.T, design_mat) / design_mat.shape[0]
#     logger.info('Condition for Fisher Info Mat#: %s',
#                 np.linalg.cond(fisher_info_mat_PI), extra=d)
#     fisher_info_mat_PI = (fisher_info_mat_PI - nugget * np.diag(np.repeat(1.0, design_mat.shape[1]), 0))
#     logger.info('Condition for Fisher Info Mat#: %s',
#                 np.linalg.cond(fisher_info_mat_PI), extra=d)

#     # Phase II
#     pred_y_PII = np.vstack(reg_model.predict(X_PII))
#     grads_PII = np.vstack(pred_y_PII - y_PII) * \
#         np.append(X_PII, np.ones([X_PII.shape[0], 1]), axis=1)

#     return grads_PI, grads_PII, fisher_info_mat_PI


class Linear_Reg(object):
    def __init__(self, X_train, y_train, penal_param, train_PI_flag, model_time_stamp, FLAGS):
        self.penal_param = penal_param
        self.train_PI_flag = train_PI_flag
        self.model_time_stamp = model_time_stamp
        self.FLAGS = FLAGS

        # # Define those function in __init__ so that they can be past into joblib function for parallelization.
        # # Inside those functions, call other functions directly without adding 'self.' prior to that.
        # def penalMatrix(dim):
        #     """ No penalty on intercept and intercept is the last column of X. """
        #     penal_matrix = np.identity(dim)
        #     penal_matrix[dim-1, dim-1] = 0
        #     return penal_matrix

        # def calGradient(reg, X, y, penal_param=0):
        #     pred = reg.predict(X)
        #     design_mat = np.append(X, np.ones([X.shape[0], 1]), axis=1)
        #     # Notice that we don't penalize intercept in the linear regression in
        #     # scikit-learn.
        #     grads = np.vstack(pred - y) * design_mat + penal_param * np.append(reg.coef_, 0)
        #     return pred, grads

        # def calGradientFisher(reg, X, y, fisher_nugget=0):
        #     pred, grads = calGradient(reg, X, y)
        #     # Notice that the intercept is appended at the last column of X.
            
        #     # Use analytic formulation for Fisher information matrix
        #     design_mat = np.append(X, np.ones([X.shape[0], 1]), axis=1)
        #     # Theoretical fisher information matrix (2nd order derivative).
        #     # The following coefficient 2.0 is critical in calculating fisher
        #     # information matrix when we have l2 regularization parameter.
        #     fisher_info_mat = 2.0 * np.matmul(design_mat.T, design_mat) / design_mat.shape[0]

        #     # # # Use covariance of score vectors to approximate Fisher information matrix
        #     # fisher_info_mat = fisher_mat_score_cov(grads)

        #     logger.info('Condition # for Fisher Info Mat (before adding fisher nugget): %s',
        #                 np.linalg.cond(fisher_info_mat), extra=d)
        #     if fisher_nugget > 0:
        #         # Be careful the sign of Fisher Information Matrix.
        #         fisher_info_mat = (fisher_info_mat + fisher_nugget * penalMatrix(fisher_info_mat.shape[1]))
        #         logger.info('Condition # for Fisher Info Mat (after adding penalization nugget): %s',
        #                     np.linalg.cond(fisher_info_mat), extra=d)
        #     return pred, grads, fisher_info_mat

        # def fisher_mat_score_cov(grads, fisher_nugget=0, penal_matrix=None):
        #     # The following coefficient 1.0 is critical in calculating fisher
        #     # information matrix when we have l2 regularization parameter.
        #     grads_mu = np.mean(grads, axis=0)
        #     grads_centered = grads - grads_mu
        #     fisher_info_mat = 1.0*np.matmul(grads_centered.T, grads_centered)/grads.shape[0]

        #     if fisher_nugget > 0:
        #         # Be careful the sign of Fisher Information Matrix.
        #         logger.info("The penalization parameter is %s\n", fisher_nugget, extra=d)
        #         logger.info('Condition # for Fisher Info Mat before including penalization: %s',
        #                     np.linalg.cond(fisher_info_mat), extra=d)
        #         fisher_info_mat = (fisher_info_mat + fisher_nugget * penal_matrix)
        #         logger.info('Condition # for Fisher Info Mat after including penalization: %s',
        #                     np.linalg.cond(fisher_info_mat), extra=d)
        #     return fisher_info_mat

        # def calDev(pred, y):
        #     dev = (y - pred)**2 # Defined as -ln(Likelihood)
        #     return dev

        # def calCumErr(abs_resi_PI, pred_PII, y_PII):
        #     abs_resi_PII = np.absolute(y_PII - pred_PII)
        #     abs_resi_PI_PII = np.hstack((abs_resi_PI, abs_resi_PII))
        #     cum_abs_resi_PI_PII = 1.0 * np.cumsum(abs_resi_PI_PII) / \
        #         (1.0 + np.arange(len(abs_resi_PI_PII)))
        #     cum_abs_resi_PI = cum_abs_resi_PI_PII[:len(abs_resi_PI)]
        #     cum_abs_resi_PII = cum_abs_resi_PI_PII[len(abs_resi_PI):]
        #     return (cum_abs_resi_PI, cum_abs_resi_PII,
        #             abs_resi_PII, abs_resi_PI_PII)

        # # Define those functions as member objects so that other class method can call them and they can be past as parameter to joblib for parallelization.
        # self.penalMatrix = penalMatrix
        # self.calGradient = calGradient
        # self.calGradientFisher = calGradientFisher
        # self.fisher_mat_score_cov = fisher_mat_score_cov
        # self.calDev = calDev

        # The training workflow starts
        print("Input shape")
        print(X_train.shape,y_train.shape)
        # alphas = 10.0**np.arange(-4, 1)  # The penalization parameter to try.
        # The cross-validation gives the best Ridge parameter as 0.1.
        reg_ridge_cv = RidgeCV(alphas=10.0**np.arange(-8, 9))
        print(np.where(np.isnan(X_train)), np.where(np.isnan(y_train)))
        reg_ridge_cv.fit(X_train, y_train)
        logger.info(
            "The Ridge cv coef: {};\nThe intercept: {};\nThe penalization param: {}.\n".format(
                reg_ridge_cv.coef_,
                reg_ridge_cv.intercept_,
                reg_ridge_cv.alpha_),
            extra=d)
        logger.info("The score for training Ridge cv: {}\n".format(
            reg_ridge_cv.score(X_train, y_train),), extra=d)
        self.reg_ridge_cv = reg_ridge_cv

        if self.train_PI_flag:
            self.reg = Ridge(alpha=reg_ridge_cv.alpha_, max_iter=20000)
            self.reg.fit(X_train, y_train)
            param = np.array(np.append(self.reg.coef_, self.reg.intercept_), ndmin=2)
            logger.info(
                "The parameter of ridge regression model (%s) is\n %s",
                self.model_time_stamp,
                param,
                extra=d)

            pickle.dump(self.reg, open(os.path.join(FLAGS.training_res_folder,
                                            '_'.join([self.model_time_stamp, FLAGS.best_model])), 'wb'))
        
        self.reg = pickle.load(open(os.path.join(FLAGS.training_res_folder, '_'.join([self.model_time_stamp, FLAGS.best_model])), 'rb'))

        param = np.array(np.append(self.reg.coef_, self.reg.intercept_), ndmin=2)
        FLAGS.lin_reg_param = param
        logger.info(
            "The parameter of reloaded ridge regression model (%s) is\n %s",
            self.model_time_stamp,
            param,
            extra=d)

    
    def cal_metrics(self, X, y, data_info='train'):
        """
            X: Predictor matrix.
            y: In shape (-1,).
        
        """
        from regression.regressors_lin_utils import calGradientFisher, calDev
        pred, grads, fisher_info_mat = calGradientFisher(self.reg, X, y, fisher_nugget=self.penal_param)

        mu = np.mean(-grads, axis=0) # Score mean
        Sinv = Inv_Cov(-grads, self.FLAGS.nugget) # Score variance

        resi = y-pred
        abs_resi = np.absolute(resi)
        dev = calDev(pred, y)

        if data_info == 'train':
            self.FLAGS.mu_train, self.FLAGS.Sinv_train, self.FLAGS.fisher_info_mat_train = mu, Sinv, fisher_info_mat
            logger.info("The training score mean is {}.".format(self.FLAGS.mu_train), extra=d)
            logger.info("The training score Sinv is {}.".format(self.FLAGS.Sinv_train), extra=d)
            self.FLAGS.best_r2_train = r2_score(y, pred)
        elif data_info == 'PI':
            self.FLAGS.mu_PI, self.FLAGS.Sinv_PI, self.FLAGS.fisher_info_mat_PI = mu, Sinv, fisher_info_mat
            # self.FLAGS.mu_train = self.FLAGS.mu_PI
            logger.info("The PI score mean is {}.".format(self.FLAGS.mu_PI), extra=d)
            logger.info("The PI score Sinv is {}.".format(self.FLAGS.Sinv_PI), extra=d)

        logger.info("The R-squared for {} data set: {}\n".format(data_info, r2_score(y, pred)), extra=d)

        return fisher_info_mat, pred.reshape((-1,)), grads, resi.reshape((-1,)), abs_resi.reshape((-1,)), dev.reshape((-1,))


def Linear_Model_Cum_ewma_resi_ewma_dev(
        X_train,
        y_train,
        X_PI,
        y_PI,
        X_PII,
        y_PII,
        N_PIIs,
        gamma, # EWMA parameter, currently it is always 0.
        start,
        penal_param,
        train_PI_flag,
        model_time_stamp,
        FLAGS):
    print("Input shape")
    print(X_train.shape,y_train.shape,X_PI.shape,y_PI.shape,X_PII.shape,y_PII.shape)
    # alphas = 10.0**np.arange(-4, 1)  # The penalization parameter to try.
    # The cross-validation gives the best Ridge parameter as 0.1.
    reg_ridge_cv = RidgeCV(alphas=10.0**np.arange(-8, 9))
    print(np.where(np.isnan(X_train)), np.where(np.isnan(y_train)))
    reg_ridge_cv.fit(X_train, y_train)
    logger.info(
        "The Ridge cv coef: {};\nThe intercept: {};\nThe penalization param: {}.\n".format(
            reg_ridge_cv.coef_,
            reg_ridge_cv.intercept_,
            reg_ridge_cv.alpha_),
        extra=d)
    logger.info("The score for training Ridge cv: {}\n".format(
        reg_ridge_cv.score(X_train, y_train),), extra=d)
    #
    # # The cross-validation gives the best Lasso parameter as 1e-6.
    # reg_lasso_cv = LassoCV(alphas=alphas, max_iter=200000)
    # reg_lasso_cv.fit(X_train, y_train)
    # logger.info(
    #     "The Lasso cv coef: {};\nThe intercept: {};\nThe penalization param: {}.\n".format(
    #         reg_lasso_cv.coef_,
    #         reg_lasso_cv.intercept_,
    #         reg_lasso_cv.alpha_),
    #     extra=d)
    # logger.info("The score for Lasso cv: {}\n".format(
    #     reg_lasso_cv.score(X_train, y_train),), extra=d)
    #
    # # The cross-validation gives the best ElasticNet parameter as 1e-4.
    # reg_elasticnet_cv = ElasticNetCV(alphas=alphas, max_iter=200000)
    # reg_elasticnet_cv.fit(X_train, y_train)
    # logger.info(
    #     "The ElasticNet cv coef: {};\nThe intercept: {};\nThe penalization param: {}.\n".format(
    #         reg_elasticnet_cv.coef_,
    #         reg_elasticnet_cv.intercept_,
    #         reg_elasticnet_cv.alpha_),
    #     extra=d)
    # logger.info("The score for ElassticNet cv: {}\n".format(
    #     reg_elasticnet_cv.score(X_train, y_train),), extra=d)

    if train_PI_flag:
        # According to the source code of sklearn
        # https://github.com/scikit-learn/scikit-learn/blob/b7b4d3e2f/sklearn/linear_model/ridge.py#L586
        # l-135 and l-591, we know that the penalization term is added to the
        # sum of error squared, not sample mean of error squared. It is important
        # in calculating the Fisher information matrix when score function includes
        # the penalization term.
        # For this function, the standard loss should be following.
        #           1/n \sum_i (-log-likelihood_i + 0.5 * penal_param * w^T * w)
        # Then, the modification of the Fisher information matrix is just to
        # add an identity matrix times penal_param (for sample mean version).

        # From the source code https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/linear_model/ridge.py#L586
        # line 554 and line 561, 581, we can see that in the package, scikit-learn
        # fit ridge regression without penalizing the intercept terms. It recovers
        # the intercept seperately after fitting ridge regression (as in
        # https://github.com/scikit-learn/scikit-learn/blob/7813f7efb5b2012412888b69e73d76f2df2b50b6/sklearn/linear_model/base.py
        # line 225). So the mean of targets equals the intercept, even though
        # we use ridge regression. If we also penalize the intercept, we wouldn't
        # have this property.

        # The good news is that we don't need to do centering ourselives, when
        # calling scikit-learn linear regression function.

        # reg = Ridge(alpha=0.5*X_train.shape[0]*penal_param, max_iter=20000)
        # reg = Ridge(alpha=penal_param, max_iter=20000)
        reg = Ridge(alpha=reg_ridge_cv.alpha_, max_iter=20000)
        reg.fit(X_train, y_train)
        param = np.array(np.append(reg.coef_, reg.intercept_), ndmin=2)
        logger.info(
            "The parameter of ridge regression model (%s) is\n %s",
            model_time_stamp,
            param,
            extra=d)

        pickle.dump(reg, open(os.path.join(FLAGS.training_res_folder,
                                           '_'.join([model_time_stamp, FLAGS.best_model])), 'wb'))

    reg = pickle.load(open(os.path.join(FLAGS.training_res_folder, '_'.join(
        [model_time_stamp, FLAGS.best_model])), 'rb'))

    param = np.array(np.append(reg.coef_, reg.intercept_), ndmin=2)
    FLAGS.lin_reg_param = param
    logger.info(
        "The parameter of reloaded ridge regression model (%s) is\n %s",
        model_time_stamp,
        param,
        extra=d)

    def penalMatrix(dim):
        """ No penalty on intercept and intercept is the last column of X. """
        penal_matrix = np.identity(dim)
        penal_matrix[dim-1, dim-1] = 0
        return penal_matrix

    def calGradient(reg, X, y, penal_param=0):
        pred = reg.predict(X)
        design_mat = np.append(X, np.ones([X.shape[0], 1]), axis=1)
        # Notice that we don't penalize intercept in the linear regression in
        # scikit-learn.
        grads = np.vstack(pred - y) * design_mat + penal_param * np.append(reg.coef_, 0)
        return pred, grads

    def calGradientFisher(reg, X, y, fisher_nugget=0):
        pred, grads = calGradient(reg, X, y)
        # Notice that the intercept is appended at the last column of X.
        
        # Use analytic formulation for Fisher information matrix
        design_mat = np.append(X, np.ones([X.shape[0], 1]), axis=1)
        # Theoretical fisher information matrix (2nd order derivative).
        # The following coefficient 2.0 is critical in calculating fisher
        # information matrix when we have l2 regularization parameter.
        fisher_info_mat = 2.0 * np.matmul(design_mat.T, design_mat) / design_mat.shape[0]

        # # # Use covariance of score vectors to approximate Fisher information matrix
        # fisher_info_mat = fisher_mat_score_cov(grads)

        logger.info('Condition # for Fisher Info Mat (before adding fisher nugget): %s',
                    np.linalg.cond(fisher_info_mat), extra=d)
        if fisher_nugget > 0:
            # Be careful the sign of Fisher Information Matrix.
            fisher_info_mat = (fisher_info_mat +
                fisher_nugget * penalMatrix(fisher_info_mat.shape[1]))
            logger.info('Condition # for Fisher Info Mat (after adding penalization nugget): %s',
                        np.linalg.cond(fisher_info_mat), extra=d)
        return pred, grads, fisher_info_mat

    def fisher_mat_score_cov(grads, fisher_nugget=0, penal_matrix=None):
        # The following coefficient 1.0 is critical in calculating fisher
        # information matrix when we have l2 regularization parameter.
        grads_mu = np.mean(grads, axis=0)
        grads_centered = grads - grads_mu
        fisher_info_mat = 1.0*np.matmul(grads_centered.T, grads_centered)/grads.shape[0]

        if fisher_nugget > 0:
            # Be careful the sign of Fisher Information Matrix.
            logger.info("The penalization parameter is %s\n", fisher_nugget, extra=d)
            logger.info('Condition # for Fisher Info Mat before including penalization: %s',
                        np.linalg.cond(fisher_info_mat), extra=d)
            fisher_info_mat = (fisher_info_mat + fisher_nugget * penal_matrix)
            logger.info('Condition # for Fisher Info Mat after including penalization: %s',
                        np.linalg.cond(fisher_info_mat), extra=d)
        return fisher_info_mat

    def calDev(pred, y):
        dev = (y - pred)**2 # Defined as -ln(Likelihood)
        return dev

    def calCumErr(abs_resi_PI, pred_PII, y_PII):
        abs_resi_PII = np.absolute(y_PII - pred_PII)
        abs_resi_PI_PII = np.hstack((abs_resi_PI, abs_resi_PII))
        cum_abs_resi_PI_PII = 1.0 * np.cumsum(abs_resi_PI_PII) / \
            (1.0 + np.arange(len(abs_resi_PI_PII)))
        cum_abs_resi_PI = cum_abs_resi_PI_PII[:len(abs_resi_PI)]
        cum_abs_resi_PII = cum_abs_resi_PI_PII[len(abs_resi_PI):]
        return (cum_abs_resi_PI, cum_abs_resi_PII,
                abs_resi_PII, abs_resi_PI_PII)

    if train_PI_flag: # Need to train model.
        # Calculate mean and covariance matrix of the score function for
        # training data.
        # Notice that previously the fisher_nugget equals to penal_param.
        pred_train, grads_train, fisher_info_mat_train = calGradientFisher(reg,
            X_train, y_train, fisher_nugget = penal_param)

        FLAGS.best_r2_val = r2_score(y_train, pred_train)

        mu_train = np.mean(-grads_train, axis=0) # Score mean
        Sinv_train = Inv_Cov(-grads_train, FLAGS.nugget) # Score variance

        abs_resi_train = np.absolute(y_train - pred_train)
        dev_train = calDev(pred_train, y_train)

        # Phase-I
        (pred_PI, grads_PI, fisher_info_mat_PI) = calGradientFisher(reg,
            X_PI, y_PI, fisher_nugget = penal_param)

        mu_PI = np.mean(- grads_PI, axis=0)
        Sinv_PI = Inv_Cov(- grads_PI, FLAGS.nugget)

        abs_resi_PI = np.absolute(y_PI - pred_PI)
        dev_PI = calDev(pred_PI, y_PI)

        logger.info("The score for testing PI Ridge cv: {}\n".format(
            reg.score(X_PI, y_PI),), extra=d)
    else:
        (mu_train, Sinv_train, fisher_info_mat_train, 
         mu_PI, Sinv_PI, fisher_info_mat_PI,
         pred_train, pred_PI, grads_train, grads_PI, 
         abs_resi_train, abs_resi_PI, dev_train, dev_PI) = (
                                 np.array([]), np.array([]), np.array([]),
                                 np.zeros([]), np.array([]), np.array([]),
                                 np.array([]), np.array([]), np.array([]),
                                 np.array([]), np.array([]), np.array([]),
                                 np.array([]), np.array([]))

    # Phase-II
    if X_PII.shape[0] > 0:

        # Absolute residual
        pred_PII, grads_PII = calGradient(reg, X_PII, y_PII)
        abs_resi_PII = np.abs(y_PII - pred_PII)
        # Deviance
        dev_PII = calDev(pred_PII, y_PII)

        logger.info("The score for testing PII Ridge cv: {}\n".format(
                reg.score(X_PII, y_PII),), extra=d)

        # Cumulative error rate
        # Currently the cumulative statistics is not really useful. The reason
        # to keep it here is just to make the api easier to adapte. The returned
        # field for cumulative statistics should not be used to plot figure.
        # In one round case (not multiple generation of Phase-II with a common
        # Phase-I data, which mainly appear in simulation), the cumulative
        # statistics is correct.
        (cum_abs_resi_PI, cum_abs_resi_PII,
        abs_resi_PII, abs_resi_PI_PII) = calCumErr(
            abs_resi_PI, pred_PII, y_PII)
    else:
        grads_PII, pred_PII, abs_resi_PII, dev_PII , abs_resi_PI_PII, cum_abs_resi_PI, cum_abs_resi_PII= (
            np.zeros((0, grads_train.shape[1])), np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]), np.array([]))

    # if gamma > 0:
    #     # Ewma residuals
    #     # ewma_abs_resi_PI_PII = Scores_ewma(np.reshape(abs_resi_PI_PII, (-1,1)), gamma, np.mean(abs_resi_PI_PII))
    #     ewma_abs_resi_PI_PII = Scores_ewma(np.reshape(
    #         abs_resi_PI_PII, (-1, 1)), gamma, np.mean(abs_resi_PI))
    #     ewma_abs_resi_PI_PII = np.reshape(ewma_abs_resi_PI_PII, (-1,))
    #     ewma_abs_resi_PI = ewma_abs_resi_PI_PII[:len(abs_resi_PI)]
    #     ewma_abs_resi_PII = ewma_abs_resi_PI_PII[len(abs_resi_PI):]

    #     # Ewma deviance
    #     # ewma_dev_PI_PII = Scores_ewma(np.reshape(dev_PI_PII, (-1,1)), gamma, np.mean(dev_PI_PII))
    #     ewma_dev_PI_PII = Scores_ewma(np.reshape(
    #         dev_PI_PII, (-1, 1)), gamma, np.mean(dev_PI))
    #     ewma_dev_PI_PII = np.reshape(ewma_dev_PI_PII, (-1,))
    #     ewma_dev_PI = ewma_dev_PI_PII[:len(dev_PI)]
    #     ewma_dev_PII = ewma_dev_PI_PII[len(dev_PI):]

    #     return (mu_train, Sinv_train, pred_PI, pred_PII, grads_PI, grads_PII,
    #             fisher_info_mat_train, cum_abs_resi_PI, cum_abs_resi_PII,
    #             ewma_abs_resi_PI, ewma_abs_resi_PII, ewma_dev_PI, ewma_dev_PII)
    # else:
    
    # Individual residuals
    # Transfer the prediction back
    # print(grads_train.shape,grads_PI.shape,grads_PII)
    return (reg, mu_train, Sinv_train, mu_PI, Sinv_PI, 
            pred_train.reshape((-1,)), pred_PI.reshape((-1,)), pred_PII.reshape((-1,)), 
            grads_train, grads_PI, grads_PII,
            fisher_info_mat_train, fisher_info_mat_PI, 
            cum_abs_resi_PI.reshape((-1,)), cum_abs_resi_PII.reshape((-1,)),
            abs_resi_train.reshape((-1,)), abs_resi_PI.reshape((-1,)), abs_resi_PII.reshape((-1,)), 
            dev_train.reshape((-1,)), dev_PI.reshape((-1,)), dev_PII.reshape((-1,)))


# def Linear_Model_Cust_Grad(
#         X_train,
#         y_train,
#         X_PI,
#         y_PI,
#         X_PII,
#         y_PII,
#         X_rest_PI,
#         X_rest_PII,
#         penal_param,
#         nugget=0):
#     """Calculate customized gradient based on entire covariates for Phase-I and Phase-II."""

#     reg_model = Lasso(alpha=penal_param, max_iter=20000)
#     reg_model.fit(X_train, y_train)
#     pred_train = reg_model.predict(X_train)

#     # Calculate prediction for Phase I and Phase II
#     param = np.array(np.append(reg_model.coef_, reg_model.intercept_), ndmin=2)
#     # Phase I
#     pred_y_PI = reg_model.predict(X_PI)
#     design_mat = np.append(np.append(X_PI, np.ones(
#         [X_PI.shape[0], 1]), axis=1), X_rest_PI, axis=1)
#     grads_PI = np.vstack(pred_y_PI - y_PI) * design_mat
#     fisher_info_mat_PI = np.matmul(design_mat.T, design_mat) / design_mat.shape[0]
#     logger.info('Condition # for Fisher Info Mat: %s',
#                 np.linalg.cond(fisher_info_mat_PI), extra=d)
#     fisher_info_mat_PI = (
#         fisher_info_mat_PI +
#         nugget *
#         np.diag(
#             np.repeat(
#                 1.0,
#                 design_mat.shape[1]),
#             0))
#     logger.info('Condition for Fisher Info Mat#: %s',
#                 np.linalg.cond(fisher_info_mat_PI), extra=d)

#     # Phase II
#     pred_y_PII = reg_model.predict(X_PII)
#     grads_PII = (np.vstack(pred_y_PII - y_PII) * np.append(np.append(X_PII,
#                                                                      np.ones([X_PII.shape[0], 1]), axis=1), X_rest_PII, axis=1))

#     return grads_PI, grads_PII


def Poisson_Model_Cum_ewma_resi_ewma_dev(
        X_train,
        y_train,
        X_PI,
        y_PI,
        X_PII,
        y_PII,
        N_PIIs,
        gamma,
        start,
        penal_param,
        train_PI_flag,
        model_time_stamp,
        FLAGS):
    # alphas = 10.0**np.arange(-4, 1)  # The penalization parameter to try.
    # # The cross-validation gives the best Ridge parameter as 0.1.
    # reg_ridge_cv = RidgeCV(alphas=alphas)
    # reg_ridge_cv.fit(X_train, y_train)
    # logger.info(
    #     "The Ridge cv coef: {};\nThe intercept: {};\nThe penalization param: {}.\n".format(
    #         reg_ridge_cv.coef_,
    #         reg_ridge_cv.intercept_,
    #         reg_ridge_cv.alpha_),
    #     extra=d)
    # logger.info("The score for Ridge cv: {}\n".format(
    #     reg_ridge_cv.score(X_train, y_train),), extra=d)
    #
    # # The cross-validation gives the best Lasso parameter as 1e-6.
    # reg_lasso_cv = LassoCV(alphas=alphas, max_iter=200000)
    # reg_lasso_cv.fit(X_train, y_train)
    # logger.info(
    #     "The Lasso cv coef: {};\nThe intercept: {};\nThe penalization param: {}.\n".format(
    #         reg_lasso_cv.coef_,
    #         reg_lasso_cv.intercept_,
    #         reg_lasso_cv.alpha_),
    #     extra=d)
    # logger.info("The score for Lasso cv: {}\n".format(
    #     reg_lasso_cv.score(X_train, y_train),), extra=d)
    #
    # # The cross-validation gives the best ElasticNet parameter as 1e-4.
    # reg_elasticnet_cv = ElasticNetCV(alphas=alphas, max_iter=200000)
    # reg_elasticnet_cv.fit(X_train, y_train)
    # logger.info(
    #     "The ElasticNet cv coef: {};\nThe intercept: {};\nThe penalization param: {}.\n".format(
    #         reg_elasticnet_cv.coef_,
    #         reg_elasticnet_cv.intercept_,
    #         reg_elasticnet_cv.alpha_),
    #     extra=d)
    # logger.info("The score for ElassticNet cv: {}\n".format(
    #     reg_elasticnet_cv.score(X_train, y_train),), extra=d)

    if train_PI_flag:
        X_train_ext = sm.add_constant(X_train, prepend=False)
        poisson_mod = sm.Poisson(y_train, X_train_ext)
        poisson_reg = poisson_mod.fit(method="newton")
        logger.info(
            "The parameter of poisson regression model (%s) is\n %s",
            model_time_stamp,
            poisson_reg.params,
            extra=d)

        pickle.dump(poisson_reg, open(os.path.join(FLAGS.training_res_folder,
                                           '_'.join([model_time_stamp, FLAGS.best_model])), 'wb'))

    poisson_reg = pickle.load(open(os.path.join(FLAGS.training_res_folder, '_'.join(
        [model_time_stamp, FLAGS.best_model])), 'rb'))

    logger.info(
        "The parameter of reloaded poisson regression model (%s) is\n %s",
        model_time_stamp,
        poisson_reg.params,
        extra=d)

    def penalMatrix(dim):
        """ No penalty on intercept and intercept is the last column of X. """
        penal_matrix = np.identity(dim)
        penal_matrix[dim-1, dim-1] = 0
        return penal_matrix

    def calGradient(reg, X, y):
        # Refer to the following page for the definition of score.
        # https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.Poisson.score.html#statsmodels.discrete.discrete_model.Poisson.score
        pred = reg.predict(X)
        design_mat = np.append(X, np.ones([X.shape[0], 1]), axis=1)
        grads = np.vstack(pred - y) * design_mat
        return pred, grads

    def calGradientFisher(reg, X, y, fisher_nugget=0):
        pred, grads = calGradient(reg, X, y)
        design_mat = np.append(X, np.ones([X.shape[0], 1]), axis=1)
        fisher_info_mat = 1.0 * np.matmul(design_mat.T, np.reshape(pred, (-1,1))*design_mat) / design_mat.shape[0]
        logger.info('Condition # for Fisher Info Mat (before adding fisher nugget): %s',
                    np.linalg.cond(fisher_info_mat), extra=d)
        if fisher_nugget > 0:
            # Be careful the sign of Fisher Information Matrix.
            fisher_info_mat = (fisher_info_mat +
                fisher_nugget * penalMatrix(design_mat.shape[1]))
            logger.info('Condition # for Fisher Info Mat (after adding fisher nugget): %s',
                        np.linalg.cond(fisher_info_mat), extra=d)
        return pred, grads, fisher_info_mat

    def calDev(pred, y):
        lg_factor_y = np.array([np.sum(np.log(np.arange(1, v+1))) for v in y])
        dev = pred-y*np.log(pred)+lg_factor_y
        return dev

    def calCumErr(abs_resi_PI, pred_PII, y_PII):
        abs_resi_PII = np.absolute(y_PII - pred_PII)
        abs_resi_PI_PII = np.hstack((abs_resi_PI, abs_resi_PII))
        cum_abs_resi_PI_PII = 1.0 * np.cumsum(abs_resi_PI_PII) / \
            (1.0 + np.arange(len(abs_resi_PI_PII)))
        cum_abs_resi_PI = cum_abs_resi_PI_PII[:len(abs_resi_PI)]
        cum_abs_resi_PII = cum_abs_resi_PI_PII[len(abs_resi_PI):]
        return (cum_abs_resi_PI, cum_abs_resi_PII,
                abs_resi_PII, abs_resi_PI_PII)

    if train_PI_flag: # Need to train model.
        # Calculate mean and covariance matrix of the score function for
        # training data.
        (pred_train, grads_train,
         fisher_info_mat_train) = calGradientFisher(
            poisson_reg, X_train, y_train, fisher_nugget=penal_param)

        mu_train = np.mean(-grads_train, axis=0)
        Sinv_train = Inv_Cov(-grads_train, FLAGS.nugget)

        # Phase-I
        pred_PI, grads_PI = calGradient(poisson_reg, X_PI, y_PI)
        abs_resi_PI = np.absolute(y_PI - pred_PI)
        dev_PI = calDev(pred_PI, y_PI)
    else:
        (mu_train, Sinv_train, pred_PI, grads_PI,
         fisher_info_mat_train, abs_resi_PI, dev_PI) = (np.array([]),
                                 np.array([]), np.array([]), np.array([]),
                                 np.zeros([]), np.array([]), np.array([]))

    # Phase-II
    pred_PII, grads_PII = calGradient(poisson_reg, X_PII, y_PII)

    # Cumulative error rate
    # Currently the cumulative statistics is not really useful. The reason
    # to keep it here is just to make the api easier to adapte. The returned
    # field for cumulative statistics should not be used to plot figure.
    # In one round case (not multiple generation of Phase-II with a common
    # Phase-I data, which mainly appear in simulation), the cumulative
    # statistics is correct.
    (cum_abs_resi_PI, cum_abs_resi_PII,
     abs_resi_PII, abs_resi_PI_PII) = calCumErr(
        abs_resi_PI, pred_PII, y_PII)

    # Absolute residual
    pred_PII, grads_PII = calGradient(poisson_reg, X_PII, y_PII)
    abs_resi_PII = np.absolute(y_PII - pred_PII)
    # Deviance
    dev_PII = calDev(pred_PII, y_PII)

    if gamma > 0:
        # Ewma error rate
        # ewma_abs_resi_PI_PII = Scores_ewma(np.reshape(abs_resi_PI_PII, (-1,1)), gamma, np.mean(abs_resi_PI_PII))
        ewma_abs_resi_PI_PII = Scores_ewma(np.reshape(
            abs_resi_PI_PII, (-1, 1)), gamma, np.mean(abs_resi_PI))
        ewma_abs_resi_PI_PII = np.reshape(ewma_abs_resi_PI_PII, (-1,))
        ewma_abs_resi_PI = ewma_abs_resi_PI_PII[:len(abs_resi_PI)]
        ewma_abs_resi_PII = ewma_abs_resi_PI_PII[len(abs_resi_PI):]

        # Ewma deviance
        # ewma_dev_PI_PII = Scores_ewma(np.reshape(dev_PI_PII, (-1,1)), gamma, np.mean(dev_PI_PII))
        dev_PI_PII = np.concatenate((dev_PI, dev_PII), axis=0)
        ewma_dev_PI_PII = Scores_ewma(np.reshape(
            dev_PI_PII, (-1, 1)), gamma, np.mean(dev_PI))
        ewma_dev_PI_PII = np.reshape(ewma_dev_PI_PII, (-1,))
        ewma_dev_PI = ewma_dev_PI_PII[:len(dev_PI)]
        ewma_dev_PII = ewma_dev_PI_PII[len(dev_PI):]

        return (mu_train, Sinv_train, pred_PI, pred_PII, grads_PI, grads_PII,
                fisher_info_mat_train, cum_abs_resi_PI, cum_abs_resi_PII,
                ewma_abs_resi_PI, ewma_abs_resi_PII, ewma_dev_PI, ewma_dev_PII)
    else:
        return (mu_train, Sinv_train, pred_PI, pred_PII, grads_PI, grads_PII,
                fisher_info_mat_train, cum_abs_resi_PI, cum_abs_resi_PII,
                abs_resi_PI, abs_resi_PII, dev_PI, dev_PII)
