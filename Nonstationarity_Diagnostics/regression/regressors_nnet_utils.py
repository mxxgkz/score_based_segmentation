import os
import re
import numpy as np
import tensorflow as tf
import logging
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, r2_score

from control_chart.utils import Batch

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s(%(funcName)s)[%(lineno)d]: %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'zkg'}
logger = logging.getLogger('regressors_nnet_utils')
logging.getLogger('regressors_nnet_utils').setLevel(logging.INFO)

# Define those function in __init__ so that they can be past into joblib function for parallelization.
# Inside those functions, call other functions directly without adding 'self.' prior to that.
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

def obj_func(inputs, targets, model, loss, penal_param, wei=None, penal_type='l2'):
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

def plot_trace(trace_train, trace_val, best_index, pos, name, training_batch_size, penal_param, FLAGS, prefix='reg'):
    fig, ax = plt.subplots()
    print((len(trace_train), training_batch_size))
    ax.plot(
        np.arange(len(trace_train)) * training_batch_size,
        trace_train, 'b-', label='Training ' + name)
    ax.plot(
        np.arange(len(trace_val)) * training_batch_size,
        trace_val, 'r-', label='Validation ' + name)
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
        try:
            os.mkdir(log_path)
        except FileExistsError as err:
            logger.info(err, extra=d)

    dict_logging_files = {"loss_train": [],
                        "r2_train": [],
                        "loss_val": [],
                        "r2_val":[]}

    # for name in dict_logging_files.keys():
    #     log_file_path = os.path.join(FLAGS.training_res_folder, name+'.txt')
    #     if os.path.exists(log_file_path):
    #         os.remove(log_file_path)

    # if loss is self.loss_reg:
    #     pred = self.pred_reg
    # elif loss is self.loss_pois:
    #     pred = self.pred_pois

    round_cnt = 0 # Each round the learning rate is divided by 2

    step_per_epoch = X_train.shape[0] // training_batch_size

    training_start_time = time.time()

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

    logger.info("The neural network training took {}s.".format(time.time()-training_start_time), extra=d)

    append_logging_to_file(dict_logging_files, model_ckpt_fname, log_path, purge_flag=False)

    if plot_trace_flag:
        model_postfix = model_ckpt_fname.split('.')[0]
        arr_loss_train = np.genfromtxt(os.path.join(log_path, 'loss_train_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
        arr_loss_val = np.genfromtxt(os.path.join(log_path, 'loss_val_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
        arr_r2_train = np.genfromtxt(os.path.join(log_path, 'r2_train_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
        arr_r2_val = np.genfromtxt(os.path.join(log_path, 'r2_val_{}.txt'.format(model_postfix)), delimiter=',').reshape((-1,))
        plot_trace(arr_loss_train, arr_loss_val, best_index, 'upper right', 'loss(mean squared or poisson loss)-{}'.format(step_per_epoch), training_batch_size, penal_param, FLAGS)
        plot_trace(arr_r2_train, arr_r2_val, best_index, 'upper right', 'R-squared-{}'.format(step_per_epoch), training_batch_size, penal_param, FLAGS)

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

def dev_reg(pred, y):
    pred, y = pred.reshape((-1,)), y.reshape((-1,))
    dev = (y - pred)**2 # Defined as -ln(Likelihood)
    return dev

def dev_pois(pred, y):
    pred, y = pred.reshape((-1,)), y.reshape((-1,))
    lg_factor_y = np.array([np.sum(np.log(np.arange(1, v+1))) for v in y])
    dev = pred-y*np.log(pred)+lg_factor_y
    return dev

def cum_abs_resi(abs_resi_PI, resi_PII):
    abs_resi_PII = np.absolute(resi_PII)
    abs_resi_PI = np.reshape(abs_resi_PI, (-1,))
    abs_resi_PII = np.reshape(abs_resi_PII, (-1,))
    abs_resi_PI_PII = np.hstack((abs_resi_PI, abs_resi_PII))
    cum_abs_resi_PI_PII = 1.0 * np.cumsum(abs_resi_PI_PII) / (1.0 + np.arange(len(abs_resi_PI_PII)))
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