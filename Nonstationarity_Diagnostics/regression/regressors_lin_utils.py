import os
import re
import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, r2_score

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s(%(funcName)s)[%(lineno)d]: %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'zkg'}
logger = logging.getLogger('regressors_lin_utils')
logging.getLogger('regressors_lin_utils').setLevel(logging.INFO)

# Define those function in __init__ so that they can be past into joblib function for parallelization.
# Inside those functions, call other functions directly without adding 'self.' prior to that.
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
        fisher_info_mat = (fisher_info_mat + fisher_nugget * penalMatrix(fisher_info_mat.shape[1]))
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
