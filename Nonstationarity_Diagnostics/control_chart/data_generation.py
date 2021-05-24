import numpy as np
import pickle
import pandas as pd
import logging
import os
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
import itertools
import tensorflow as tf
import math
import torchvision

from scipy.sparse.linalg import eigsh, eigs
from tensorflow.keras.constraints import max_norm
from joblib import Parallel, delayed, parallel_backend

from sklearn.decomposition import PCA
# from control_chart.hotelling import EwmaT2PI, calEwmaT2StatisticsPI, calEwmaT2StatisticsPII, EwmaPI, calEwmaStatisticsPI, calEwmaStatisticsPII, calEwmaStatisticsHelper
from constants import *

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s(%(funcName)s)[%(lineno)d]: %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'zkg'}
logger = logging.getLogger('data_generation')
logging.getLogger('data_generation').setLevel(logging.INFO)

def Generate_AR_1D_Reg_Data(sigma, init_vals, coeffs, N, intcp=0):
    X = np.zeros((N, coeffs.shape[0]))
    X[0,:] = init_vals
    y = np.zeros(N)
    white_noise = np.random.normal(0,sigma,N)
    for i in range(N):
        y[i] = intcp + np.dot(X[i], coeffs) + white_noise[i]
        if i < N-1:
            X[i+1, :-1], X[i+1, -1] = X[i, 1:], y[i]
    return X, y

def Generate_Stepwise_AR_1D_Reg_Data(sigma, init_vals, ls_coeffs, ls_N_PIIs, intcp=0):
    N_PII = np.sum(ls_N_PIIs)
    X_PII = np.zeros((N_PII, init_vals.shape[0]))
    y_PII = np.zeros(N_PII)

    acc_num = 0
    for num_PII, coeffs in zip(ls_N_PIIs, ls_coeffs):
        X_PII[acc_num:acc_num+num_PII, :], y_PII[acc_num:acc_num+num_PII] = Generate_AR_1D_Reg_Data(sigma, init_vals, coeffs, num_PII, intcp)
        acc_num += num_PII
        init_vals = X_PII[acc_num-1]
    
    return X_PII, y_PII

def Image_Arr_Float_To_Int(img_arr):
    """Transform float image array to np.uint8 image array."""
    img_arr = (img_arr-np.min(img_arr))/(np.max(img_arr)-np.min(img_arr))*255
    img_arr = img_arr.astype(np.uint8)
    return img_arr

# def Generate_AR_2D_Reg_Data(sigma, wind_hei, wind_wid, init_vals, coeffs, intcp=0):
#     """ Generate autoregressive 2D array for regression.

#         Args:
#             sigma: The standard deviation of the white noise.
#             wind_hei: The window height of the neighbors for autoregressive lag terms.
#             wind_wid: The window width of the neighbors for autoregressive lag terms.
#             init_vals: The initial values of extended image (extended with margin on 
#                        the north and west sides) filled with white noise.
#             coeffs: A coefficent vector for generating this autoregressive image.
#             intcp: The intercept of autoregressive model.
#         Returns:
#             X: The neighbor lag terms for autoregressive model, filled row-by-row.
#                The shape is (img_hei*img_wid, x_dim).
#             y: The response for autoregressive model. The shape is (img_hei*img_wid,).
#             img_arr: The np.uint8 image array with shape (img_hei, img_wid).

#     """
#     ext_img_hei, ext_img_wid = init_vals.shape
#     img_hei, img_wid = ext_img_hei-wind_hei+1, ext_img_wid-wind_wid+1
#     x_dim = coeffs.shape[0]
#     X = np.zeros((img_hei*img_wid, x_dim))
#     white_noise = np.random.normal(0,sigma,img_hei*img_wid)

#     # The wind_hei and wind_wid are defined as the window height
#     # and width including the response as the bottom-right cornor.
#     patch_idx = 0
#     for ci in range(wind_wid-1, ext_img_wid):
#         xy_wind_ls = list(np.reshape(init_vals[:wind_hei, (ci-wind_wid+1):(ci+1)], (-1,)))
#         X[patch_idx, :] = np.array(xy_wind_ls)[:-1]
#         init_vals[wind_hei-1, ci] = xy_wind_ls[-1] = intcp + np.matmul(coeffs, X[patch_idx, :]) + white_noise[patch_idx]
#         patch_idx += 1
#         for ri in range(wind_hei, ext_img_hei):
#             xy_wind_ls = xy_wind_ls[wind_wid:]+list(init_vals[ri, (ci-wind_wid+1):(ci+1)])
#             X[patch_idx, :] = np.array(xy_wind_ls)[:-1]
#             init_vals[ri, ci] = xy_wind_ls[-1] = intcp + np.matmul(coeffs, X[patch_idx, :]) + white_noise[patch_idx]
#             patch_idx += 1
#     # The dimension 0 of X and y is in order of row-by-row, so that 2D matrix are linearly
#     # filling the X and y row-by-row.
#     X = X.reshape((img_wid, img_hei, x_dim)).transpose([1,0,2]).reshape((-1,x_dim))
#     y = init_vals[wind_hei-1:, wind_wid-1:].reshape((-1,))
#     img_arr = Image_Arr_Float_To_Int(init_vals[wind_hei-1:, wind_wid-1:])
#     return X, y, img_arr

def Cal_AR_2D_Mean_Std_Given_Intcp(gen_func, sigma, intcp, wind_hei, wind_wid, init_vals, coeffs):
    """ For 2D AR model, calculate mean and std for a given intercept and coeffs value. """
    _, y, _, _ = Generate_AR_2D_Data(gen_func, sigma, wind_hei, wind_wid, init_vals, coeffs, intcp)
    return np.mean(y), np.std(y)


def Generate_AR_2D_Data(gen_func, sigma, gen_wind_hei, gen_wind_wid, init_vals, coeffs, intcp=0, stdv=1.0, z_scale=1.0, latent_gen_func=None):
    """ Generate autoregressive 2D array for regression or classification.
        Args:
            gen_func: The generating function for image. If it is identity function, it is
                      regression model. If it is sigmoid function with random emulator, it
                      is a regression or classification model.
            sigma: The standard deviation of the white noise.
            gen_wind_hei: The window height of the neighbors for autoregressive lag terms.
            gen_wind_wid: The window width of the neighbors for autoregressive lag terms.
            init_vals: The initial values of extended image (extended with margin on 
                       the north and west sides) filled with white noise.
            coeffs: A coefficent vector for generating this autoregressive image.
            intcp: The intercept of autoregressive model.
            stdv: The standard deviation that I probably need to standardize the y-value (observations).
            z_scale: The scale of latent variables, because latent variable can be any scale.
        
        Returns:
            X: The neighbor lag terms for autoregressive model, filled row-by-row.
               The shape is (img_hei*img_wid, x_dim).
            y: The response for autoregressive model. The shape is (img_hei*img_wid,).
            z_arr: The latent variable used to generated y after applying gen_func.
            gen_img: The np.uint8 image array with shape (img_hei, img_wid).
    """
    ext_img_hei, ext_img_wid = init_vals.shape
    img_hei, img_wid = ext_img_hei-gen_wind_hei, ext_img_wid-gen_wind_wid # Doesn't include the margin of generating te 2D materials sample.
    x_dim = coeffs.shape[0]
    X = np.zeros((img_hei*img_wid, x_dim))
    gen_img_arr = np.zeros((img_hei, img_wid), dtype=type(gen_func(1)))
    white_noise = np.random.normal(0,sigma,img_hei*img_wid)

    # The gen_wind_hei and gen_wind_wid are defined as the window height
    # and width excluding the response as the bottom-right cornor.
    patch_idx = 0
    for ci in range(gen_wind_wid, ext_img_wid):
        z_wind_ls = list(np.reshape(init_vals[:(gen_wind_hei+1), (ci-gen_wind_wid):(ci+1)], (-1,)))
        x_wind_ls = [gen_func(z*z_scale) for z in z_wind_ls]
        X[patch_idx, :] = np.array(x_wind_ls[:-1])
        if latent_gen_func is None:
            init_vals[gen_wind_hei, ci] = z_wind_ls[-1] = intcp + np.matmul(coeffs, np.array(z_wind_ls[:-1])) + white_noise[patch_idx]
        else:
            init_vals[gen_wind_hei, ci] = z_wind_ls[-1] = latent_gen_func(intcp + np.matmul(coeffs, np.array(z_wind_ls[:-1])) + white_noise[patch_idx])
        gen_img_arr[0, ci-gen_wind_wid] = x_wind_ls[-1] = gen_func(z_wind_ls[-1]*z_scale)
        patch_idx += 1
        for ri in range(gen_wind_hei+1, ext_img_hei):
            new_z_ls = list(init_vals[ri, (ci-gen_wind_wid):(ci+1)])
            z_wind_ls = z_wind_ls[(gen_wind_wid+1):]+new_z_ls
            x_wind_ls = x_wind_ls[(gen_wind_wid+1):]+[gen_func(z*z_scale) for z in new_z_ls]
            X[patch_idx, :] = np.array(x_wind_ls[:-1])
            if latent_gen_func is None:
                init_vals[ri, ci] = z_wind_ls[-1] = intcp + np.matmul(coeffs, np.array(z_wind_ls[:-1])) + white_noise[patch_idx]
            else:
                init_vals[ri, ci] = z_wind_ls[-1] = latent_gen_func(intcp + np.matmul(coeffs, np.array(z_wind_ls[:-1])) + white_noise[patch_idx])
            gen_img_arr[ri-gen_wind_hei, ci-gen_wind_wid] = x_wind_ls[-1] = gen_func(z_wind_ls[-1]*z_scale)
            patch_idx += 1
    # The dimension 0 of X and y should be in order of row-by-row, so that 2D matrix are linearly
    # filling the X and y row-by-row.
    X = X.reshape((img_wid, img_hei, x_dim)).transpose([1,0,2]).reshape((-1,x_dim)).astype(type(gen_func(1)))
    y = gen_img_arr.reshape((-1,)).astype(type(gen_func(1)))
    z_arr = init_vals[gen_wind_hei:, gen_wind_wid:] # Latent variable for generating the data y.
    gen_img = Image_Arr_Float_To_Int(gen_img_arr)
    return X, y, z_arr, gen_img


def Generate_AR_2D_Nois_Data(gen_func, sigma, gen_wind_hei, gen_wind_wid, init_vals, init_vals_nois, coeffs, coeffs_nois, 
                             nois_profile_sigma, nois_size, nois_scale, intcp=0, stdv=1.0, z_scale=1.0):
    _, _, z_arr, _ = Generate_AR_2D_Data(gen_func, sigma, gen_wind_hei, gen_wind_wid, init_vals, coeffs, intcp, stdv, z_scale)
    _, _, z_arr_nois, _ = Generate_AR_2D_Data(gen_func, sigma, gen_wind_hei, gen_wind_wid, init_vals_nois, coeffs_nois, intcp, stdv, z_scale)
    img_hei, img_wid = z_arr.shape
    r_nois, c_nois = np.random.randint(1, img_hei, size=nois_size), np.random.randint(1, img_wid, size=nois_size)
    
    x_dim = coeffs.shape[0]
    nois_z_arr = np.zeros_like(z_arr)
    nois_gen_img_arr = np.zeros_like(z_arr, dtype=type(gen_func(1)))
    nois_wei_arr = np.zeros_like(z_arr)
    
    for ri in range(img_hei):
        for ci in range(img_wid):
            nois_wei_arr[ri, ci] = np.sum(nois_scale*np.exp(-((ri-r_nois)**2+(ci-c_nois)**2)/2/nois_profile_sigma**2))
            nois_z_arr[ri, ci] = (z_arr[ri, ci]+nois_wei_arr[ri,ci]*z_arr_nois[ri, ci])/(1+nois_wei_arr[ri,ci])
            nois_gen_img_arr[ri, ci] = gen_func(nois_z_arr[ri, ci]*z_scale)
    
    nois_y = nois_gen_img_arr.reshape((-1,)).astype(type(gen_func(1)))
    nois_gen_img = Image_Arr_Float_To_Int(nois_gen_img_arr)
    
    return nois_wei_arr, nois_y, nois_z_arr, nois_gen_img


def Generate_AR_2D_Data_Causal(gen_func, sigma, gen_wind_hei, gen_wind_wid, init_vals, coeffs, intcp=0):
    """ Generate autoregressive 2D array for classification.

        Args:
            gen_func: The generating function for image. If it is identity function, it is
                      regression model. If it is sigmoid function with random emulator, it
                      is classification model.
            sigma: The standard deviation of the white noise.
            gen_wind_hei: The window height of the neighbors for autoregressive lag terms.
                      See 2016 Ramin paper for definition.
            gen_wind_wid: The window width of the neighbors for autoregressive lag terms.
                      See 2016 Ramin paper for definition.
            init_vals: The initial values of extended image (extended with margin on 
                       the north and west sides) filled with white noise.
            coeffs: A coefficent vector for generating this autoregressive image.
            intcp: The intercept of autoregressive model.
        
        Returns:
            X: The neighbor lag terms for autoregressive model, filled row-by-row.
               The shape is (img_hei*img_wid, x_dim).
            y: The response for autoregressive model. The shape is (img_hei*img_wid,).
            z_arr: The latent variable used to generated y after applying gen_func.
            gen_img: The np.uint8 image array with shape (img_hei, img_wid).

    """
    ext_img_hei, ext_img_wid = init_vals.shape
    img_hei, img_wid = ext_img_hei-4*gen_wind_hei, ext_img_wid-4*gen_wind_wid
    x_dim = coeffs.shape[0]
    X = np.zeros((img_hei*img_wid, x_dim))
    gen_img_arr = np.zeros((img_hei, img_wid), dtype=type(gen_func(1)))
    white_noise = np.random.normal(0,sigma,img_hei*img_wid)

    # The gen_wind_hei and gen_wind_wid are defined as the window height
    # and width excluding the response as the bottom-middle pixel.
    # This is causal-window defined in Ramin 2016 paper.
    patch_idx = 0
    for ci in range(gen_wind_wid, ext_img_wid-gen_wind_wid):
        z_wind_ls_1 = list(init_vals[:(2*gen_wind_hei+1), (ci-gen_wind_wid):ci].reshape((-1,))) 
        z_wind_ls_2 = list(init_vals[:gen_wind_hei, ci].reshape((-1,)))
        z_wind_ls = z_wind_ls_1 + z_wind_ls_2
        x_wind_ls_1 = [gen_func(z) for z in z_wind_ls_1]
        x_wind_ls_2 = [gen_func(z) for z in z_wind_ls_2]
        x_wind_ls = x_wind_ls_1 + x_wind_ls_2
        X[patch_idx, :] = np.array(x_wind_ls)
        init_vals[gen_wind_hei, ci] = intcp + np.matmul(coeffs, np.array(z_wind_ls)) + white_noise[patch_idx]
        gen_img_arr[0, ci-gen_wind_wid] = gen_func(init_vals[gen_wind_hei, ci])
        patch_idx += 1
        for ri in range(gen_wind_hei+1, ext_img_hei-gen_wind_hei):
            new_z_ls = list(init_vals[ri+gen_wind_hei, (ci-gen_wind_wid):ci])
            z_wind_ls_1 = z_wind_ls_1[gen_wind_wid:]+new_z_ls
            z_wind_ls_2 = z_wind_ls_2[1:] + [init_vals[ri-1, ci]]
            z_wind_ls = z_wind_ls_1 + z_wind_ls_2
            x_wind_ls_1 = x_wind_ls_1[gen_wind_wid:]+[gen_func(z) for z in new_z_ls]
            x_wind_ls_2 = x_wind_ls_2[1:] + [gen_func(init_vals[ri-1, ci])]
            x_wind_ls = x_wind_ls_1 + x_wind_ls_2
            if 2*gen_wind_wid<=ci<(ext_img_wid-2*gen_wind_wid) and 2*gen_wind_hei<=ri<(ext_img_hei-2*gen_wind_hei):
                X[patch_idx, :] = np.array(x_wind_ls)
                patch_idx += 1
                gen_img_arr[ri-2*gen_wind_hei, ci-2*gen_wind_wid] = gen_func(init_vals[ri, ci])
            init_vals[ri, ci] = intcp + np.matmul(coeffs, np.array(z_wind_ls)) + white_noise[patch_idx]
            
    # The dimension 0 of X and y should be in order of row-by-row, so that 2D matrix are linearly
    # filling the X and y row-by-row.
    X = X.reshape((img_wid, img_hei, x_dim)).transpose([1,0,2]).reshape((-1,x_dim)).astype(type(gen_func(1)))
    y = gen_img_arr.reshape((-1,)).astype(type(gen_func(1)))
    z_arr = init_vals[2*gen_wind_hei:(ext_img_hei-2*gen_wind_hei), 2*gen_wind_wid:(ext_img_wid-2*gen_wind_wid)] # Latent variable for generating the data y.
    gen_img = Image_Arr_Float_To_Int(gen_img_arr)
    return X, y, z_arr, gen_img


def Generate_Blockwise_AR_2D_Data(gen_func, sigma, gen_wind_hei, gen_wind_wid, init_vals, ls_coeffs, ls_row_grid_pts, ls_col_grid_pts, intcp=0, z_scale=1.0):
    """ Generate blockwise autoregressive 2D regression image.

        Args:
            gen_func: The generating function for image. If it is identity function, it is
                      regression model. If it is sigmoid function with random emulator, it
                      is classification model.
            sigma: The standard deviation of the white noise.
            gen_wind_hei: The window height of the neighbors for autoregressive lag terms.
            gen_wind_wid: The window width of the neighbors for autoregressive lag terms.
            init_vals: The initial values of extended image (extended with margin on 
                       the north and west sides) filled with white noise.
            ls_coeffs: A list of lists of coefficient vectors corresponding to the block position.
            ls_row_grid_pts: A list of starting indices of pixels of rows for different blocks (not including the northern and western margin).
            ls_col_grid_pts: A list of starting indices of pixels of columns for different blocks (not including the northern and western margin).
            intcp: The intercept of autoregressive model.
        
        Returns:
            X: The neighbor lag terms for autoregressive model, filled row-by-row.
               The shape is (img_hei*img_wid, x_dim).
            y: The response for autoregressive model. The shape is (img_hei*img_wid,).
            z_arr: The latent variable used to generated y after applying gen_func.
            gen_img: The np.uint8 image array with shape (img_hei, img_wid).

    """
    ext_img_hei, ext_img_wid = init_vals.shape
    img_hei, img_wid = ext_img_hei-gen_wind_hei, ext_img_wid-gen_wind_wid

    # The ls_grid_pts has two lists. The first one is the starting row indices.
    # The second one is the starting column indices.
    ls_row_block_sizes, ls_col_block_sizes = np.diff(ls_row_grid_pts+[img_hei]), np.diff(ls_col_grid_pts+[img_wid])
    x_dim = ls_coeffs[0][0].shape[0]
    X = np.zeros((0, img_wid, x_dim), dtype=type(gen_func(1)))
    y = np.zeros((0, img_wid), dtype=type(gen_func(1)))
    for ri, (rs, rl) in enumerate(zip(ls_row_grid_pts, ls_row_block_sizes)):
        logger.info("The start row idx and block row size is (%s, %s).\n", rs, rl, extra=d)
        X_row_block = np.zeros((rl, 0, x_dim), dtype=type(gen_func(1))) # A block of several rows of the X matrix.
        y_row_block = np.zeros((rl, 0), dtype=type(gen_func(1))) # A block of several rows of the response y vector.
        for ci, (cs, cl) in enumerate(zip(ls_col_grid_pts, ls_col_block_sizes)):
            logger.info("The start col idx and block col size is (%s, %s).\n", cs, cl, extra=d)
            block_init_vals = init_vals[rs:rs+rl+gen_wind_hei, cs:cs+cl+gen_wind_wid].copy()
            X_temp, y_temp, z_arr_temp, _ = Generate_AR_2D_Data(gen_func, sigma, gen_wind_hei, gen_wind_wid, block_init_vals, ls_coeffs[ri][ci], intcp=intcp, z_scale=z_scale)
            init_vals[rs+gen_wind_hei:rs+rl+gen_wind_hei, cs+gen_wind_wid:cs+cl+gen_wind_wid] = z_arr_temp
            X_row_block = np.concatenate((X_row_block, X_temp.reshape((rl,cl,x_dim))), axis=1)
            y_row_block = np.concatenate((y_row_block, y_temp.reshape((rl,cl))), axis=1)
        X = np.concatenate((X, X_row_block), axis=0)
        y = np.concatenate((y, y_row_block), axis=0)
    
    X = X.reshape((-1, x_dim)).astype(type(gen_func(1))) # In raster order
    gen_img = Image_Arr_Float_To_Int(y)
    y = y.reshape((-1,)).astype(type(gen_func(1))) # In raster order
    z_arr = init_vals[gen_wind_hei:, gen_wind_wid:] # Latent variable for generating the data y.   
    return X, y, z_arr, gen_img


# def Generate_Blockwise_AR_2D_Reg_Data(sigma, wind_hei, wind_wid, init_vals, ls_coeffs, ls_row_grid_pts, ls_col_grid_pts, intcp=0):
#     """ Generate blockwise autoregressive 2D regression image.

#         Args:
#             sigma: The standard deviation of the white noise.
#             wind_hei: The window height of the neighbors for autoregressive lag terms.
#             wind_wid: The window width of the neighbors for autoregressive lag terms.
#             init_vals: The initial values of extended image (extended with margin on 
#                        the north and west sides) filled with white noise.
#             ls_coeffs: A list of lists of coefficient vectors corresponding to the block position.
#             ls_row_grid_pts: A list of starting indices of pixels of rows for different blocks.
#             ls_col_grid_pts: A list of starting indices of pixels of columns for different blocks.
#             intcp: The intercept of autoregressive model.
        
#         Returns:
#             X: The neighbor lag terms for autoregressive model, filled row-by-row.
#                The shape is (img_hei*img_wid, x_dim).
#             y: The response for autoregressive model. The shape is (img_hei*img_wid,).
#             img_arr: The np.uint8 image array with shape (img_hei, img_wid).
#     """
#     ext_img_hei, ext_img_wid = init_vals.shape
#     img_hei, img_wid = ext_img_hei-wind_hei+1, ext_img_wid-wind_wid+1

#     # The ls_grid_pts has two lists. The first one is the starting row indices.
#     # The second on is the starting column indices.
#     ls_row_block_sizes, ls_col_block_sizes = np.diff(ls_row_grid_pts+[img_hei]), np.diff(ls_col_grid_pts+[img_wid])
#     x_dim = ls_coeffs[0][0].shape[0]
#     X = np.zeros((0, img_wid, x_dim))
#     y = np.zeros((0, img_wid))
#     for ri, (rs, rl) in enumerate(zip(ls_row_grid_pts, ls_row_block_sizes)):
#         X_row_block = np.zeros((rl, 0, x_dim)) # A block of several rows of the X matrix.
#         y_row_block = np.zeros((rl, 0)) # A block of several rows of the response y vector.
#         for ci, (cs, cl) in enumerate(zip(ls_col_grid_pts, ls_col_block_sizes)):
#             block_init_vals = init_vals[rs:rs+rl+wind_hei-1, cs:cs+cl+wind_wid-1].copy()
#             X_temp, y_temp, _ = Generate_AR_2D_Reg_Data(sigma, wind_hei, wind_wid, block_init_vals, ls_coeffs[ri][ci], intcp)
#             init_vals[rs+wind_hei-1:rs+rl+wind_hei-1, cs+wind_wid-1:cs+cl+wind_wid-1] = y_temp.reshape((rl, cl))
#             X_row_block = np.concatenate((X_row_block, X_temp.reshape((rl,cl,x_dim))), axis=1)
#             y_row_block = np.concatenate((y_row_block, y_temp.reshape((rl,cl))), axis=1)
#         X = np.concatenate((X, X_row_block), axis=0)
#         y = np.concatenate((y, y_row_block), axis=0)
    
#     # for idx, (ri, rl), (ci, cl) in enumerate(itertools.product(zip(ls_row_grid_pts, ls_row_block_sizes), zip(ls_col_grid_pts, ls_col_block_sizes))):
#     #     block_init_vals = init_vals[ri:ri+rl+wind_hei-1, ci:ci+cl+wind_wid-1].copy()
#     #     X_temp, y_temp, _ = Generate_AR_2D_Reg_Data(sigma, wind_hei, wind_wid, block_init_vals, ls_coeffs[idx], intcp)
#     #     X, y = np.vstack((X, X_temp)), np.vstack((y, y_temp))

#     X = X.reshape((-1, x_dim))
#     y = y.reshape((-1,))    
#     img_arr = Image_Arr_Float_To_Int(init_vals[wind_hei-1:, wind_wid-1:])
#     return X, y, img_arr


def Regenerate_Materials_Data(img_arr, model, model_pred, FLAGS):
    """ Regenerate 2D spatial data for inspection.

        Args:
            img_arr: Initial value for regeneration.
                     For grey-scale image, use: img_arr = np.random.normal(0, FLAGS.nois_sigma, (img_wid, img_hei))
                     For binary image, use: img_arr = np.random.binomial(1, 0.5, (5,5))
            model: The trained model used to regenerate the image.
            model_pred: The function to make prediction using the model.
                        For classification model use:
                        
                            def model_pred(input_arr, model, FLAGS):
                                if FLAGS.nnet:
                                    logits = model(np.array(input_arr).reshape((1,-1)))
                                else:
                                    logits = model.predict_proba(np.array(input_arr).reshape((1,-1)))
                                return 1*(np.random.random(1)[0] < 1/(1+np.exp(logits[0, 0]-logits[0, 1]))) 

                        For regression model use:

                            def model_pred(input_arr, model, FLAGS):
                                if FLAGS.nnet:
                                    return model(np.array(input_arr).reshape((1,-1)))[0, 0]
                                else:
                                    return model.predict(np.array(input_arr).reshape((1,-1)))[0]

            FLAGS: Some information about the model.

        Return:
            img_arr: The generate image.
    """
    # img_hei, img_wid = FLAGS.wind_hei+FLAGS.regen_grid_size-1, FLAGS.wind_wid+FLAGS.regen_grid_size-1
    img_hei, img_wid = img_arr.shape

    if FLAGS.materials_model == 'causal':
        # The FLAGS.wind_hei and FLAGS.wind_wid are defined as the window height
        # and width including the response as the bottom-right cornor.  
        for ci in range(FLAGS.wind_wid, img_wid-FLAGS.wind_wid):
            xy_wind_ls = list(np.reshape(img_arr[:(FLAGS.wind_hei+1), (ci-FLAGS.wind_wid):(ci+1)], (-1,)))
            # This step xy_wind_ls[-1] cannot be skipped. It is very easy to skip this one.
            img_arr[FLAGS.wind_hei, ci] = xy_wind_ls[-1] = model_pred(xy_wind_ls[:-1], model, FLAGS)
            for ri in range(FLAGS.wind_hei+1, img_hei-FLAGS.wind_hei):
                xy_wind_ls = xy_wind_ls[(FLAGS.wind_wid+1):]+list(img_arr[ri, (ci-FLAGS.wind_wid):(ci+1)])
                img_arr[ri, ci] = xy_wind_ls[-1] = model_pred(xy_wind_ls[:-1], model, FLAGS)
        FLAGS.cur_vf = np.sum(img_arr)/(img_arr.shape[0]*img_arr.shape[1])
        print("The portion of positive labels in the regenerated image is {}.\n".format(FLAGS.cur_vf, ))
        img_arr = (img_arr-np.min(img_arr))/(np.max(img_arr)-np.min(img_arr))*255
        return img_arr, img_arr[FLAGS.wind_hei:(img_hei-FLAGS.wind_hei), FLAGS.wind_wid:(img_wid-FLAGS.wind_wid)].copy()
    elif FLAGS.materials_model == 'causal_1':
        # The FLAGS.wind_hei and FLAGS.wind_wid are defined as the window height
        # and width including the response as the bottom-right cornor.
        # Check the definition of Ramin 2016 paper.
        for ci in range(FLAGS.wind_wid, img_wid-FLAGS.wind_wid):
            x_wind_ls_1 = list(img_arr[:(2*FLAGS.wind_hei+1), (ci-FLAGS.wind_wid):ci].reshape((-1,)))
            x_wind_ls_2 = list(img_arr[:FLAGS.wind_hei, ci].reshape((-1,)))
            x_wind_ls = x_wind_ls_1 + x_wind_ls_2
            img_arr[FLAGS.wind_hei, ci] = model_pred(x_wind_ls, model, FLAGS)
            for ri in range(FLAGS.wind_hei+1, img_hei-FLAGS.wind_hei):
                x_wind_ls_1 = x_wind_ls_1[FLAGS.wind_wid:]+list(img_arr[ri+FLAGS.wind_hei, (ci-FLAGS.wind_wid):ci])
                x_wind_ls_2 = x_wind_ls_2[1:] + [img_arr[ri-1, ci]]
                x_wind_ls = x_wind_ls_1 + x_wind_ls_2
                img_arr[ri, ci] = model_pred(x_wind_ls, model, FLAGS)
        FLAGS.cur_vf = np.sum(img_arr)/(img_arr.shape[0]*img_arr.shape[1])
        print("The portion of positive labels in the regenerated image is {}.\n".format(FLAGS.cur_vf, ))
        img_arr = (img_arr-np.min(img_arr))/(np.max(img_arr)-np.min(img_arr))*255
        return img_arr, img_arr[FLAGS.wind_hei:(img_hei-FLAGS.wind_hei), FLAGS.wind_wid:(img_wid-FLAGS.wind_wid)].copy()


def Generate_Materials_Data(img_arr, FLAGS):
    """ Generate materials micro-structure data from image array.

        Args:
            img_arr: The image array.
            FLAGS: All flags.

        Returns:
            X: The numpy array of pixels in neighborhood windows (causal or non_causal).
               This is filled in row-by-row in the image pixel order.
            y: The responses of supervised-learning. It is actually all pixel values filled row-by-row.
            n_hei: The number of pixels in height.
            n_wid: The number of pixels in width.
            
    """
    img_hei, img_wid = img_arr.shape
    n_hei = (img_hei-2*FLAGS.wind_hei) # The number of pixels in height direction. The FLAGS.wind_hei is the neighborhood features for training supervised learning model. 
    n_wid = (img_wid-2*FLAGS.wind_wid) # The number of pixels in horizontal direction.
    n_sample = n_hei*n_wid # The number of pixels as response.
    row_idx = 0
    if FLAGS.materials_model == 'causal':
        # The FLAGS.wind_hei and FLAGS.wind_wid are defined as the window height -1
        # and width-1 including the response as the bottom-right cornor.  
        xy_dim = (FLAGS.wind_hei+1)*(FLAGS.wind_wid+1)
        Xy = np.zeros((n_sample, xy_dim))
        for ci in range(FLAGS.wind_wid, img_wid-FLAGS.wind_wid):
            xy_wind_ls = list(np.reshape(img_arr[:(FLAGS.wind_hei+1), (ci-FLAGS.wind_wid):(ci+1)], (-1,)))
            Xy[row_idx,:] = np.array(xy_wind_ls)
            row_idx += 1
            for ri in range(FLAGS.wind_hei+1, img_hei-FLAGS.wind_hei):
                xy_wind_ls = xy_wind_ls[(FLAGS.wind_wid+1):]+list(img_arr[ri, (ci-FLAGS.wind_wid):(ci+1)])
                Xy[row_idx,:] = np.array(xy_wind_ls)
                row_idx += 1
        # The filling order has been checked in the following line.
        Xy = Xy.reshape(n_wid, n_hei, xy_dim).transpose([1,0,2]).reshape((-1, xy_dim))
        X, y = Xy[:,:-1], Xy[:,-1]
    elif FLAGS.materials_model == 'causal_1':
        # The FLAGS.wind_hei and FLAGS.wind_wid are defined as the window height -1
        # and width - 1 excluding the response as the bottom-right cornor.
        # Check Ramin 2016 paper.
        xy_dim = (2*FLAGS.wind_hei+1)*FLAGS.wind_wid+FLAGS.wind_hei+1
        Xy = np.zeros((n_sample, xy_dim))
        for ci in range(FLAGS.wind_wid, img_wid-FLAGS.wind_wid):
            xy_wind_ls_1 = list(img_arr[:(2*FLAGS.wind_hei+1), (ci-FLAGS.wind_wid):ci].reshape((-1,)))
            xy_wind_ls_2 = list(img_arr[:(FLAGS.wind_hei+1), ci].reshape((-1,)))
            xy_wind_ls = xy_wind_ls_1 + xy_wind_ls_2
            Xy[row_idx,:] = np.array(xy_wind_ls)
            row_idx += 1
            for ri in range(FLAGS.wind_hei+1, img_hei-FLAGS.wind_hei):
                xy_wind_ls_1 = xy_wind_ls_1[FLAGS.wind_wid:] + list(img_arr[ri+FLAGS.wind_hei, (ci-FLAGS.wind_wid):ci])
                xy_wind_ls_2 = xy_wind_ls_2[1:] + [img_arr[ri, ci]]
                xy_wind_ls = xy_wind_ls_1 + xy_wind_ls_2
                Xy[row_idx,:] = np.array(xy_wind_ls)
                row_idx += 1
        # The filling order has been checked in the following line.
        Xy = Xy.reshape(n_wid, n_hei, xy_dim).transpose([1,0,2]).reshape((-1, xy_dim))
        X, y = Xy[:,:-1], Xy[:,-1]
    elif FLAGS.materials_model == 'non_causal':
        # The FLAGS.wind_hei and FLAGS.wind_wid are defined as the number of pixels from the target
        # to the left or to the right boundary (excluding the target one). And FLAGS.width is the 
        # number of pixels from the target to the up and down boundary (excluding the target one).
        logger.info("The shape of the data for building a model is ({}, {}).".format(n_sample, (2*FLAGS.wind_hei+1)*(2*FLAGS.wind_wid+1)), extra=d)
        xy_dim = (2*FLAGS.wind_hei+1)*(2*FLAGS.wind_wid+1)
        Xy = np.zeros((n_sample, xy_dim))
        for ci in range(FLAGS.wind_wid, img_wid-FLAGS.wind_wid):
            xy_wind_ls = list(np.reshape(img_arr[:(2*FLAGS.wind_hei+1), (ci-FLAGS.wind_wid):(ci+FLAGS.wind_wid+1)], (-1,)))
            Xy[row_idx,:] = np.array(xy_wind_ls)
            row_idx += 1
            for ri in range(FLAGS.wind_hei+1, img_hei-FLAGS.wind_hei):
                xy_wind_ls = xy_wind_ls[(2*FLAGS.wind_wid+1):]+list(img_arr[ri+FLAGS.wind_hei, (ci-FLAGS.wind_wid):(ci+FLAGS.wind_wid+1)])
                Xy[row_idx,:] = np.array(xy_wind_ls)
                row_idx += 1
        n_col = Xy.shape[1]
        Xy = Xy.reshape((n_wid, n_hei, xy_dim)).transpose([1,0,2]).reshape((-1,xy_dim))
        X, y = Xy[:,list(range(n_col//2))+list(range((n_col//2+1),n_col))], Xy[:,n_col//2]
    return X, y, n_hei, n_wid


def Generate_Predictor(mean, cov, N, transf_mat):
    X = np.matmul(
        np.random.multivariate_normal(
            mean,
            cov,
            N).astype('float32'),
        transf_mat)
    return X


def Generate_Stepwise_Predictor(
        mean,
        cov,
        sigma,
        transf_mat,
        coeffs,
        mean2,
        cov2,
        N_steps):
    X = Generate_Predictor(mean, cov, N_steps[0], transf_mat)

    for num in N_steps[1:]:
        X_temp = Generate_Predictor(mean2, cov2, num, transf_mat)
        X = np.vstack((X, X_temp))
    return X


def Generate_Reg_Data(mean, cov, sigma, transf_mat, coeffs, N, intcp=0):
    X = Generate_Predictor(mean, cov, N, transf_mat)
    # y has shape (-1,)
    y = np.dot(X, coeffs)+intcp+np.random.normal(0,sigma,N)
    return X, y


def Generate_Stepwise_Reg_Data(
        mean,
        cov,
        sigma,
        transf_mat,
        coeffs,
        mean2,
        cov2,
        N_PIIs,
        change_factors,
        intcp_ls = None):
    X_PII = Generate_Stepwise_Predictor(
            mean,
            cov,
            sigma,
            transf_mat,
            coeffs,
            mean2,
            cov2,
            N_PIIs)

    y_PII = Generate_Stepwise_Reg_Response(
            X_PII,
            sigma,
            transf_mat,
            coeffs,
            N_PIIs,
            mean2,
            cov2,
            change_factors,
            intcp_ls)
    return X_PII, y_PII


def Generate_Stepwise_Reg_Response(
        X_PII,
        sigma,
        transf_mat,
        coeffs,
        N_PIIs,
        mean2,
        cov2,
        change_factors,
        intcp_ls = None):
    # X_PII_1 = X_PII[:N_PIIs[0],:]
    # y_PII_1 = (np.random.random(N_PIIs[0]) > 1 / (1 + np.exp((-1)*np.dot(
    #     X_PII_1, coeffs) (-1)*np.random.normal(0, sigma, N_PIIs[0])))).astype('int')

    y_PII = np.array([], dtype=np.float32)
    N_PIIs_pre = 0
    if not intcp_ls:
        intcp_ls = [0]*len(change_factors)

    for num, ch_factor, b0 in zip(N_PIIs, change_factors, intcp_ls):
        coeffs2 = coeffs * (1)
        ch_arr_flag = True # mark if this ch_factor is a number or an array
        # try:
        #     len(ch_factor)
        # except TypeError:
        #     ch_arr_flag = False
        #
        # if ch_arr_flag == True:
        #     if len(coeffs2) >= len(ch_factor):
        #         coeffs2[:len(ch_factor)] *= ch_factor
        #     else:
        #         logger.info("The length of change factor is longer than the coefficients.", extra=d)
        #         coeffs2[:] *= ch_factor[:len(coeffs2)]
        # else:
        #
        if len(coeffs2) > 8: # This is for legacy example when 10 predictors exist.
            # coeffs2[0] *= (ch_factor)
            # coeffs2[1] *= (ch_factor)
            # coeffs2[2] *= (ch_factor)
            # coeffs2[3] *= (ch_factor)
            coeffs2[:4] *= ch_factor
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        else:
            # This is for new experiments that we can put non-uniform changes.
            # The ch_factor can be np.array with length not longer than coeff2.
            try:
                coeffs2[:len(ch_factor)] *= ch_factor
            except TypeError:
                coeffs2[:] *= ch_factor
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        X_temp = X_PII[N_PIIs_pre:N_PIIs_pre+num,:]
        N_PIIs_pre += num
        # np.matmul(
        #     np.random.multivariate_normal(
        #         mean2,
        #         cov2,
        #         num).astype('float32'),
        #     transf_mat)
        y_temp = np.dot(X_temp, coeffs2)+b0+np.random.normal(0, sigma, num)
        y_PII = np.append(y_PII, y_temp)

    return y_PII


def Generate_Gradual_Reg_Data(
        mean,
        cov,
        sigma,
        transf_mat,
        coeffs,
        mean2,
        cov2,
        N_PIIs,
        change_factors):
    X_PII = Generate_Stepwise_Predictor(
            mean,
            cov,
            sigma,
            transf_mat,
            coeffs,
            mean2,
            cov2,
            N_PIIs)

    y_PII = Generate_Gradual_Reg_Response(
            X_PII,
            sigma,
            transf_mat,
            coeffs,
            N_PIIs,
            mean2,
            cov2,
            change_factors)
    return X_PII, y_PII


def Generate_Gradual_Reg_Response(
        X_PII,
        sigma,
        transf_mat,
        coeffs,
        N_PIIs,
        mean2,
        cov2,
        change_factors):
    # X_PII_1 = X_PII[:N_PIIs[0],:]
    # y_PII_1 = (np.random.random(N_PIIs[0]) > 1 / (1 + np.exp((-1)*np.dot(
    #     X_PII_1, coeffs) (-1)*np.random.normal(0, sigma, N_PIIs[0])))).astype('int')
    y_PII = np.array([], dtype=np.float32)
    N_PIIs_pre = 0
    coeffs_old = coeffs

    for num, ch_factor in zip(N_PIIs, change_factors):
        coeffs2 = coeffs * (1)
        if len(coeffs2) > 8: # This is for legacy example when 10 predictors exist.
            # coeffs2[0] *= (ch_factor)
            # coeffs2[1] *= (ch_factor)
            # coeffs2[2] *= (ch_factor)
            # coeffs2[3] *= (ch_factor)
            coeffs2[:4] *= ch_factor
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        else:
            # This is for new experiments that we can put non-uniform changes.
            try:
                coeffs2[:len(ch_factor)] *= ch_factor
            except TypeError:
                coeffs2[:] *= ch_factor
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        X_temp = X_PII[N_PIIs_pre:N_PIIs_pre+num,:]
        coeffs_dyn = np.zeros_like(X_temp)
        for idx, (coeff1, coeff2) in enumerate(zip(coeffs_old, coeffs2)):
            coeffs_dyn[:, idx] = np.linspace(coeff1, coeff2, num=num)
        N_PIIs_pre += num
        # np.matmul(
        #     np.random.multivariate_normal(
        #         mean2,
        #         cov2,
        #         num).astype('float32'),
        #     transf_mat)
        y_temp = (np.sum(X_temp * coeffs_dyn, axis=1, keepdims=True) +
                  np.random.normal(0, sigma, (num, 1)))
        y_PII = np.append(y_PII, y_temp) # No matter dimension of y_PII and y_temp, it returns a row vector.
        coeffs_old = coeffs2

    return y_PII


def Generate_Pois_Reg_Data(mean, cov, sigma, transf_mat, coeffs, N, intcp=0):
    X = Generate_Predictor(mean, cov, N, transf_mat)
    # y has shape (-1,)
    y = 1.0*np.random.poisson(lam=np.exp(np.dot(X, coeffs)+intcp+np.random.normal(0,sigma,N)))
    logger.info("The possion responses %s with coeffs %s.", y, coeffs, extra=d)
    # Here, there is a very subtle bug. Looks like the poisson output would generate
    # np.float64 by default. This would result in datatype incompatable in tensorflow
    # calculation, because in log_posson_loss the multiplication cannot be done
    # between np.float32(float) and np.float64(double).
    return X, y.astype(np.float32)


def Generate_Stepwise_Pois_Reg_Data(
        mean,
        cov,
        sigma,
        transf_mat,
        coeffs,
        mean2,
        cov2,
        N_PIIs,
        change_factors,
        intcp_ls=None):
    X_PII = Generate_Stepwise_Predictor(
            mean,
            cov,
            sigma,
            transf_mat,
            coeffs,
            mean2,
            cov2,
            N_PIIs)

    y_PII = Generate_Stepwise_Pois_Reg_Response(
            X_PII,
            sigma,
            transf_mat,
            coeffs,
            N_PIIs,
            mean2,
            cov2,
            change_factors,
            intcp_ls)

    return X_PII, y_PII


def Generate_Stepwise_Pois_Reg_Response(
        X_PII,
        sigma,
        transf_mat,
        coeffs,
        N_PIIs,
        mean2,
        cov2,
        change_factors,
        intcp_ls=None):
    # X_PII_1 = X_PII[:N_PIIs[0],:]
    # y_PII_1 = (np.random.random(N_PIIs[0]) > 1 / (1 + np.exp((-1)*np.dot(
    #     X_PII_1, coeffs) (-1)*np.random.normal(0, sigma, N_PIIs[0])))).astype('int')

    y_PII = np.array([], dtype=np.float32)
    N_PIIs_pre = 0

    if not intcp_ls:
        intcp_ls = [0]*len(change_factors)

    for num, ch_factor, b0 in zip(N_PIIs, change_factors, intcp_ls):
        coeffs2 = coeffs * (1)
        if len(coeffs2) > 8: # This is for legacy example when 10 predictors exist.
            # coeffs2[0] *= (ch_factor)
            # coeffs2[1] *= (ch_factor)
            # coeffs2[2] *= (ch_factor)
            # coeffs2[3] *= (ch_factor)
            coeffs2[:4] *= ch_factor
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        else:
            # This is for new experiments that we can put non-uniform changes.
            try:
                coeffs2[:len(ch_factor)] *= ch_factor
            except TypeError:
                coeffs2[:] *= ch_factor
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        X_temp = X_PII[N_PIIs_pre:N_PIIs_pre+num,:]
        N_PIIs_pre += num
        y_temp = 1.0*np.random.poisson(lam=np.exp(np.dot(X_temp, coeffs2)+b0+np.random.normal(0,sigma,num)))
        y_PII = np.append(y_PII, y_temp)

        logger.info("The stepwise possion responses %s with coeffs %s.", y_PII, coeffs2, extra=d)

    return y_PII.astype(np.float32)


def Generate_Logi_Data(mean, cov, sigma, transf_mat, coeffs, N, intcp=0):
    X = Generate_Predictor(mean, cov, N, transf_mat)
    # y has shape (-1,)
    y = (np.random.random(N) > 1 / (1 + np.exp((-1)*np.dot(X, coeffs) - intcp -
                    np.random.normal(0, sigma, N)))).astype('int')
    return X, y


def Generate_Stepwise_Logi_Data(
        mean,
        cov,
        sigma,
        transf_mat,
        coeffs,
        mean2,
        cov2,
        N_PIIs,
        change_factors,
        intcp_ls=None,
        sigma_ls=None):
    X_PII = Generate_Stepwise_Predictor(
            mean,
            cov,
            sigma,
            transf_mat,
            coeffs,
            mean2,
            cov2,
            N_PIIs)

    y_PII = Generate_Stepwise_Logi_Response(
            X_PII,
            sigma,
            transf_mat,
            coeffs,
            N_PIIs,
            mean2,
            cov2,
            change_factors,
            intcp_ls,
            sigma_ls)
    return X_PII, y_PII


def Generate_Stepwise_Logi_Response(
        X_PII,
        sigma,
        transf_mat,
        coeffs,
        N_PIIs,
        mean2,
        cov2,
        change_factors,
        intcp_ls=None,
        sigma_ls=None):
    # X_PII_1 = X_PII[:N_PIIs[0],:]
    # y_PII_1 = (np.random.random(N_PIIs[0]) > 1 / (1 + np.exp((-1)*np.dot(
    #     X_PII_1, coeffs) (-1)*np.random.normal(0, sigma, N_PIIs[0])))).astype('int')

    y_PII = np.array([], dtype=np.int16)
    N_PIIs_pre = 0

    if not intcp_ls:
        intcp_ls = [0]*len(change_factors)

    if not sigma_ls:
        sigma_ls = [sigma]*len(change_factors)

    for num, ch_factor, b0, sig in zip(N_PIIs, change_factors, intcp_ls, sigma_ls):
        coeffs2 = coeffs * (1)
        if len(coeffs2) > 8: # This is for legacy example when 10 predictors exist.
            # coeffs2[0] *= (ch_factor)
            # coeffs2[1] *= (ch_factor)
            # coeffs2[2] *= (ch_factor)
            # coeffs2[3] *= (ch_factor)
            coeffs2[:4] *= ch_factor
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        else:
            # This is for new experiments that we can put non-uniform changes.
            try:
                coeffs2[:len(ch_factor)] *= ch_factor
            except TypeError:
                coeffs2[:] *= ch_factor
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        # print coeffs2
        X_temp = X_PII[N_PIIs_pre:N_PIIs_pre+num,:]
        N_PIIs_pre += num
        y_temp = (np.random.random(num) > 1 /
            (1 + np.exp((-1) * np.dot(X_temp, coeffs2) - b0 -
            np.random.normal(0, sig, num)))).astype('int')
        y_PII = np.append(y_PII, y_temp)

    return y_PII


def Generate_Gradual_Logi_Data(
        mean,
        cov,
        sigma,
        transf_mat,
        coeffs,
        mean2,
        cov2,
        N_PIIs,
        change_factors):
    X_PII = Generate_Stepwise_Predictor(
            mean,
            cov,
            sigma,
            transf_mat,
            coeffs,
            mean2,
            cov2,
            N_PIIs)

    y_PII = Generate_Gradual_Logi_Response(
            X_PII,
            sigma,
            transf_mat,
            coeffs,
            N_PIIs,
            mean2,
            cov2,
            change_factors)
    return X_PII, y_PII


def Generate_Gradual_Logi_Response(
        X_PII,
        sigma,
        transf_mat,
        coeffs,
        N_PIIs,
        mean2,
        cov2,
        change_factors):
    # X_PII_1 = X_PII[:N_PIIs[0],:]
    # y_PII_1 = (np.random.random(N_PIIs[0]) > 1 / (1 + np.exp((-1)*np.dot(
    #     X_PII_1, coeffs) (-1)*np.random.normal(0, sigma, N_PIIs[0])))).astype('int')
    y_PII = np.array([], dtype=np.int16)
    N_PIIs_pre = 0
    coeffs_old = coeffs

    for num, ch_factor in zip(N_PIIs, change_factors):
        coeffs2 = coeffs_old * (1)
        if len(coeffs2) > 8: # This is for legacy example when 10 predictors exist.
            # coeffs2[0] *= (ch_factor)
            # coeffs2[1] *= (ch_factor)
            # coeffs2[2] *= (ch_factor)
            # coeffs2[3] *= (ch_factor)
            coeffs2[:4] *= ch_factor
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        else:
            # This is for new experiments that we can put non-uniform changes.
            try:
                coeffs2[:len(ch_factor)] *= ch_factor
            except TypeError:
                coeffs2[:] *= ch_factor
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        # print coeffs2
        X_temp = X_PII[N_PIIs_pre:N_PIIs_pre+num,:]
        coeffs_dyn = np.zeros_like(X_temp)
        for idx, (coeff1, coeff2) in enumerate(zip(coeffs_old, coeffs2)):
            coeffs_dyn[:, idx] = np.linspace(coeff1, coeff2, num=num)
        N_PIIs_pre += num
        # np.matmul(
        #     np.random.multivariate_normal(
        #         mean2,
        #         cov2,
        #         num).astype('float32'),
        #     transf_mat)
        y_temp = (np.random.random(num) > 1 /
                  (1 + np.exp((-1)*np.sum(X_temp * coeffs_dyn, axis=1, keepdims=False)))).astype('int')
        y_PII = np.append(y_PII, y_temp) # No matter dimension of y_PII and y_temp, it returns a row vector.
        coeffs_old = coeffs2

    return y_PII


def Generate_Multi_Cla_Data(mean, cov, sigma, transf_mat, coeffs, N_samp, K_cla):
    X = Generate_Predictor(mean, cov, N_samp, transf_mat)
    # y has shape (-1,)
    y_1 = np.exp(np.dot(X, coeffs))
    y_2 = y_1 / np.sum(y_1, axis=1, keepdims=True)
    y = np.reshape([np.random.choice(K_cla, size=1, p=pmf) for pmf in y_2], (-1,))
    return X, y

def Generate_Stepwise_Multi_Cla_Data(
        mean,
        cov,
        sigma,
        transf_mat,
        coeffs,
        mean2,
        cov2,
        N_PIIs,
        change_factors,
        K_cla):
    X_PII = Generate_Stepwise_Predictor(
            mean,
            cov,
            sigma,
            transf_mat,
            coeffs,
            mean2,
            cov2,
            N_PIIs)

    y_PII = Generate_Stepwise_Multi_Cla_Response(
            X_PII,
            sigma,
            transf_mat,
            coeffs,
            N_PIIs,
            mean2,
            cov2,
            change_factors,
            K_cla)

    return X_PII, y_PII


def Generate_Stepwise_Multi_Cla_Response(
        X_PII,
        sigma,
        transf_mat,
        coeffs,
        N_PIIs,
        mean2,
        cov2,
        change_factors,
        K_cla):
    # X_PII_1 = X_PII[:N_PIIs[0],:]
    # y_PII_1 = (np.random.random(N_PIIs[0]) > 1 / (1 + np.exp((-1)*np.dot(
    #     X_PII_1, coeffs) (-1)*np.random.normal(0, sigma, N_PIIs[0])))).astype('int')

    # Have to initialize y_PII as empty numpy array with integer dtype, otherwise
    # later appended y_PII will be converted to float but y_PII will be used
    # as index.
    y_PII = np.array([], dtype=np.int16)
    N_PIIs_pre = 0

    for num, ch_factor in zip(N_PIIs, change_factors):
        coeffs2 = coeffs * (1)
        if len(coeffs2) > 8: # This is for legacy example when 10 predictors exist.
            # # coeffs2[0] *= (ch_factor)
            # # coeffs2[1] *= (ch_factor)
            # # coeffs2[2] *= (ch_factor)
            # # coeffs2[3] *= (ch_factor)
            # coeffs2[:4] *= ch_factor

            # The multi-class is special, we cannot use the way of other methods to change the coefficients.
            coeffs2[0, 1] = coeffs2[0, 1] * (ch_factor)
            coeffs2[1, 1] = coeffs2[1, 1] * (ch_factor)
            coeffs2[2, 1] = coeffs2[2, 1] * (ch_factor)
            coeffs2[3, 1] = coeffs2[3, 1] * (ch_factor)
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        else:
            # This is for new experiments that we can put non-uniform changes.
            try:
                coeffs2[:len(ch_factor)] *= ch_factor
            except TypeError:
                coeffs2[:] *= ch_factor
            logger.info('The changed coefficients %s (%s) and change factor %s.', coeffs2, coeffs, ch_factor, extra=d)
        X_temp = X_PII[N_PIIs_pre:N_PIIs_pre+num,:]
        N_PIIs_pre += num
        y_temp_temp_1 = np.exp(np.dot(X_temp, coeffs2))
        y_temp_temp_2 = y_temp_temp_1 / \
            np.sum(y_temp_temp_1, axis=1, keepdims=True)
        # y_temp = np.argmax(y_temp_temp_2, axis=1).astype('int')
        y_temp = np.reshape([np.random.choice(K_cla, size=1, p=pmf)
                             for pmf in y_temp_temp_2], (-1,))
        y_PII = np.append(y_PII, y_temp)

    return y_PII


def Gen_Img_Patches_from_BW_Cla_Parallel(img_arr_folder_path: str, img_arr_fname: str, dest_folder_path: str, wind_hei: int, wind_wid: int, materials_model: str):
    """ Generate images patches for fastai library to train a CNN regression model."""
    if not os.path.exists(dest_folder_path):
        os.makedirs(dest_folder_path)
    # img_arr = open_image(img_path, div=False, convert_mode=convert_mode).data.numpy()[0]
    np2pil_bw = torchvision.transforms.ToPILImage(mode='L')
    # Without the .to(torch.uint8), the np2pil will make some centering and scaling on the floating type tensor of image.
    img_arr = np.genfromtxt(os.path.join(
        img_arr_folder_path, img_arr_fname), delimiter=',')
    img_bw = np2pil_bw(img_arr.astype(np.uint8)*255)
    img_bw.save(os.path.join(img_arr_folder_path,
                             img_arr_fname.split('.')[0]+'_bw.png'))
    img_rgb = img_bw.convert(mode='RGB')
    img_rgb.save(os.path.join(img_arr_folder_path,
                              img_arr_fname.split('.')[0]+'_rgb.png'))
    img_hei, img_wid = img_arr.shape
    print(img_hei, img_wid)
    # The number of pixels in height direction.
    n_hei = (img_hei-2*wind_hei)
    # The number of pixels in horizontal direction.
    n_wid = (img_wid-2*wind_wid)
    n_sample = n_hei*n_wid  # The number of pixels as response.
    img_idx = 0

    if not os.path.exists(dest_folder_path):
        os.makedirs(dest_folder_path)
    ls_labels = ['0', '1']
    for lab in ls_labels:
        dest_lab_folder_path = os.path.join(dest_folder_path, lab)
        # if os.path.exists(dest_lab_folder_path):
        #     shutil.rmtree(dest_lab_folder_path) # Remove non-empty folder
        if not os.path.exists(dest_lab_folder_path):
            os.mkdir(dest_lab_folder_path)
        
    def get_img_idx(ri, ci, wind_hei, wind_wid, img_hei):
        n_hei = img_hei-2*wind_hei
        return (ci-wind_wid)*n_hei+ri-wind_hei
    
    def save_one_col_patch_causal(ci, wind_hei, wind_wid, img_hei, img_wid, img_arr, dest_folder_path, np2pil_bw):
        for ri in range(wind_hei, img_hei-wind_hei):
            img_patch = img_arr[(ri-wind_hei):(ri+1),
                                (ci-wind_wid):(ci+1)].copy()
            img_patch_label = np.uint(img_arr[ri, ci])
            img_patch[wind_hei, wind_wid] = 0
            # The .astype(np.uint8) is necessary to get rgb image correct.
            np2pil_bw(img_patch.astype(np.uint8)*255).convert(mode='RGB').save(os.path.join(
                dest_folder_path, str(img_patch_label)+'/'+str(get_img_idx(ri, ci, wind_hei, wind_wid, img_hei))+'.png'))
            
    def save_one_col_patch_causal_1(ci, wind_hei, wind_wid, img_hei, img_wid, img_arr, dest_folder_path, np2pil_bw):
        for ri in range(wind_hei, img_hei-wind_hei):
            img_patch = img_arr[(ri-wind_hei):(ri+1),
                                (ci-wind_wid):(ci+wind_wid+1)].copy()
            img_patch_label = np.uint(img_arr[ri, ci])
            img_patch[wind_hei, wind_wid:] = 0
            # The .astype(np.uint8) is necessary to get rgb image correct.
            np2pil_bw(img_patch.astype(np.uint8)*255).convert(mode='RGB').save(os.path.join(
                dest_folder_path, str(img_patch_label)+'/'+str(get_img_idx(ri, ci, wind_hei, wind_wid, img_hei))+'.png'))
    
    
    if materials_model == 'causal':
        # The wind_hei and wind_wid are defined as the window height
        # and width excluding the response as the bottom-right cornor.
        ls_tasks = [(ci, wind_hei, wind_wid, img_hei, img_wid, img_arr, dest_folder_path, np2pil_bw) for ci in range(wind_wid, img_wid-wind_wid)]
        with parallel_backend('loky', n_jobs=20):
            Parallel(verbose=5, pre_dispatch='2*n_jobs')(delayed(save_one_col_patch_causal)(*task) for task in ls_tasks)
    elif materials_model == 'causal_1':
        # The definition of wind_hei and wind_wid are in Ramin 2016 paper.
        ls_tasks = [(ci, wind_hei, wind_wid, img_hei, img_wid, img_arr, dest_folder_path, np2pil_bw) for ci in range(wind_wid, img_wid-wind_wid)]
        with parallel_backend('loky', n_jobs=20):
            Parallel(verbose=5, pre_dispatch='2*n_jobs')(delayed(save_one_col_patch_causal_1)(*task) for task in ls_tasks)
        # for ci in range(wind_wid, img_wid-wind_wid):
        #     for ri in range(wind_hei, img_hei-wind_hei):
        #         if img_idx % 1000 == 0:
        #           print(img_idx)
        #         img_patch = img_arr[(ri-wind_hei):(ri+1),
        #                             (ci-wind_wid):(ci+wind_wid+1)].copy()
        #         img_patch_label = np.uint(img_arr[ri, ci])
        #         img_patch[wind_hei, wind_wid:] = 0
        #         # The .astype(np.uint8) is necessary to get rgb image correct.
        #         np2pil_bw(img_patch.astype(np.uint8)*255).convert(mode='RGB').save(os.path.join(
        #             dest_folder_path, str(img_patch_label)+'/'+str(img_idx)+'.png'))
        #         img_idx += 1
    elif materials_model == 'non_causal':
        # The wind_hei and wind_wid are defined as the number of pixels from the target
        # to the left or to the right boundary. And width is the number of pixels from the target
        # to the up and down boundary.
        for ci in range(wind_wid, img_wid-wind_wid):
            for ri in range(wind_hei, img_hei-wind_hei):
                if img_idx % 1000 == 0:
                    print(img_idx)
                img_patch = img_arr[(ri-wind_hei):(ri+wind_hei+1),
                                    (ci-wind_wid):(ci+wind_wid+1)].copy()
                img_patch_label = np.uint(img_arr[ri, ci])
                img_patch[wind_hei, wind_wid] = 0
                np2pil_bw(img_patch.astype(np.uint8)*255).convert(mode='RGB').save(os.path.join(
                    dest_folder_path, str(img_patch_label)+'/'+str(img_idx)+'.png'))
                img_idx += 1

