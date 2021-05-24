# %%
import math
import torch
import gpytorch # Only work for 0.3.5, doesn't work for 1.1.1.
import numpy as np
import pandas as pd
import os
import sys
import time
import matplotlib.pyplot as plt
from PIL import Image
from itertools import product
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from control_chart.data_generation import *


# %% [markdown]
class GridGPPriorGenerator(gpytorch.models.ExactGP):
    """
    Generate prior distribution for GP.
    """

    def __init__(self, grid, train_x, train_y, lengthscale, likelihood, opscale=1):
        """ Generate a GP prior realization just by giving only one pair of training observation and it is far away from all testing data input.
            Args:
                grid: The input value grids to get the GP prior.
                train_x: One example of training x.
                train_y: One example of training y.
                lengthscale: An array of lengthscale along each axis.
                likelihood: The likelihood function to mapping latent function to observation y (e.g., Gaussian for regression).
                outputscale: The scale for the kernel (https://gpytorch.readthedocs.io/en/latest/kernels.html#gpytorch.kernels.RBFKernel).

        """
        super(GridGPPriorGenerator, self).__init__(train_x, train_y, likelihood)
        self.dims = train_x.size(-1)
        self.opscale = opscale
        self.mean_module = gpytorch.means.ConstantMean() # Default is 0 constant mean.
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1))
        print("The default length scale is {}.".format(kernel.lengthscale)) # Default value: 
        # Without the explicit dtype casting, there will be an error.
        kernel.lengthscale = lengthscale.to(torch.float) # Assign the lengthscale values.
        # gpytorch.kernels.ScaleKernel will learn the outputscale from data.
        # opscale_kernel = gpytorch.kernels.ScaleKernel(base_kernel=kernel, outputscale_prior=opscale)
        self.covar_module = gpytorch.kernels.GridKernel(kernel, grid=grid)
        # self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1))
        # self.covar_module.lengthscale = lengthscale.to(torch.float) # Assign the lengthscale values. 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)*self.opscale
        print(mean_x.shape, covar_x.shape)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def ax_plot(y_labels, title, save_path):
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(y_labels, cmap=cm.coolwarm)
    ax.set_title(title)
    f.colorbar(im, shrink=0.8, aspect=10.0)
    plt.savefig(save_path)


def ax_plot_3d(grid_size, z_labels, title, save_path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(grid_size)
    Y = np.arange(grid_size)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, z_labels, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(title)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.8, aspect=10.0)

    plt.savefig(save_path)


# %% [markdown]
# ## Function to generate and save plots of realization of GP.
def Gen_GP_Latent_Func(dim: int, train_x: torch.tensor, train_y: torch.tensor,
                       grid_size: int, lengthscale: torch.tensor, opscale: float=1, noise: float=0):
    """
        dim: The dimension of input space.
        train_x: One example of training x.
        train_y: One example of training y.
        grid_size: The number of grid points along each axis.
        lengthscale: The diagonal parameters of theta for specifying length-scale along each axis: https://gpytorch.readthedocs.io/en/latest/kernels.html#gpytorch.kernels.RBFKernel.
        noise: The variance of noise: https://github.com/cornellius-gp/gpytorch/blob/92e07cf4dae26083fe0aed926e1dfd483443924e/gpytorch/likelihoods/gaussian_likelihood.py#L106.
        opscale: The outputscale for the kernel.

    """
    grid = torch.zeros(grid_size, dim, dtype=torch.float)
    for ci in range(dim):
        grid[:, ci] = torch.arange(grid_size)
    test_x = gpytorch.utils.grid.create_data_from_grid(grid)
    # print("The test inputs has shape {}({},{}).".format(test_x.size(),test_x.size(0),test_x.size(1)))

    # The noise is the variance of the noise: https://gpytorch.readthedocs.io/en/latest/kernels.html#gpytorch.kernels.RBFKernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise*torch.ones(train_x.shape[0]))
    # print(train_x.shape, train_x.size(), train_y.size(), grid.size(), lengthscale.size())
    model = GridGPPriorGenerator(grid, train_x, train_y, lengthscale, likelihood, opscale=opscale)

    model.train()
    likelihood.train()

    print(model.covar_module.base_kernel.lengthscale)
    # print(model.covar_module.lengthscale)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # observed_pred = likelihood(model(test_x), noise=noise*torch.ones(test_x.size(0)))
        observed_pred = likelihood(model(test_x))

    return observed_pred

def Grayscale_Latent_Func(rfield):
    min_v, max_v = np.min(rfield), np.max(rfield)
    rfield = (rfield-min_v)/(max_v-min_v)*255
    return rfield.astype(dtype=np.uint8)

def Binary_Latent_Func(rfield, thre=0.5):
    logit_val_arr = rfield
    # gp_scale = 1  # The multiplier for the logit_val_arr.
    logit_val_arr += np.log(thre/(1-thre))
    proba_val_arr = 1/(1+np.exp(-logit_val_arr))
    binary_img_arr = (0.5 <= proba_val_arr)*255
    return binary_img_arr.astype(dtype=np.uint8)

def Binary_Rand_Latent_Func(rfield, thre=0.5):
    logit_val_arr = rfield
    # gp_scale = 1  # The multiplier for the logit_val_arr.
    logit_val_arr += np.log(thre/(1-thre))
    proba_val_arr = 1/(1+np.exp(-logit_val_arr))
    grid_size = logit_val_arr.shape[0]
    rand_binary_img_arr = (np.random.rand(grid_size, grid_size) <= proba_val_arr)*255
    return rand_binary_img_arr.astype(dtype=np.uint8)

def Gen_Save_GP_Micro_Struct(dim: int, train_x: torch.tensor, train_y: torch.tensor, 
                             grid_size: int, lengthscale: torch.tensor, opscale: float, noise: float,
                             lat_func: object, save_path: str, postfix: str = "", lat_func_kwargs: dict = {}):
    gp_proc = Gen_GP_Latent_Func(dim, train_x, train_y, grid_size, lengthscale, opscale, noise)

    start_time = time.time()
    # rfield_tensor = gp_proc.rsample(sample_shape=torch.Size((grid_size, grid_size)))
    rfield_tensor = gp_proc.rsample().view(grid_size, grid_size)
    print(rfield_tensor.size())
    print("The prediction for the 40000 grid points takes time: {}s.\n".format(
        time.time()-start_time,))
    print("The random realization is:\n {}.\n".format(rfield_tensor,))
    rfield = rfield_tensor.detach().numpy()
    rfield += np.random.normal(scale=noise**0.5, size=(grid_size, grid_size))
    print(rfield.shape, np.mean(rfield), np.std(rfield))

    # Postfix for plots.
    if postfix != "":
        postfix += '_'
    postfix += '_'.join([str(int(lengthscale[0])), str(int(lengthscale[1]))])
    save_subfolder_path = os.path.join(save_path, 'gp_obse_'+postfix)
    if not os.path.exists(save_subfolder_path):
        os.makedirs(save_subfolder_path)

    # ## Have to use .detach().numpy(), ow I will get error: "Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead."
    ax_plot(rfield, "One realization of GP param_"+postfix,
            os.path.join(save_subfolder_path, "one_gp_obse_{}.png".format(postfix,)))

    # ## Plot 3D surface
    ax_plot_3d(grid_size, rfield, "One realization of GP (3D) param_"+postfix,
               os.path.join(save_subfolder_path, "one_gp_obse_3d_{}.png".format(postfix,)))

    # Latent function to micro-structure pixel observations.
    img_arr = lat_func(rfield, **lat_func_kwargs)

    img = Image.fromarray(img_arr)

    img.save(os.path.join(save_subfolder_path, "one_gp_obse_micro_struct_{}.png".format(postfix,)))

    return img
    

# %%
# Normalizing the shading of images
def Normal_Img_Median(img, median_value=127.5, shrink_ratio=0.8):
    img_arr = np.array(img)
    print("Mean, median, min, max and std before scaling: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}.".format(np.mean(img_arr), np.median(img_arr), np.min(img_arr), np.max(img_arr), np.std(img_arr)))
    img_arr = img_arr/255
    img_arr = (img_arr-np.median(img_arr))*shrink_ratio*255 + median_value
    # bw_img_arr = img_arr
    img_arr = img_arr.astype(np.uint8)
    print("Mean, median, min, max and std after scaling: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}.".format(np.mean(img_arr), np.median(img_arr), np.min(img_arr), np.max(img_arr), np.std(img_arr)))
    return img_arr, Image.fromarray(img_arr, mode='P')


# %%
def gen_func_reg(val):
    return np.float(val)

def gen_func_exp(val):
    return np.exp(val)

def gen_func_cla(val):
    return (1*(np.random.random(1) < 1/(1+np.exp((-1)*val)))).astype(np.int)[0]

def gen_func_max_cap_exp(val, cap=5.0):
    return np.min([np.exp(val), cap])

def gen_func_sigmoid(val):
    return 1/(1+np.exp(-val))

def gen_func_max_cap(val, cap=1.2):
    return np.min([val, cap])

def gen_func_min_cap(val, cap=-1.8):
    return np.max([val, cap])

def gen_func_max_min_cap(val, min_cap=-1.1, max_cap=1.1):
    return np.min([np.max([val, min_cap]), max_cap])

def gen_func_max_min_cap_exp(val, min_cap=0.05, max_cap=5.0):
    return np.min([np.max([np.exp(val), min_cap]), max_cap])

# ## Calculate eigen-values of vector-autoregressive models.
def Gen_AR_Mat(vec: np.ndarray):
    dim = vec.shape[0]
    # I realized that previously the coefficient has wrong order.
    return np.vstack((vec[::-1], np.eye(dim-1, dim, 0)))

# Generate AR 2D Regression data with some noise.
def Gen_AR_2D_Noisy_Reg_Data(gen_func, latent_gen_func, sigma, intcp, ext_img_hei, ext_img_wid, coeffs, FLAGS, noise_level=0, normal_flag=True):
    coeffs_mat = Gen_AR_Mat(coeffs)
    eigvs = np.linalg.eigvals(coeffs_mat)
    max_abs_eigv = np.max(np.abs(eigvs))
    if max_abs_eigv >=1:
        print("!!!The largest eigenvalues for the 2D AR is not less than 1 so that the 2D AR model is not stable!!!{}({})".format(eigvs, max_abs_eigv))
    else:
        print("The eigenvalues of this 2D AR model are {}({}).".format(eigvs, max_abs_eigv))

    init_vals = np.random.normal(0, sigma, (ext_img_hei, ext_img_wid))
    nois_mean, _ = Cal_AR_2D_Mean_Std_Given_Intcp(
        gen_func_cla, sigma, intcp, FLAGS.gen_wind_hei, FLAGS.gen_wind_wid, init_vals, coeffs) # Specifically, use gen_func_cla here, not gen_func_reg
    print(nois_mean, intcp, sigma)
    init_vals = np.random.normal(nois_mean, sigma, (ext_img_hei, ext_img_wid))
    _, y_temp, z_arr, img_arr = Generate_AR_2D_Data(
        gen_func, sigma, FLAGS.gen_wind_hei, FLAGS.gen_wind_wid, init_vals, coeffs, intcp=intcp, latent_gen_func=latent_gen_func)
    noise_arr = np.random.normal(size=img_arr.shape)*255
    img_arr = img_arr.astype(np.float32) # img_arr is [0, 255] with saturated gray-scale.
    img_arr += noise_arr.astype(np.float32)*noise_level
    if normal_flag:
        img_arr = (img_arr-np.mean(img_arr))/np.std(img_arr)
    return img_arr

