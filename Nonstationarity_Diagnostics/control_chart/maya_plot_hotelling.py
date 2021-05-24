import numpy as np
import os
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
# plt.switch_backend('agg') # Needed for running on quest
# plt.style.use('Solarize_Light2') # https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
# plt.style.use('_classic_test')
plt.style.use('seaborn-bright') # https://matplotlib.org/tutorials/introductory/customizing.html
from joblib import Parallel, parallel_backend, delayed
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, art3d
from constants import *
from control_chart.utils import *
from collections import OrderedDict
from sklearn.cluster import KMeans
from PIL import Image
from mayavi import mlab

# def Set_Axis_Prop(ax, ls_axis_lab, labelsize=LAB_SIZE, labelpad=LAB_PAD, title="", rot=0):
#     ax.set_xlabel(ls_axis_lab[0], size=0.75*labelsize, labelpad=labelpad)
#     ax.set_ylabel(ls_axis_lab[1], size=0.75*labelsize, labelpad=labelpad)
#     # x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
#     # print(x_lim, y_lim)
#     # if tick_int:
#     #     x_tick_num, y_tick_num = TICK_NUM, TICK_NUM
#     #     while int(x_lim[1]-x_lim[0])%x_tick_num!=0:
#     #         x_tick_num += 1
#     #     while int(y_lim[1]-y_lim[0])%y_tick_num!=0:
#     #         y_tick_num += 1
#     # Set the number of tick labels: https://stackoverflow.com/a/55690467/4307919
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(TICK_NUM))
#     # ax.xaxis.set_minor_locator(ticker.MaxNLocator(x_tick_num*5))
#     ax.yaxis.set_major_locator(ticker.MaxNLocator(TICK_NUM))
#     # ax.yaxis.set_minor_locator(ticker.MaxNLocator(y_tick_num*5))
#     # ax.set_xticks(list(np.arange(*x_lim,(x_lim[1]-x_lim[0])/TICK_NUM)))
#     # ax.set_yticks(list(np.arange(*y_lim,(y_lim[1]-y_lim[0])/TICK_NUM)))
#     ax.xaxis.set_tick_params(labelsize=0.65*labelsize, rotation=rot)
#     ax.yaxis.set_tick_params(labelsize=0.65*labelsize, rotation=rot)
#     if len(ls_axis_lab)==3:
#         ax.set_zlabel(ls_axis_lab[2], size=0.75*labelsize, labelpad=labelpad)
#         # z_lim = ax.get_zlim()
#         # print(z_lim, list(np.arange(*z_lim,(z_lim[1]-z_lim[0])/TICK_NUM)))
#         # ax.set_zticks(list(np.arange(*z_lim,(z_lim[1]-z_lim[0])//TICK_NUM)))
#         ax.zaxis.set_major_locator(ticker.MaxNLocator(TICK_NUM))
#         # ax.zaxis.set_minor_locator(ticker.MaxNLocator(TICK_NUM*5))
#         ax.zaxis.set_tick_params(labelsize=0.65*labelsize)
#         # ax.pbaspect = [1.0, 1.0, 0.2]
#     else:
#         ax.set_aspect(1.0)
    
#     if title!='' and (title is not None):
#         ax.set_title(title, size=labelsize, pad=labelpad)
#         # ax_title.set_position([.5, 1.05])

def PlotSpatial3DHeatMap_Maya(arr_2D, fig, FLAGS=None, vmin=None, vmax=None, cmap=CMAP, 
                              title='', fig_name='', title_size=LAB_SIZE, label_pad=LAB_PAD, upperleft_corner=(0,0), 
                              config_prop=True, thre_flag=False, thre_kwargs={}, arr_thres=None, extent=[0,1,0,1,0,1], ranges=[0,1,0,1,0,1], save_sep=False):
        
    X, Y = np.meshgrid(upperleft_corner[0]+np.arange(arr_2D.shape[1]), upperleft_corner[1]+np.arange(arr_2D.shape[0]))
    # https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#mayavi.mlab.surf
    # https://scipy-lectures.org/packages/3d_plotting/index.html#arbitrary-regular-mesh
    surf = mlab.mesh(X, Y, arr_2D.astype(np.float), figure=fig, colormap=cmap, vmin=vmin, vmax=vmax, line_width=0, opacity=0.8)
    
    if thre_flag:
        ls_fc = [(0,1,0), (1,0,0)]
        for idx in range(len(arr_thres)):
            cl = mlab.mesh(X, Y, np.ones_like(X)*arr_thres[idx], figure=fig, color=ls_fc[idx], line_width=0, opacity=thre_kwargs['alpha'])

    if config_prop:
        cbar = mlab.colorbar(surf, orientation='vertical', nb_labels=0) # Don't show the number in colorbar because they are scaled values (not original values).
        # https://docs.enthought.com/mayavi/mayavi/auto/example_wigner.html
        mlab.outline(surf, figure=fig, extent=extent) # https://docs.enthought.com/mayavi/mayavi/auto/mlab_other_functions.html?highlight=axes#outline
        mlab.axes(surf, figure=fig, extent=extent, ranges=ranges, nb_labels=TICK_NUM, xlabel="X", ylabel="Y", zlabel="Z")
        mlab.view(142, -90, 800)
        # mlab.title(title, size=5)

    if save_sep:
        fig_path = os.path.join(FLAGS.training_res_folder, fig_name)
        mlab.savefig(fig_path)
        mlab.clf(fig)
        return fig_path


def show_image_ax(ax, fpath, title='', size=LAB_SIZE, pad=LAB_PAD):
    img = Image.open(fpath)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, size=size, pad=pad)

def normal_max_min(arr, ls_arr, scale=1):
    vmin, vmax = np.min(arr), np.max(arr)
    return (arr.astype(np.float)-vmin)/(vmax-vmin)*scale, [(arr1.astype(np.float)-vmin)/(vmax-vmin)*scale for arr1 in ls_arr]

def PlotSaveSpatial3DHeatMap_Other_Score_Maya_Prosp(
        img_arr_PI, ls_img_arr_PII,
        arr_t2_scores_spatial_ewma_PI,
        ls_arr_t2_scores_spatial_ewma_PII,
        ls_arr_comp_spatial_ewma_PI,
        ls_ls_arr_comp_spatial_ewma_PII,
        ls_comp_name,
        folder_path,
        fig_name,
        FLAGS,
        cmap=CMAP,
        label_pad = LAB_PAD,
        title_size = LAB_SIZE,
        thre_flag = True,
        alarm_level = 99, # percentail
        thre_kwargs = {'linewidth': 2, 'edgecolor': 'k', 'linestyle': '-', 'alpha': 0.4},
        fig_size = (1200, 900),
        save_sep = False):
    """ Plot 3D heatmap using mayavi."""
    num_comp_metric = len(ls_arr_comp_spatial_ewma_PI)
    num_plt_row = 2+num_comp_metric
    fig = plt.figure(num=None, figsize=(2.4 *  2 * ONE_FIG_HEI, num_plt_row * 2 * ONE_FIG_HEI), dpi=100, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = WSPACE)
    # Phase-I and -II image
    ax = plt.subplot2grid((num_plt_row,2), (0,0))
    ax.imshow(img_arr_PI, cmap = GRAY_CMAP)
    ax.invert_yaxis()
    # ax.set_title('PI image', size = title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PI image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PI_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    ax = plt.subplot2grid((num_plt_row,2), (0,1))
    comb_img_arr_PII = np.block([[ls_img_arr_PII[0], ls_img_arr_PII[1]],[ls_img_arr_PII[2], ls_img_arr_PII[3]]])
    ax.imshow(comb_img_arr_PII, cmap = GRAY_CMAP)
    ax.invert_yaxis()
    # ax.set_title('PII image', size = title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PII image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PII_orig.png'),
                    bbox_inches=extent.expanded(1.4, 1.4))
    # Hotelling T2 scores
    ucl = np.percentile(arr_t2_scores_spatial_ewma_PI, (100+alarm_level)/2)
    lcl = np.percentile(arr_t2_scores_spatial_ewma_PI, (100-alarm_level)/2)
    arr_thres = np.array([lcl, ucl])

    hei, wid = img_arr_PI.shape
    n_hei, n_wid = arr_t2_scores_spatial_ewma_PI.shape
    extent = [0, n_wid, 0, n_hei, 0, n_hei]
    cmap_vmin = min(np.min(arr_t2_scores_spatial_ewma_PI), np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    cmap_vmax = max(np.max(arr_t2_scores_spatial_ewma_PI), np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    cmap_mval = np.array([cmap_vmin, cmap_vmax])
    z_min, z_max = np.min(arr_t2_scores_spatial_ewma_PI), np.max(arr_t2_scores_spatial_ewma_PI)
    ranges = copy.copy(extent)
    ranges[-2:] = [z_min, z_max]
    plt_arr_t2_scores_spatial_ewma_PI, [plt_cmap_mval, plt_arr_thres] = normal_max_min(arr_t2_scores_spatial_ewma_PI, [cmap_mval, arr_thres], n_hei)
    print(plt_cmap_mval, plt_arr_thres)
    ax = plt.subplot2grid((num_plt_row,2), (1,0))
    title = 'Hotelling T2 Score PI'
    fig = mlab.figure(fgcolor=(0,0,0), bgcolor=(1,1,1), size=fig_size)
    fpath = PlotSpatial3DHeatMap_Maya(plt_arr_t2_scores_spatial_ewma_PI, fig, FLAGS, plt_cmap_mval[0], plt_cmap_mval[1], cmap=cmap,
                         title=title, fig_name=fig_name.split('.')[0]+'_PI_score.png', title_size=title_size, 
                         thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=plt_arr_thres, extent=extent, ranges=ranges, save_sep=True)
    show_image_ax(ax, fpath, title=title, size=title_size, pad=label_pad)
    
    hei, wid = ls_img_arr_PII[0].shape
    n_hei, n_wid = ls_arr_t2_scores_spatial_ewma_PII[0].shape
    extent = [0, 2*n_wid, 0, 2*n_hei, 0, 2*n_hei]
    z_min, z_max = min(lcl, np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII])), max(ucl, np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    ranges = copy.copy(extent)
    ranges[-2:] = [z_min, z_max]
    _, ls_arr = normal_max_min(np.concatenate(ls_arr_t2_scores_spatial_ewma_PII, axis=0), ls_arr_t2_scores_spatial_ewma_PII+[cmap_mval, arr_thres], 2*n_hei)
    plt_ls_arr_t2_scores_spatial_ewma_PII, plt_cmap_mval, plt_arr_thres = ls_arr[:4], ls_arr[4], ls_arr[5]
    print(plt_cmap_mval, plt_arr_thres)
    ax = plt.subplot2grid((num_plt_row,2), (1,1))
    title = 'Hotelling T2 Score PII'
    ls_upperleft_corner = [(0, 0),
                           (n_wid, 0),
                           (0, n_hei),
                           (n_wid, n_hei)]
    fig = mlab.figure(fgcolor=(0,0,0), bgcolor=(1,1,1), size=fig_size)
    for idx, (upperleft_corner, arr_t2_scores_spatial_ewma_PII) in enumerate(zip(ls_upperleft_corner, plt_ls_arr_t2_scores_spatial_ewma_PII)):
        # sub_extent = copy.copy(extent)
        # sub_extent[0], sub_extent[1], sub_extent[2], sub_extent[3] = upperleft_corner[0], upperleft_corner[0]+n_wid, upperleft_corner[1], upperleft_corner[1]+n_hei
        if idx==FLAGS.num_PII-1:
            fpath = PlotSpatial3DHeatMap_Maya(arr_t2_scores_spatial_ewma_PII, fig, FLAGS, plt_cmap_mval[0], plt_cmap_mval[1], cmap=cmap,
                         title=title, fig_name=fig_name.split('.')[0]+'_PII_score.png', title_size=title_size, upperleft_corner=upperleft_corner, 
                         thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=plt_arr_thres, extent=extent, ranges=ranges, save_sep=True)
        else:
            fpath = PlotSpatial3DHeatMap_Maya(arr_t2_scores_spatial_ewma_PII-(z_min+z_max)/2, fig, FLAGS, plt_cmap_mval[0], plt_cmap_mval[1], cmap=cmap,
                         title=None, fig_name=None, title_size=title_size, upperleft_corner=upperleft_corner, config_prop=False, 
                         thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=plt_arr_thres, save_sep=False)
    show_image_ax(ax, fpath, title=title, size=title_size, pad=label_pad)

    # Comparing metric
    for idx_comp, (comp_name, arr_comp_spatial_ewma_PI, ls_arr_comp_spatial_ewma_PII) in enumerate(zip(ls_comp_name, ls_arr_comp_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PII)):
        ucl = np.percentile(arr_comp_spatial_ewma_PI, (100+alarm_level)/2)
        lcl = np.percentile(arr_comp_spatial_ewma_PI, (100-alarm_level)/2)
        arr_thres = np.array([lcl, ucl])

        hei, wid = img_arr_PI.shape
        n_hei, n_wid = arr_comp_spatial_ewma_PI.shape
        extent = [0, n_wid, 0, n_hei, 0, n_hei]
        cmap_vmin = min(np.min(arr_comp_spatial_ewma_PI), np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        cmap_vmax = max(np.max(arr_comp_spatial_ewma_PI), np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        cmap_mval = np.array([cmap_vmin, cmap_vmax])
        z_min, z_max = np.min(arr_comp_spatial_ewma_PI), np.max(arr_comp_spatial_ewma_PI)
        ranges = copy.copy(extent)
        ranges[-2:] = [z_min, z_max]
        plt_arr_comp_spatial_ewma_PI, [plt_cmap_mval, plt_arr_thres] = normal_max_min(arr_comp_spatial_ewma_PI, [cmap_mval, arr_thres], n_hei)
        print(plt_cmap_mval, plt_arr_thres)
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,0))
        title = ' '.join([comp_name, 'PI'])
        fig = mlab.figure(fgcolor=(0,0,0), bgcolor=(1,1,1), size=fig_size)
        fpath = PlotSpatial3DHeatMap_Maya(plt_arr_comp_spatial_ewma_PI, fig, FLAGS, plt_cmap_mval[0], plt_cmap_mval[1], cmap=cmap,
                             title=title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, 
                             thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=plt_arr_thres, extent=extent, ranges=ranges, save_sep=True)
        show_image_ax(ax, fpath, title=title, size=title_size, pad=label_pad)
        
        hei, wid = ls_img_arr_PII[0].shape
        n_hei, n_wid = ls_arr_comp_spatial_ewma_PII[0].shape
        extent = [0, 2*n_wid, 0, 2*n_hei, 0, 2*n_hei]
        z_min, z_max = min(lcl, np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII])), max(ucl, np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        ranges = copy.copy(extent)
        ranges[-2:] = [z_min, z_max]
        _, ls_arr = normal_max_min(np.concatenate(ls_arr_comp_spatial_ewma_PII, axis=0), ls_arr_comp_spatial_ewma_PII+[cmap_mval, arr_thres], 2*n_hei)
        plt_ls_arr_comp_spatial_ewma_PII, plt_cmap_mval, plt_arr_thres = ls_arr[:4], ls_arr[4], ls_arr[5]
        print(plt_cmap_mval, plt_arr_thres)
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,1))
        title = ' '.join([comp_name, 'PII'])
        ls_upperleft_corner = [(0, 0),
                               (n_wid, 0),
                               (0, n_hei),
                               (n_wid, n_hei)]
        fig = mlab.figure(fgcolor=(0,0,0), bgcolor=(1,1,1), size=fig_size)
        for idx, (upperleft_corner, arr_comp_spatial_ewma_PII) in enumerate(zip(ls_upperleft_corner, plt_ls_arr_comp_spatial_ewma_PII)):
            if idx==FLAGS.num_PII-1:
                fpath = PlotSpatial3DHeatMap_Maya(arr_comp_spatial_ewma_PII, fig, FLAGS, plt_cmap_mval[0], plt_cmap_mval[1], cmap=cmap,
                             title=title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, upperleft_corner=upperleft_corner, 
                             thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=plt_arr_thres, extent=extent, ranges=ranges, save_sep=True)
            else:
                fpath = PlotSpatial3DHeatMap_Maya(arr_comp_spatial_ewma_PII, fig, FLAGS, plt_cmap_mval[0], plt_cmap_mval[1], cmap=cmap,
                             title=None, fig_name=None, title_size=title_size, upperleft_corner=upperleft_corner, config_prop=False,
                             thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=plt_arr_thres, save_sep=False)
        show_image_ax(ax, fpath, title=title, size=title_size, pad=label_pad)
    
    # plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.savefig(os.path.join(folder_path, fig_name))
    plt.close()


