import numpy as np
import os
import logging
import seaborn as sns
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# Follow this link to enable plotting with latex text in figures
# https://matplotlib.org/tutorials/text/usetex.html?highlight=latex
# mpl.rcParams.update(mpl.rcParamsDefault) # Sometimes this would mess up jupyter notebook in terms of showing plots.
plt.switch_backend('Agg') # Needed for running on quest
# plt.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath,bm}"] # https://stackoverflow.com/a/14324826/4307919

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s(%(funcName)s)[%(lineno)d]: %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'zkg'}
logger = logging.getLogger('hotelling')
logging.getLogger('hotelling').setLevel(logging.INFO)

marker = ['o', 'v', '^', '<', '>']
line_patt = ['b--', 'g-.', 'r:', 'm,', 'c.', 'y--,', 'o', 'v', '^', '<', '>']
line_style = OrderedDict(
    [('dotted', (0, (1, 5))),
     ('dashed', (0, (5, 5))),
        ('densely dotted', (0, (1, 1))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),

        ('loosely dashed', (0, (5, 10))),

        ('densely dashed', (0, (5, 1))),
        ('loosely dotted', (0, (1, 10))),

        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
        ('solid', (0, ()))])
line_colors = ['xkcd:blue', 'xkcd:red', 'xkcd:green', 'xkcd:cyan', 'xkcd:orange', 'xkcd:magenta', 'xkcd:brown', 'xkcd:purple', 'xkcd:fuchsia']


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def set_xticks_xticklabels(ax, time_stamp, time_step, label_size, rotation=ROTATION, to_decode=False):
    ax.set_xticklabels(['']*len(ax.get_xticks())) # Set the xtick as empty
    ls_xtick_pos = []
    for i in np.arange(time_stamp[0]+1, time_stamp[-1]+1, time_step):
        j = np.argmax(time_stamp >= i)
        ls_xtick_pos.append(j)
        ax.text(j, ax.get_ylim()[0]-YR_TICK_MARGIN*(ax.get_ylim()[1]-ax.get_ylim()[0]), #ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                MonIdx2Date(time_stamp[j], date_index_flag=True, to_decode=to_decode),
                rotation=rotation, fontsize=label_size * AX_LAB_SCALE * 1.5)
    ax.set_xticks(ls_xtick_pos)


def ScoresSpatialEWMA(scores, n_hei, n_wid, sigma, wind_len, n_jobs=N_JOBS):
    """ Calculate spatial EWMA of scores or some metrics at each loacation of 2D image.
    
        Args:
            scores: An array of scores of 2D image at each pixel (in order of row-by-row in a raster mode).
            n_hei: The number of pixels in height direction.
            n_wid: The number of pixels in width direction.
            sigma: The length scale of exponential weight (Gaussian weight).
            wind_len: The window length is 2*wind_len+1 for spatial EWMA.

        Returns:
            score_spatial_ewma_arr: The array of spatial EWMA using window length of size wind_len.

    """
    def cal_exp_weight_wind(wind_len, sigma):
        """ Calculate weights for a window of size wind_len."""
        exp_weight_wind = np.zeros((2*wind_len+1, 2*wind_len+1))
        cent = np.array([wind_len, wind_len])
        for ri in range(2*wind_len+1):
            for ci in range(2*wind_len+1):
                pos = np.array([ri, ci])
                exp_weight_wind[ri, ci] = np.exp(-np.sum((pos-cent)**2)/2.0/sigma**2)
        exp_weight_wind /= np.sum(exp_weight_wind)
        return exp_weight_wind

    # def find_score(pos, scores):
    #     return scores[n_hei*pos[1]+pos[0]]
    def find_score(pos, scores):
        """ Find the score given the index position in the image.
        
            Previously, the scores are filled col-by-col of the 2D image. Later, I change it to row-by-row of the image.
        """
        return scores[n_wid*pos[0]+pos[1]]

    def spatial_ewma(cent, wind_len, scores, exp_weight_wind, row_idx):
        """ Calculate the spatial EWMA given the pixel location as cent.

            Args:
                cent: The location of pixel where EWMA need to calculate.
                wind_len: The EWMA window size.
                scores: The array of scores.
                exp_weight_wind: The weight array of EWMA.

            Returns:
                score_spatial_ewma: The spatical EWMA of score at location cent. 
        """
        score_spatial_ewma = np.zeros_like(scores[0]).astype(np.float)
        for ri in range(2*wind_len+1):
            for ci in range(2*wind_len+1):
                pos = [cent[0]+ri-wind_len, cent[1]+ci-wind_len]
                score_spatial_ewma += find_score(pos, scores) * exp_weight_wind[ri, ci]
        return score_spatial_ewma, row_idx

    n_ewma_hei, n_ewma_wid = n_hei-2*wind_len, n_wid-2*wind_len

    exp_weight_wind = cal_exp_weight_wind(wind_len, sigma)
    # logger.info("The spatial EWMA weights matix is %s.\n", exp_weight_wind, extra=d)
    
    # score_spatial_ewma_arr = np.zeros((n_ewma_hei*n_ewma_wid, scores.shape[1]))
    # row_idx = 0
    # # The score_spatial_ewma_arr are filled row-by-row of the image.
    # for cent_ri in range(wind_len, n_hei-wind_len):
    #     for cent_ci in range(wind_len, n_wid-wind_len):
    #         cent = [cent_ri, cent_ci]
    #         score_spatial_ewma_arr[row_idx,:] = spatial_ewma(cent, wind_len, scores, exp_weight_wind)
    #         row_idx += 1

    ls_tasks = [([cent_ri, cent_ci], wind_len, scores, exp_weight_wind, ridx*n_ewma_wid+cidx) 
                for ridx, cent_ri in enumerate(range(wind_len, n_hei-wind_len)) 
                    for cidx, cent_ci in enumerate(range(wind_len, n_wid-wind_len))]
    with parallel_backend('loky', n_jobs=n_jobs):
        score_spatial_ewma_row_idx = Parallel(verbose=2, pre_dispatch="2*n_jobs")(delayed(spatial_ewma)(*task) for task in ls_tasks)
    
    logger.info("The dimension of metric is %s.", scores.shape[-1], extra=d)
    logger.info("The shape before and after ewma calculateion: (%s, %s, %s, %s).\n", n_hei, n_wid, n_ewma_hei, n_ewma_wid, extra=d)
    # logger.info("(%s, %s) The first 2 scores: %s.\n", sigma, wind_len, scores[:2], extra=d)
    # logger.info("(%s, %s) The first 2 ewma scores: %s.\n", sigma, wind_len, score_spatial_ewma_row_idx[:2], extra=d)
    
    score_spatial_ewma_row_idx.sort(key=lambda x: x[1]) # Sort incrementally inplace
    ls_score_spatial_ewma, ls_row_idx = zip(*score_spatial_ewma_row_idx)
    score_spatial_ewma_arr = np.array(ls_score_spatial_ewma)
    
    # row_idx_arr = np.array(ls_row_idx)
    # score_spatial_ewma_arr = score_spatial_ewma_arr[row_idx_arr,:]

    # Return in a form of 2D spatial arr corresponding to the original spatial locations.
    return score_spatial_ewma_arr.reshape((n_ewma_hei, n_ewma_wid, score_spatial_ewma_arr.shape[-1]))


def SpatialHotellingT2Retro(scores, n_hei, n_wid, sigma, wind_len, fig_name, FLAGS, save_sep=False, reshape_2d_arr=False):
    """ Retrospective analysis. """
    # Calculate spatial EWMA
    score_spatial_ewma_arr = ScoresSpatialEWMA(scores, n_hei, n_wid, sigma, wind_len)
    # Calculate spatial Hotelling T2
    Sinv = Inv_Cov(scores, FLAGS.nugget)
    # Because we need to average the neighboring window of scores, there is a margin of 
    # EWMA window size. In this margin, the scores don't have spatial EWMA.
    t2_n_hei, t2_n_wid = n_hei-2*wind_len, n_wid-2*wind_len
    t2_scores_spatial_ewma_arr = np.zeros((t2_n_hei, t2_n_wid))
    score_mu = np.mean(scores, axis = 0)

    # The t2_scores_spatial_ewma_arr is filled row-by-row of the image.
    for ri in range(t2_n_hei):
        for ci in range(t2_n_wid):
            # The score_spatial_ewma_arr is filled row-by-row of the image.
            t2_scores_spatial_ewma_arr[ri, ci] = HotellingT2(score_spatial_ewma_arr[ri, ci], score_mu, Sinv)

    PlotSaveSpatialHeatMap(t2_scores_spatial_ewma_arr, FLAGS.training_res_folder, fig_name)

    Spatial3DScoreRetro(score_spatial_ewma_arr.reshape(-1,score_spatial_ewma_arr.shape[-1]), scores.mean(axis=0), n_hei, n_wid, sigma, wind_len, fig_name, FLAGS, plot_flag=False, n_comp=FLAGS.n_comp, save_sep=save_sep)

    Spatial3DScoreRetro(score_spatial_ewma_arr.reshape(-1,score_spatial_ewma_arr.shape[-1]), scores.mean(axis=0), n_hei, n_wid, sigma, wind_len, fig_name[:-4]+'_loc_info.png', FLAGS, plot_flag=False, n_comp=FLAGS.n_comp, loc_coord_flag=True, save_sep=save_sep)

    visu_fig_name = '3D_score_visu.png'
    PlotNormColorMap(score_spatial_ewma_arr.reshape(-1,score_spatial_ewma_arr.shape[-1]), visu_fig_name, FLAGS)

    if reshape_2d_arr:
        score_spatial_ewma_arr = score_spatial_ewma_arr.reshape(-1,score_spatial_ewma_arr.shape[-1]) 

    return t2_scores_spatial_ewma_arr, score_spatial_ewma_arr


# def SpatialHotellingT2RetroInspect(scores, n_hei, n_wid, sigma, wind_len, ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, margin_coef=0.5):
#     """ Retrospective analysis. For the purpose of inspection. """
#     # Calculate spatial EWMA
#     score_spatial_ewma_arr = ScoresSpatialEWMA(scores, n_hei, n_wid, sigma, wind_len)
#     # Calculate spatial Hotelling T2
#     Sinv = Inv_Cov(scores, FLAGS.nugget)
#     # Because we need to average the neighboring window of scores, there is a margin of 
#     # EWMA window size. In this margin, the scores don't have spatial EWMA.
#     t2_n_hei, t2_n_wid = n_hei-2*wind_len, n_wid-2*wind_len
#     t2_scores_spatial_ewma_arr = np.zeros((t2_n_hei, t2_n_wid))
#     score_mu = np.mean(scores, axis = 0)

#     # The t2_scores_spatial_ewma_arr is filled row-by-row of the image.
#     for ri in range(t2_n_hei):
#         for ci in range(t2_n_wid):
#             # The score_spatial_ewma_arr is filled row-by-row of the image.
#             t2_scores_spatial_ewma_arr[ri, ci] = HotellingT2(score_spatial_ewma_arr[t2_n_wid*ri + ci], score_mu, Sinv)

#     visu_fig_name = '3D_score_visu.png'
#     PlotNormColorMap(score_spatial_ewma_arr, visu_fig_name, FLAGS)

#     PlotSaveSpatialHeatMap(t2_scores_spatial_ewma_arr, FLAGS.training_res_folder, fig_name)

#     Spatial3DScoreRetroInspect(score_spatial_ewma_arr, scores.mean(axis=0), n_hei, n_wid, sigma, wind_len, ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, margin_coef=margin_coef, plot_flag=False, n_comp=FLAGS.n_comp)

#     Spatial3DScoreRetroInspect(score_spatial_ewma_arr, scores.mean(axis=0), n_hei, n_wid, sigma, wind_len, ls_row_grid_pts, ls_col_grid_pts, fig_name[:-4]+'_loc_info.png', FLAGS, margin_coef=margin_coef, plot_flag=False, n_comp=FLAGS.n_comp, loc_coord_flag=True)

#     return t2_scores_spatial_ewma_arr, score_spatial_ewma_arr


def PlotNormColorMap(score_spatial_ewma_arr, visu_fig_name, FLAGS, label_size=LAB_SIZE, plot_flag=False):
    ewma_n_hei, ewma_n_wid = FLAGS.moni_stat_hei-2*FLAGS.spatial_ewma_wind_len, FLAGS.moni_stat_wid-2*FLAGS.spatial_ewma_wind_len
    # Plot eigen-values
    fig_name = "try.png"
    score_ewma_centered_arr, eigvects = PlotEigenValues(score_spatial_ewma_arr, FLAGS.mu_train, fig_name, 10, FLAGS)
    # 3D scattering
    score_ewma_3D_arr = np.matmul(score_ewma_centered_arr, eigvects[:,:-4:-1])
    norm_score_ewma_3D_arr = np.sum(score_ewma_3D_arr**2, axis=1)**0.5

    # Plot new visualization.
    R, C = np.arange(1, ewma_n_hei+1), np.arange(1, ewma_n_wid+1)
    C, R = np.meshgrid(C, R) # Have to be in this order
    Z = norm_score_ewma_3D_arr.reshape((ewma_n_hei, ewma_n_wid)) # The score is in razer mode.
    print(R.shape, C.shape, Z.shape)
    min_lmt, max_lmt = np.min(score_ewma_3D_arr, axis=0), np.max(score_ewma_3D_arr, axis=0)
    facecolors = score_ewma_3D_arr.reshape((ewma_n_hei, ewma_n_wid, -1))
    facecolors = (facecolors-min_lmt)/(max_lmt-min_lmt)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(R, C, Z, facecolors=facecolors, linewidth=0.1, antialiased=False)
    plt.title('3D visualization for scores', size=label_size)
    ax.set_xticklabels(['']*len(ax.get_xticks()))
    ax.set_yticklabels(['']*len(ax.get_yticks()))
    ax.set_zticklabels(['']*len(ax.get_zticks()))
    plt.savefig(os.path.join(FLAGS.training_res_folder, visu_fig_name), bbox_inches='tight')
    ax.view_init(elev=90., azim=0.)
    plt.savefig(os.path.join(FLAGS.training_res_folder, visu_fig_name[:-4]+'_top.png'), bbox_inches='tight')
    
    if plot_flag:
        plt.show()


def Spatial3DScoreRetro(score_spatial_ewma_arr, mu_score, n_hei, n_wid, sigma, wind_len, fig_name, FLAGS, 
                        label_size=LAB_SIZE, plot_flag=False, n_comp=4, clustering_verbose=0, 
                        rand_col_flag=False, loc_coord_flag=False, save_sep=False):
    # Because we need to average the neighboring window of scores, there is a margin of 
    # EWMA window size. In this margin, the scores don't have spatial EWMA.
    t2_n_hei, t2_n_wid = n_hei-2*wind_len, n_wid-2*wind_len
    t2_scores_spatial_ewma_arr = np.zeros((t2_n_hei, t2_n_wid))

    # Plot clustering of scores. 
    # Do clustering first and then dimension reduction. (0)
    # Do dimension reduction first and then do clustering. (1)
    for ord_dr_cl in [1, 0]:
        if loc_coord_flag:
            arr_loc_coord = FLAGS.loc_coord_wei*np.array([[r, c] for r in range(t2_n_hei) for c in range(t2_n_wid)])
            score_ewma_loc_coord_arr = np.hstack((score_spatial_ewma_arr, arr_loc_coord))
            mu_loc_coord_train = np.hstack((FLAGS.mu_train, np.mean(arr_loc_coord, axis=0)))
            PlotSaveClusteringSpatialScore3D(score_ewma_loc_coord_arr, mu_loc_coord_train, t2_n_hei, t2_n_wid, fig_name, FLAGS, 
                                             label_size=label_size,
                                             title='Score clustering\n({}, {})'.format('dr-cl' if ord_dr_cl else 'cl-dr', 'loc' if loc_coord_flag else 'no-loc'),
                                             n_comp=n_comp, plot_flag=plot_flag, 
                                             clustering_verbose=clustering_verbose, ord_dr_cl=ord_dr_cl, rand_col_flag=rand_col_flag, 
                                             loc_coord_flag=loc_coord_flag, save_sep=save_sep)
        else:
            PlotSaveClusteringSpatialScore3D(score_spatial_ewma_arr, mu_score, t2_n_hei, t2_n_wid, fig_name, FLAGS, 
                                             label_size=label_size,
                                             title='Score clustering\n({}, {})'.format('dr-cl' if ord_dr_cl else 'cl-dr', 'loc' if loc_coord_flag else 'no-loc'),
                                             n_comp=n_comp, plot_flag=plot_flag, 
                                             clustering_verbose=clustering_verbose, ord_dr_cl=ord_dr_cl, rand_col_flag=rand_col_flag, 
                                             loc_coord_flag=loc_coord_flag, save_sep=save_sep)


# def Spatial3DScoreRetroInspect(score_spatial_ewma_arr, mu_score, n_hei, n_wid, sigma, wind_len, ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, margin_coef=0.5, plot_flag=False, n_comp=4, clustering_verbose=0, loc_coord_flag=False):
#     """ 3D spatial score retrospective analysis. For the purpose of inspection. """
#     # Because we need to average the neighboring window of scores, there is a margin of 
#     # EWMA window size. In this margin, the scores don't have spatial EWMA.
#     t2_n_hei, t2_n_wid = n_hei-2*wind_len, n_wid-2*wind_len
#     t2_scores_spatial_ewma_arr = np.zeros((t2_n_hei, t2_n_wid))

#     ls_t2_row_grid_pts = [max(0, ele-wind_len) for ele in ls_row_grid_pts]
#     ls_t2_col_grid_pts = [max(0, ele-wind_len) for ele in ls_col_grid_pts]
    
#     PlotSaveSpatialScore3DScatterInspect(score_spatial_ewma_arr, mu_score, t2_n_hei, t2_n_wid, ls_t2_row_grid_pts, ls_t2_col_grid_pts, fig_name, FLAGS, margin_coef=margin_coef, plot_flag=plot_flag)

#     # Plot arrows with fixed colors.
#     PlotSaveSpatialScore3DArrowInspect(score_spatial_ewma_arr, mu_score, t2_n_hei, t2_n_wid, ls_t2_row_grid_pts, ls_t2_col_grid_pts, fig_name, FLAGS, margin_coef=margin_coef, color_flag='fixed', plot_flag=plot_flag)

#     # Plot arrows with colormap based on scores.
#     PlotSaveSpatialScore3DArrowInspect(score_spatial_ewma_arr, mu_score, t2_n_hei, t2_n_wid, ls_t2_row_grid_pts, ls_t2_col_grid_pts, fig_name, FLAGS, margin_coef=margin_coef, color_flag='map', plot_flag=plot_flag)

#     # Plot clustering of scores. 
#     # Do clustering first and then dimension reduction. (0)
#     # Do dimension reduction first and then do clustering. (1)
#     for ord_dr_cl in [1, 0]:
#         if loc_coord_flag:
#             arr_loc_coord = FLAGS.loc_coord_wei*np.array([[r, c] for r in range(t2_n_hei) for c in range(t2_n_wid)])
#             score_ewma_loc_coord_arr = np.hstack((score_spatial_ewma_arr, arr_loc_coord))
#             mu_loc_coord_train = np.hstack((FLAGS.mu_train, np.mean(arr_loc_coord, axis=0)))
#             PlotSaveClusteringSpatialScore3DInspect(score_ewma_loc_coord_arr, mu_loc_coord_train, t2_n_hei, t2_n_wid, 
#                                              ls_t2_row_grid_pts, ls_t2_col_grid_pts, fig_name, FLAGS, 
#                                              title='Score clustering ({}, {})'.format('dr-cl' if ord_dr_cl else 'cl-dr', 'loc' if loc_coord_flag else 'no-loc'),
#                                              n_comp=n_comp, plot_flag=plot_flag, 
#                                              clustering_verbose=clustering_verbose, ord_dr_cl=ord_dr_cl, 
#                                              loc_coord_flag=loc_coord_flag)
#         else:
#             PlotSaveClusteringSpatialScore3DInspect(score_spatial_ewma_arr, mu_score, t2_n_hei, t2_n_wid, 
#                                              ls_t2_row_grid_pts, ls_t2_col_grid_pts, fig_name, FLAGS,
#                                              title='Score clustering ({}, {})'.format('dr-cl' if ord_dr_cl else 'cl-dr', 'loc' if loc_coord_flag else 'no-loc'), 
#                                              n_comp=n_comp, plot_flag=plot_flag, 
#                                              clustering_verbose=clustering_verbose, ord_dr_cl=ord_dr_cl, 
#                                              loc_coord_flag=loc_coord_flag)


def PlotEigenValues(score_spatial_ewma_arr, mu_score, fig_name, num_eigvs, FLAGS, plot_flag=False, save_fig=False):
    # Plot eigen-values.
    print(score_spatial_ewma_arr.shape, mu_score.shape)
    score_ewma_centered_arr = score_spatial_ewma_arr-mu_score
    cov_score_ewma = np.matmul(score_ewma_centered_arr.T, score_ewma_centered_arr)
    eigvs, eigvects = np.linalg.eigh(cov_score_ewma)
    if save_fig:
        plt.figure(num=None, figsize=(ONE_FIG_HEI, ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
        plt.plot(np.arange(num_eigvs), eigvs[:-(num_eigvs+1):-1])
        plt.title('Eigenvalues')
        plt.savefig(os.path.join(FLAGS.training_res_folder, 'eigvalues_'+fig_name), bbox_inches='tight')
        if plot_flag:
            plt.show()
    return score_ewma_centered_arr, eigvects


# def PlotSaveTSScore3DScatter(score_spatial_ewma_arr, mu_score, ls_grid_pts, fig_name, FLAGS, num_eigvs=10):
#     margin = [0, int(FLAGS.spatial_ewma_sigma)] # The size of margin that will not be plotted in scattering figures.
#     PlotSaveSpatialScore3DScatterInspect(score_spatial_ewma_arr, mu_score, 1, score_spatial_ewma_arr.shape[0], [0], ls_grid_pts, fig_name, margin, FLAGS, num_eigvs)


def CalSpatialScore3D(score_spatial_ewma_arr, mu_score, n_hei, n_wid, ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, num_eigvs=10, margin_coef=1):
    """ Calculate 2D spatial scores after PCA to see clustering of scores. 
    
        Args:
            ls_row_grid_pts: The starting idx of row for separating different blocks.
            ls_col_grid_pts: The starting idx of col for separating different blocks.
            
    """
    # Add a margin so that we don't plot the monitoring statistics in the margins (near the transition boundaries).
    margin = [int(max(FLAGS.spatial_ewma_sigma, FLAGS.wind_hei)*margin_coef)]*2 # The size of margin that will not be plotted in scattering figures.
    
    # Plot eigen-values
    score_ewma_centered_arr, eigvects = PlotEigenValues(score_spatial_ewma_arr, mu_score, fig_name, num_eigvs, FLAGS)
    
    # 3D scattering
    score_ewma_3d_arr = np.matmul(score_ewma_centered_arr, eigvects[:,:-4:-1])
    print("The number of element is {} (expected: {}).".format(score_ewma_3d_arr.shape[0], n_hei*n_wid))
    ls_row_block_sizes, ls_col_block_sizes = np.diff(ls_row_grid_pts+[n_hei]), np.diff(ls_col_grid_pts+[n_wid])
    ls_block_spatial_coords = []
    ls_block_score_coords = []
    print(margin)
    for rs, rl in zip(ls_row_grid_pts, ls_row_block_sizes):
        for cs, cl in zip(ls_col_grid_pts, ls_col_block_sizes):
            rs_margin, cs_margin, rl_margin, cl_margin = rs+margin[0], cs+margin[1], rl-margin[0], cl-margin[1]
            ulc_idx = rs_margin*n_wid+cs_margin # upper-left corner index
            ls_temp0 = [[r, c] for r in range(rs_margin, rs_margin+rl_margin) for c in range(cs_margin, cs_margin+cl_margin)]
            ls_temp1 = [np.arange(rs_temp, rs_temp+cl_margin) for rs_temp in np.arange(ulc_idx, ulc_idx+rl_margin*n_wid, n_wid)]
            ls_block_spatial_coords.append(np.array(ls_temp0))
            ls_block_score_coords.append(score_ewma_3d_arr[np.array(ls_temp1).reshape(-1,), :])
    return ls_block_spatial_coords, ls_block_score_coords


def PlotSaveSpatialScore3DScatterInspect(score_spatial_ewma_arr, mu_score, n_hei, n_wid, ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, num_eigvs=10, margin_coef=1, plot_flag=False):
    """ Plot and save 2D spatial scores after PCA to see clustering of scores. 
        The different colors for clustering information are based on true labels. 
        This is for inspection purpose.
    
        Args:
            ls_row_grid_pts: The starting idx of row for separating different blocks.
            ls_col_grid_pts: The starting idx of col for separating different blocks.
            
    """
    _, ls_block_score_coords = CalSpatialScore3D(score_spatial_ewma_arr, mu_score, n_hei, n_wid, ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, num_eigvs=num_eigvs, margin_coef=margin_coef)

    fig = plt.figure(num=None, figsize=(ONE_FIG_HEI, ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    ax = plt.subplot2grid((1,1), (0,0), projection='3d')
    rand_line_colors = line_colors[:len(ls_block_score_coords)]
    np.random.shuffle(rand_line_colors)
    for i, b_coord in enumerate(ls_block_score_coords):
        ax.scatter(b_coord[:,0],
                   b_coord[:,1],
                   b_coord[:,2], 
                   c=rand_line_colors[i%len(rand_line_colors)],
                   marker=marker[0],
                   s=0.5)
    ax.set_xlabel('$PC1$')
    ax.set_ylabel('$PC2$')
    ax.set_zlabel('$PC3$')
    markers, labels = ax.get_legend_handles_labels()
    ax.legend(markers, labels)
    plt.title('3D score scatter plots')

    fig_name_3d = '3D_score_scatter_plots_'+fig_name
    plt.savefig(os.path.join(FLAGS.training_res_folder, fig_name_3d), bbox_inches='tight')
    pickle.dump(fig, open(os.path.join(FLAGS.training_res_folder, fig_name_3d.replace('.png', '.h5')), 'wb'))
    
    if plot_flag:
        plt.show()


def PlotSaveSpatialScore3DArrowInspect(score_spatial_ewma_arr, mu_score, n_hei, n_wid, ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, num_eigvs=10, step=10, scale=5, spa_scale=0.01, margin_coef=0.5, color_flag='map', plot_flag=False):
    ls_block_spatial_coords, ls_block_score_coords = CalSpatialScore3D(score_spatial_ewma_arr, mu_score, n_hei, n_wid, ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, num_eigvs=num_eigvs, margin_coef=margin_coef)
        
    block_max_score_coord = np.array([np.max(arr, axis=0) for arr in ls_block_score_coords])
    block_min_score_coord = np.array([np.min(arr, axis=0) for arr in ls_block_score_coords])
    max_score_coord, min_score_coord = np.max(block_max_score_coord, axis=0), np.min(block_min_score_coord, axis=0)
    
    fig = plt.figure(num=None, figsize=(ONE_FIG_HEI, ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    ax = plt.subplot2grid((1,1), (0,0), projection='3d')
    rand_line_colors = line_colors[:len(ls_block_score_coords)]
    np.random.shuffle(rand_line_colors)
    for i, (b_spa_coord, b_sco_coord) in enumerate(zip(ls_block_spatial_coords, ls_block_score_coords)):
        for spa_coord, sco_coord in zip(b_spa_coord[::step,:], b_sco_coord[::step,:]):
            scaled_spa_coord, scaled_sco_coord = spa_scale*spa_coord, scale*sco_coord
            arr_color = (sco_coord-min_score_coord)/(max_score_coord-min_score_coord) if color_flag=='map' else rand_line_colors[i%len(rand_line_colors)]
            a = Arrow3D([scaled_spa_coord[0],scaled_spa_coord[0]+scaled_sco_coord[0]], 
                        [scaled_spa_coord[1],scaled_spa_coord[1]+scaled_sco_coord[1]], 
                        [0,scaled_sco_coord[2]],
                        mutation_scale=20, lw=0.5, arrowstyle="->", color=arr_color)
            ax.add_artist(a)
        min_x, max_x, min_y, max_y = np.min(b_spa_coord[:,0])*spa_scale, np.max(b_spa_coord[:,0])*spa_scale, np.min(b_spa_coord[:,1])*spa_scale, np.max(b_spa_coord[:,1])*spa_scale
        face_vert = [(min_x, min_y, 0), (min_x, max_y, 0), (max_x, max_y, 0), (max_x, min_y, 0)]
        face = art3d.Poly3DCollection([face_vert], alpha=0.2, linewidth=0.5)
        alpha = 0.7
        face.set_facecolor((0,0,0,alpha))
        ax.add_collection3d(face)

    ax.set_xlabel('$X$')
    ax.set_xlim((20*spa_scale, 160*spa_scale)) 
    ax.set_ylabel('$Y$')
    ax.set_ylim((20*spa_scale, 160*spa_scale))
    ax.set_zlabel('$Z$')
    ax.set_zlim((-0.3, 0.3))
    markers, labels = ax.get_legend_handles_labels()
    ax.legend(markers, labels)
    plt.title('3D arrow plot for scores')
    plt.draw()
    plt.show()
    fig_name_3d = ('3D_score_colored_arrow_plots_' if color_flag=='map' else '3D_score_arrow_plots_') + fig_name
    plt.savefig(os.path.join(FLAGS.training_res_folder, fig_name_3d), bbox_inches='tight')
    pickle.dump(fig, open(os.path.join(FLAGS.training_res_folder, fig_name_3d.replace('.png', '.h5')), 'wb'))
    
    if plot_flag:
        plt.show()


def PlotSaveClusteringTSScore3D(score_spatial_ewma_arr, mu_score, ls_grid_pts, fig_name, FLAGS, 
                                title="", num_eigvs=10, n_comp=4, plot_flag=False, clustering_verbose=0, 
                                ord_dr_cl=1, loc_coord_flag=False):
    """ Do clustering on score for time-series data comparing with ground truth. """
    # Plot eigen-values
    score_ewma_centered_arr, eigvects = PlotEigenValues(score_spatial_ewma_arr, mu_score, fig_name, num_eigvs, FLAGS, plot_flag=plot_flag, save_fig=True)
    
    # 3D scattering
    score_ewma_3d_arr = np.matmul(score_ewma_centered_arr, eigvects[:,:-4:-1])
    
    km_inertia = []
    start_time = time.time()
    for n_clusters in range(1, FLAGS.km_max_n_clusters+1):
        print(n_clusters)
        km_cluster = KMeans(n_clusters=n_clusters, verbose=clustering_verbose, random_state=FLAGS.rand_seed, n_jobs=N_JOBS)
        if ord_dr_cl:
            # Do dimension reduction first, then clustering.
            km_cluster.fit(score_ewma_3d_arr)
        else:
            # Do clustering first then dimension reduction.
            km_cluster.fit(score_spatial_ewma_arr)
        km_inertia.append(km_cluster.inertia_)
    print("The time took to run clustering is {}s.".format(time.time()-start_time,))
        
    fig = plt.figure(num=None, figsize=(ONE_FIG_HEI, ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    ax = plt.subplot2grid((1,1), (0,0))
    ax.plot(range(1,len(km_inertia)+1), km_inertia)
    ax.set_title('K-means inertia vs\n number of components')
    plt.xticks(np.arange(1, FLAGS.km_max_n_clusters+1, step=1))
    
    if ord_dr_cl:
        fig_name_cluster = 'clustering_inertia_dr_cl_plots_'+fig_name
    else:
        fig_name_cluster = 'clustering_inertia_cl_dr_plots_'+fig_name

    plt.savefig(os.path.join(FLAGS.training_res_folder, fig_name_cluster), bbox_inches='tight')
    pickle.dump(fig, open(os.path.join(FLAGS.training_res_folder, fig_name_cluster.replace('.png', '.h5')), 'wb'))

    if plot_flag:
        plt.show()
    
    # Plot 3D clustering of scores.
    km_cluster = KMeans(n_clusters=n_comp, verbose=clustering_verbose, random_state=FLAGS.rand_seed, n_jobs=N_JOBS)
    if ord_dr_cl:
        km_cluster.fit(score_ewma_3d_arr)
    else:
        km_cluster.fit(score_spatial_ewma_arr)
    print("The inertia is {}.".format(km_cluster.inertia_,))
    fig = plt.figure(num=None, figsize=(2*ONE_FIG_HEI, ONE_FIG_HEI), dpi=300, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = 0)
    # 3D clustering
    ax = fig.add_subplot(121, projection='3d') # NotImplementedError: It is not currently possible to manually set the aspect on 3D axes
    # ax = plt.subplot2grid((1,1), (0,0), projection='3d')
    uni_labels = np.unique(km_cluster.labels_)
    rand_line_colors = line_colors[:uni_labels.shape[0]]
    np.random.shuffle(rand_line_colors)
    for i, lab in enumerate(uni_labels):
        b_coord = score_ewma_3d_arr[np.where(km_cluster.labels_==lab)[0],:]
        ax.scatter(b_coord[:,0],
                   b_coord[:,1],
                   b_coord[:,2],
                   c=rand_line_colors[i%len(rand_line_colors)],
                   marker=marker[0],
                   s=0.5)
    ax.set_xlabel('$PC1$', size=AX_LAB_SCALE*LAB_SIZE)
    ax.set_ylabel('$PC2$', size=AX_LAB_SCALE*LAB_SIZE)
    ax.set_zlabel('$PC3$', size=AX_LAB_SCALE*LAB_SIZE)
    markers, labels = ax.get_legend_handles_labels()
    ax.legend(markers, labels)
    ax.set_title("3D score clustering results", size=0.7*LAB_SIZE)
    # ax.set_tick_params(labelbottom=False, labelleft=False, labeltop=False, labelright=False)
    # Looking for api in page: https://matplotlib.org/api/axes_api.html
    ax.set_xticklabels(['']*len(ax.get_xticks()))
    ax.set_yticklabels(['']*len(ax.get_yticks()))
    ax.set_zticklabels(['']*len(ax.get_zticks()))
    ax.grid(True)

    # Plot the clustering results with ground truth
    _, ls_block_score_coords = CalSpatialScore3D(score_spatial_ewma_arr, mu_score, 1, score_spatial_ewma_arr.shape[0], [0], ls_grid_pts, fig_name, FLAGS, num_eigvs=num_eigvs, margin_coef=0)

    ax = fig.add_subplot(122, projection='3d')
    for i, b_coord in enumerate(ls_block_score_coords):
        ax.scatter(b_coord[:,0],
                   b_coord[:,1],
                   b_coord[:,2], 
                   c=line_colors[i%len(line_colors)],
                   marker=marker[0],
                   s=0.5)
    ax.set_xlabel('$PC1$', size=AX_LAB_SCALE*LAB_SIZE)
    ax.set_ylabel('$PC2$', size=AX_LAB_SCALE*LAB_SIZE)
    ax.set_zlabel('$PC3$', size=AX_LAB_SCALE*LAB_SIZE)
    markers, labels = ax.get_legend_handles_labels()
    ax.legend(markers, labels)
    ax.set_title("3D score clustering ground truth", size=0.7*LAB_SIZE)
    ax.set_xticklabels(['']*len(ax.get_xticks()))
    ax.set_yticklabels(['']*len(ax.get_yticks()))
    ax.set_zticklabels(['']*len(ax.get_zticks()))
    # plt.title('3D score scatter plots (ground truth)')

    plt.tight_layout()
    fig_name_3d = '3D_score_scatter_plots_'+fig_name
    plt.savefig(os.path.join(FLAGS.training_res_folder, fig_name_3d), bbox_inches='tight')
    pickle.dump(fig, open(os.path.join(FLAGS.training_res_folder, fig_name_3d.replace('.png', '.h5')), 'wb'))
    
    if plot_flag:
        plt.show()


def PlotSaveClusteringSpatialScore3D(score_spatial_ewma_arr, mu_score, n_hei, n_wid, fig_name, FLAGS, 
                                     label_size=LAB_SIZE,
                                     title="", num_eigvs=10, n_comp=4, plot_flag=False, clustering_verbose=0, 
                                     ord_dr_cl=1, loc_coord_flag=False, rand_col_flag=False, save_sep=False):
    """ Do clustering on score with or without spatial location information. """
    # Plot eigen-values
    fig_name_cluster = 'loc_info_{}_'.format(loc_coord_flag) + fig_name
    score_ewma_centered_arr, eigvects = PlotEigenValues(score_spatial_ewma_arr, mu_score, fig_name_cluster, num_eigvs, FLAGS, plot_flag=plot_flag)
    
    # 3D scattering
    score_ewma_3d_arr = np.matmul(score_ewma_centered_arr, eigvects[:,:-4:-1])
    print("The number of element is {} (expected: {}).".format(score_ewma_3d_arr.shape[0], n_hei*n_wid))
    
    km_inertia = []
    start_time = time.time()
    for n_clusters in range(1, FLAGS.km_max_n_clusters+1):
        print(n_clusters)
        km_cluster = KMeans(n_clusters=n_clusters, verbose=clustering_verbose, random_state=FLAGS.rand_seed, n_jobs=N_JOBS)
        if ord_dr_cl:
            # Do dimension reduction first, then clustering.
            km_cluster.fit(score_ewma_3d_arr)
        else:
            # Do clustering first then dimension reduction.
            km_cluster.fit(score_spatial_ewma_arr)
        km_inertia.append(km_cluster.inertia_)
    print("The time took to run clustering is {}s.".format(time.time()-start_time,))
        
    fig = plt.figure(num=None, figsize=(ONE_FIG_HEI, ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    ax = plt.subplot2grid((1,1), (0,0))
    ax.plot(range(1,len(km_inertia)+1), km_inertia)
    ax.set_title('K-means inertia vs\n number of components')
    plt.xticks(np.arange(1, FLAGS.km_max_n_clusters+1, step=1))
    
    if ord_dr_cl:
        fig_name_cluster = 'clustering_inertia_dr_cl_plots_'+fig_name
    else:
        fig_name_cluster = 'clustering_inertia_cl_dr_plots_'+fig_name

    plt.savefig(os.path.join(FLAGS.training_res_folder, fig_name_cluster), bbox_inches='tight')
    pickle.dump(fig, open(os.path.join(FLAGS.training_res_folder, fig_name_cluster.replace('.png', '.h5')), 'wb'))

    if plot_flag:
        plt.show()
    
    # Plot 3D clustering of scores.
    km_cluster = KMeans(n_clusters=n_comp, verbose=clustering_verbose, random_state=FLAGS.rand_seed, n_jobs=N_JOBS)
    if ord_dr_cl:
        km_cluster.fit(score_ewma_3d_arr)
    else:
        km_cluster.fit(score_spatial_ewma_arr)
    print("The inertia is {}.".format(km_cluster.inertia_,))
    fig = plt.figure(num=None, figsize=(2*ONE_FIG_HEI, 2*ONE_FIG_HEI), dpi=300, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = HSPACE)
    # 3D clustering
    ax = fig.add_subplot(221, projection='3d') # NotImplementedError: It is not currently possible to manually set the aspect on 3D axes
    # ax = plt.subplot2grid((1,1), (0,0), projection='3d')
    uni_labels = np.unique(km_cluster.labels_)
    rand_line_colors = line_colors[:uni_labels.shape[0]]
    if rand_col_flag:
        np.random.shuffle(rand_line_colors)
    for i, lab in enumerate(uni_labels):
        b_coord = score_ewma_3d_arr[np.where(km_cluster.labels_==lab)[0],:]
        ax.scatter(b_coord[:,0],
                   b_coord[:,1],
                   b_coord[:,2],
                   c=rand_line_colors[i%len(rand_line_colors)],
                   marker=marker[0],
                   s=0.5)
    ax.set_xlabel('$PC1$', size=0.75*label_size)
    ax.set_ylabel('$PC2$', size=0.75*label_size)
    ax.set_zlabel('$PC3$', size=0.75*label_size)
    markers, labels = ax.get_legend_handles_labels()
    ax.legend(markers, labels)
    ax.set_title(title, size=label_size)
    # ax.set_tick_params(labelbottom=False, labelleft=False, labeltop=False, labelright=False)
    # Looking for api in page: https://matplotlib.org/api/axes_api.html
    ax.set_xticklabels(['']*len(ax.get_xticks()))
    ax.set_yticklabels(['']*len(ax.get_yticks()))
    ax.set_zticklabels(['']*len(ax.get_zticks()))
    ax.grid(True)
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, 
                                 fig_name.split('.')[0]+'_score_clustering.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    

    # Clustering on original figure coordinate
    ax = fig.add_subplot(222, aspect=1)
    for i, lab in enumerate(uni_labels):
        idx_arr = np.where(km_cluster.labels_==lab)[0]
        coord_x, coord_y = idx_arr % n_wid, idx_arr // n_wid
        ax.scatter(coord_x, coord_y, c=rand_line_colors[i%len(rand_line_colors)], marker=marker[0], s=0.5, alpha=1.0)
    ax.invert_yaxis()
    ax.set_title('Clustering in image', size=label_size)
    ax.xaxis.set_tick_params(labelsize=0.7*label_size)
    ax.yaxis.set_tick_params(labelsize=0.7*label_size)
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, 
                                 fig_name.split('.')[0]+'_clustering_image.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))

    # Show original image
    ax = fig.add_subplot(223, aspect=1)
    orig_img = np.genfromtxt(FLAGS.real_img_abs_path, delimiter=',')
    # orig_img = Image.open(FLAGS.real_img_abs_path).convert(mode='LA')
    ax.imshow(orig_img, cmap = GRAY_CMAP)
    ax.set_title('Original image', size=label_size)
    ax.xaxis.set_tick_params(labelsize=0.7*label_size)
    ax.yaxis.set_tick_params(labelsize=0.7*label_size)
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, 
                                 fig_name.split('.')[0]+'_original_image.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))

    # Show clustering labels on original image
    ax = fig.add_subplot(224, aspect=1)
    ax.imshow(orig_img, cmap = GRAY_CMAP)
    mask = np.zeros_like(orig_img)
    orig_hei, orig_wid = orig_img.shape
    print(orig_hei, orig_wid, n_hei, n_wid)
    min_x, min_y, max_x, max_y = n_wid, n_hei, 0, 0
    for i, lab in enumerate(uni_labels):
        idx_arr = np.where(km_cluster.labels_==lab)[0]
        coord_x, coord_y = idx_arr % n_wid-n_wid//2+orig_wid//2, idx_arr // n_wid-n_hei//2+orig_hei//2
        min_x, min_y, max_x, max_y = min(np.min(coord_x), min_x), min(np.min(coord_y), min_y), max(np.max(coord_x), max_x), max(np.max(coord_y), max_y)
        ax.scatter(coord_x, coord_y, c=rand_line_colors[i%len(rand_line_colors)], marker=marker[0], s=0.5, alpha=0.3)
        mask[coord_y, coord_x] = lab # Generate masks of predicted segmentation
    ax.set_title('Clustering in image\nwith the original', size=label_size)
    ax.xaxis.set_tick_params(labelsize=0.7*label_size)
    ax.yaxis.set_tick_params(labelsize=0.7*label_size)
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, 
                                 fig_name.split('.')[0]+'_cl_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.5))
    
    # # Save original image and masks.
    Image.fromarray(orig_img.astype(np.uint8)).save(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_seg_orig.png'))
    Image.fromarray(orig_img[min_y:max_y+1,min_x:max_x+1,...].astype(np.uint8)).save(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_seg_orig_cropped.png'))
    Image.fromarray(mask.astype(np.uint8)).save(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_seg_mask.png'))
    Image.fromarray(mask[min_y:max_y+1,min_x:max_x+1,...].astype(np.uint8)).save(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_seg_mask_cropped.png'))

    # Save figure
    if ord_dr_cl:
        # Firt dr then cl.
        fig_name_3d = '3D_score_clustering_dr_cl_plots_'+fig_name
    else:
        fig_name_3d = '3D_score_clustering_cl_dr_plots_'+fig_name

    # https://stackoverflow.com/a/45161551/4307919
    plt.tight_layout(rect=[0, -0.1, 1, 1.1])
    plt.savefig(os.path.join(FLAGS.training_res_folder, fig_name_3d), bbox_inches='tight')
    # plt.savefig(os.path.join(FLAGS.training_res_folder, fig_name_3d))
    pickle.dump(fig, open(os.path.join(FLAGS.training_res_folder, fig_name_3d.replace('.png', '.h5')), 'wb'))
    
    if plot_flag:
        plt.show()

    return km_cluster 

# def PlotSaveClusteringSpatialScore3DInspect(score_spatial_ewma_arr, mu_score, n_hei, n_wid, ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, title="", num_eigvs=10, n_comp=4, plot_flag=False, clustering_verbose=0, ord_dr_cl=1, loc_coord_flag=False):
#     """ Do clustering on score with or without spatial location information. For the purpose of inspection. """

#     km_cluster = PlotSaveClusteringSpatialScore3D(score_spatial_ewma_arr, mu_score, n_hei, n_wid, ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, title=title,
#                                                   num_eigvs=num_eigvs, n_comp=n_comp, plot_flag=plot_flag, clustering_verbose=clustering_verbose, ord_dr_cl=ord_dr_cl, loc_coord_flag=loc_coord_flag)    

#     # Calculate inertia with or without location coordinates information.
#     arr_loc_coord = FLAGS.loc_coord_wei*np.array([[r, c] for r in range(n_hei) for c in range(n_wid)])
#     arr_loc_coord_centered = arr_loc_coord-np.mean(arr_loc_coord, axis=0)
#     norm_arr_loc_coord = np.diag(np.dot(arr_loc_coord_centered.T, arr_loc_coord_centered))**0.5
#     arr_loc_coord_scaled = arr_loc_coord_centered/norm_arr_loc_coord 
#     score_paral_loc_coord = np.dot(arr_loc_coord_scaled, np.dot(arr_loc_coord_scaled.T, score_ewma_3d_arr))
#     score_orth_loc_coord = score_ewma_3d_arr-score_paral_loc_coord
    
#     manu_inertia = 0
#     if loc_coord_flag:
#         extra_loc_inertia = 0
#         if ord_dr_cl:
#             for lab in np.unique(km_cluster.labels_):
#                 block_score_paral_loc_coord = score_paral_loc_coord[np.where(km_cluster.labels_==lab)[0],:]
#                 block_score_orth_loc_coord = score_orth_loc_coord[np.where(km_cluster.labels_==lab)[0],:]
#                 manu_inertia += np.sum((block_score_orth_loc_coord-np.mean(block_score_orth_loc_coord, axis=0))**2)
#                 extra_loc_inertia += np.sum((block_score_paral_loc_coord-np.mean(block_score_paral_loc_coord, axis=0))**2)
#         else:
#             for lab in np.unique(km_cluster.labels_):
#                 block_score = score_spatial_ewma_arr[np.where(km_cluster.labels_==lab)[0],:]
#                 manu_inertia += np.sum((block_score[:,:-2]-np.mean(block_score[:,:-2], axis=0))**2)
#                 extra_loc_inertia += np.sum((block_score[:,-2:]-np.mean(block_score[:,-2:], axis=0))**2)
#     else:
#         for lab in np.unique(km_cluster.labels_):
#             if ord_dr_cl:
#                 block_score = score_ewma_3d_arr[np.where(km_cluster.labels_==lab)[0],:]
#             else:
#                 block_score = score_spatial_ewma_arr[np.where(km_cluster.labels_==lab)[0],:]
#             manu_inertia += np.sum((block_score-np.mean(block_score, axis=0))**2)

#     # Calculate inertia with priori location information.
#     manu_priori_inertia = 0
#     ls_block_spatial_coords, ls_block_score_coords = CalSpatialScore3D(
#         score_spatial_ewma_arr, mu_score, n_hei, n_wid,
#         ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, margin_coef=0)
#     if loc_coord_flag:
#         if ord_dr_cl:
#             for block_loc in ls_block_spatial_coords:
#                 arr_idx = block_loc[:,0]*n_wid+block_loc[:,1]
#                 block_score_paral_loc_coord = score_paral_loc_coord[arr_idx,:]
#                 block_score_orth_loc_coord = score_orth_loc_coord[arr_idx,:]
#                 manu_priori_inertia += np.sum((block_score_orth_loc_coord-np.mean(block_score_orth_loc_coord, axis=0))**2)
#             (FLAGS.loc_info_dr_cl_km_inertia, FLAGS.loc_info_dr_cl_recal_km_inertia, FLAGS.loc_info_dr_cl_manu_inertia, FLAGS.loc_info_dr_cl_manu_priori_inertia) = (
#                 km_cluster.inertia_, km_cluster.inertia_-extra_loc_inertia, manu_inertia, manu_priori_inertia)
#         else:
#             for block_loc in ls_block_spatial_coords:
#                 arr_idx = block_loc[:,0]*n_wid+block_loc[:,1]
#                 block_score = score_spatial_ewma_arr[arr_idx,:-2]
#                 manu_priori_inertia += np.sum((block_score-np.mean(block_score, axis=0))**2)
#             (FLAGS.loc_info_cl_dr_km_inertia, FLAGS.loc_info_cl_dr_recal_km_inertia, FLAGS.loc_info_cl_dr_manu_inertia, FLAGS.loc_info_cl_dr_manu_priori_inertia) = (
#                 km_cluster.inertia_, km_cluster.inertia_-extra_loc_inertia, manu_inertia, manu_priori_inertia)
#         print("The built-in, calibrated, manually calculated, and manually calculated priori inertia are {}, {}, {}, {}.".format(
#             km_cluster.inertia_, km_cluster.inertia_-extra_loc_inertia, manu_inertia, manu_priori_inertia))
#     else:
#         if ord_dr_cl:
#             for block_loc in ls_block_spatial_coords:
#                 arr_idx = block_loc[:,0]*n_wid+block_loc[:,1]
#                 block_score = score_ewma_3d_arr[arr_idx,:]
#                 manu_priori_inertia += np.sum((block_score-np.mean(block_score, axis=0))**2)
#             (FLAGS.dr_cl_km_inertia, FLAGS.dr_cl_manu_inertia, FLAGS.dr_cl_manu_priori_inertia) = (
#                 km_cluster.inertia_, manu_inertia, manu_priori_inertia)
#         else:
#             for block_loc in ls_block_spatial_coords:
#                 arr_idx = block_loc[:,0]*n_wid+block_loc[:,1]
#                 block_score = score_spatial_ewma_arr[arr_idx,:]
#                 manu_priori_inertia += np.sum((block_score-np.mean(block_score, axis=0))**2)
#             (FLAGS.cl_dr_km_inertia, FLAGS.cl_dr_manu_inertia, FLAGS.cl_dr_manu_priori_inertia) = (
#                 km_cluster.inertia_, manu_inertia, manu_priori_inertia)
    
#         print("The built-in, manually calculated, and manually calculated priori inertia are {}, {}, {}.".format(
#             km_cluster.inertia_, manu_inertia, manu_priori_inertia))


# def PlotSaveClusteringSpatialScore3DInspect1(score_spatial_ewma_arr, mu_score, n_hei, n_wid, ls_row_grid_pts, ls_col_grid_pts, fig_name, FLAGS, num_eigvs=10, n_comp=4, plot_flag=False, clustering_verbose=0):
#     """ Do clustering first and then do the dimension reduction. """
#     # Plot eigen-values
#     score_ewma_centered_arr, eigvects = PlotEigenValues(score_spatial_ewma_arr, mu_score, fig_name, num_eigvs, FLAGS)
    
#     # 3D scattering
#     score_ewma_3d_arr = np.matmul(score_ewma_centered_arr, eigvects[:,:-4:-1])
#     print("The number of element is {} (expected: {}).".format(score_ewma_3d_arr.shape[0], n_hei*n_wid))
    
#     km_inertia = []
#     start_time = time.time()
#     for n_clusters in range(1, FLAGS.km_max_n_clusters+1):
#         print(n_clusters)
#         km_cluster = KMeans(n_clusters=n_clusters, verbose=clustering_verbose)
#         km_cluster.fit(score_spatial_ewma_arr)
#         km_inertia.append(km_cluster.inertia_)
#     print("The time took to run clustering is {}s.".format(time.time()-start_time,))
        
#     fig = plt.figure(num=None, figsize=(10, 10), dpi=DPI, facecolor='w', edgecolor='k')
#     ax = plt.subplot2grid((1,1), (0,0))
#     ax.plot(range(1,len(km_inertia)+1), km_inertia)
#     plt.xticks(np.arange(1, FLAGS.km_max_n_clusters+1, step=1))
#     fig_name_cluster = 'clustering_inertia_cl_dr_plots_'+fig_name 
#     if not plot_flag:
#         plt.savefig(os.path.join(FLAGS.training_res_folder, fig_name_cluster))
#         pickle.dump(fig, open(os.path.join(FLAGS.training_res_folder, fig_name_cluster.replace('.png', '.h5')), 'wb'))
#         plt.close()
#     else:
#         plt.show()
    
#     # Plot 3D clustering of scores.
#     km_cluster = KMeans(n_clusters=n_comp, verbose=clustering_verbose)
#     km_cluster.fit(score_spatial_ewma_arr)
#     print("The inertia is {}.".format(km_cluster.inertia_,))
#     fig = plt.figure(num=None, figsize=(10, 10), dpi=DPI, facecolor='w', edgecolor='k')
#     ax = plt.subplot2grid((1,1), (0,0), projection='3d')
#     uni_labels = np.unique(km_cluster.labels_)
#     for i, lab in enumerate(uni_labels):
#         b_coord = score_ewma_3d_arr[np.where(km_cluster.labels_==lab)[0],:]
#         ax.scatter(b_coord[:,0],
#                    b_coord[:,1],
#                    b_coord[:,2], 
#                    c=line_colors[i%len(line_colors)],
#                    marker=marker[i%len(marker)],
#                    s=0.5)
#     ax.set_xlabel('$PC1$')
#     ax.set_ylabel('$PC2$')
#     ax.set_zlabel('$PC3$')
#     markers, labels = ax.get_legend_handles_labels()
#     ax.legend(markers, labels)
#     fig_name_3d = '3D_score_clustering_cl_dr_plots_'+fig_name
#     if not plot_flag:
#         plt.savefig(os.path.join(FLAGS.training_res_folder, fig_name_3d))
#         pickle.dump(fig, open(os.path.join(FLAGS.training_res_folder, fig_name_3d.replace('.png', '.h5')), 'wb'))
#         plt.close()
#     else:
#         plt.show()


def SpatialBlockwise(
        comp_stat, 
        n_hei, 
        n_wid, 
        ls_row_grid_pts, 
        ls_col_grid_pts,
        nugget,
        fig_name,
        FLAGS):
    # Calculate spatial blockwise average of comp_stat.
    comp_stat = comp_stat.reshape((n_hei, n_wid))
    comp_stat_blockwise_ave_arr = np.zeros((n_hei, n_wid))
    ls_row_block_sizes, ls_col_block_sizes = np.diff(ls_row_grid_pts+[n_hei]), np.diff(ls_col_grid_pts+[n_wid])
    arr_block_ave_comp_stat = np.zeros(0)
    for rs, rl in zip(ls_row_grid_pts, ls_row_block_sizes):
        for cs, cl in zip(ls_col_grid_pts, ls_col_block_sizes):
            block_ave_comp_stat = comp_stat[rs:rs+rl, cs:cs+cl].mean()
            comp_stat_blockwise_ave_arr[rs:rs+rl, cs:cs+cl] = block_ave_comp_stat
            arr_block_ave_comp_stat = np.hstack((arr_block_ave_comp_stat, block_ave_comp_stat))
    print(("The block average comp_stat are: {}.\n".format(arr_block_ave_comp_stat,)))
    
    PlotSaveSpatialHeatMap(comp_stat_blockwise_ave_arr, FLAGS.training_res_folder, fig_name)
    
    return comp_stat_blockwise_ave_arr


def SpatialBlockwiseT2(
        scores, 
        n_hei, 
        n_wid, 
        ls_row_grid_pts, 
        ls_col_grid_pts,
        nugget,
        mu_train, Sinv_train,
        fig_name,
        FLAGS):
    # Calculate spatial blockwise average of score.
    score_dim = scores[0].shape[0]
    scores = scores.reshape((n_hei, n_wid, score_dim))
    t2_scores_blockwise_ave_arr = np.zeros((n_hei, n_wid))
    ls_row_block_sizes, ls_col_block_sizes = np.diff(ls_row_grid_pts+[n_hei]), np.diff(ls_col_grid_pts+[n_wid])
    arr_block_ave_scores = np.zeros((0, score_dim))
    arr_t2_block_ave_scores = np.zeros(0)
    for rs, rl in zip(ls_row_grid_pts, ls_row_block_sizes):
        for cs, cl in zip(ls_col_grid_pts, ls_col_block_sizes):
            print((rs, rl, cs, cl))
            block_ave_score = scores[rs:rs+rl, cs:cs+cl, :].mean(axis=(0,1))
            t2_scores_blockwise_ave_arr[rs:rs+rl, cs:cs+cl] = HotellingT2(block_ave_score, mu_train, Sinv_train)
            arr_block_ave_scores = np.vstack((arr_block_ave_scores, block_ave_score))
            arr_t2_block_ave_scores = np.hstack((arr_t2_block_ave_scores, t2_scores_blockwise_ave_arr[rs, cs]))
    print(("The block average scores are: {}.\n".format(arr_block_ave_scores,)))
    print(("The block average scores t2 are: {}.\n".format(arr_t2_block_ave_scores,)))
    
    PlotSaveSpatialHeatMap(t2_scores_blockwise_ave_arr, FLAGS.training_res_folder, fig_name)
    
    return t2_scores_blockwise_ave_arr


def SpatialHotellingEWMAT2(
        scores, 
        n_hei, 
        n_wid,
        sigma,
        wind_len,
        nugget,
        mu_train, Sinv_train,
        FLAGS,
        fig_name='',
        fplot=False):
    # Return the shapre corresponding to the spatial shape.
    # Calculate spatial EWMA.
    # logger.info("%s: (%s, %s) The first 2 scores(%s): %s.\n", FLAGS.rand_seed, sigma, wind_len, np.array(scores).shape, scores[:2], extra=d)
    score_spatial_ewma_arr = ScoresSpatialEWMA(scores, n_hei, n_wid, sigma, wind_len)
    # logger.info("%s: (%s, %s) The first 2 scores: %s.\n", FLAGS.rand_seed, sigma, wind_len, scores[:2], extra=d)
    # logger.info("%s: (%s, %s) The first 2 ewma scores(%s): %s.\n", FLAGS.rand_seed, sigma, wind_len, np.array(score_spatial_ewma_arr).shape, score_spatial_ewma_arr[:2], extra=d)
    # print("The first 10 scores: {}.".format(scores))
    # print("The first 10 ewma scores: {}.".format(score_spatial_ewma_arr))
    # Calculate spatial Hotelling T2
    t2_n_hei, t2_n_wid = n_hei-2*wind_len, n_wid-2*wind_len
    t2_scores_spatial_ewma_arr = np.zeros((t2_n_hei, t2_n_wid))
    
    # The t2_scores_spatial_ewma is filled row-by-row.
    start_time = time.time()
    for ri in range(t2_n_hei):
        for ci in range(t2_n_wid):
            t2_scores_spatial_ewma_arr[ri, ci] = HotellingT2(score_spatial_ewma_arr[ri,ci], mu_train, Sinv_train)
    logger.info("The time for calculating %s T2 is %s s.", scores.shape, time.time()-start_time, extra=d)

    if fplot:
        PlotSaveSpatialHeatMap(t2_scores_spatial_ewma_arr, FLAGS.training_res_folder, fig_name)

    return t2_scores_spatial_ewma_arr

def SpatialHotellingEWMAT2_Other_Score_Multi_Img_Prosp(
        ls_img_arr_PI, ls_img_arr_PII,
        ls_score_PI, ls_score_PII,
        ls_ls_comp_PI, ls_ls_comp_PII,
        n_hei_PI, n_wid_PI,
        n_hei_PII, n_wid_PII,
        ewma_sigma, ewma_wind_len, nugget,
        fig_name, ls_comp_name, ls_comp_short_name, FLAGS, ls_comp_oper=None, ls_comp_oper_name=None,
        lbd_alarm_level=95, ubd_alarm_level=100, max_iter=30, tol=1e-6,
        cmap=CMAP, title_size=LAB_SIZE, plot_flag=True, save_proc_data=False, save_sep=False):
    # Spatial EWMA-T2 of scores.
    ls_arr_t2_scores_spatial_ewma_PI = [SpatialHotellingEWMAT2(score, n_hei_PI, n_wid_PI, ewma_sigma, ewma_wind_len, nugget, FLAGS.mu_train, FLAGS.Sinv_train, FLAGS) for score in ls_score_PI]
    ls_arr_t2_scores_spatial_ewma_PII = [SpatialHotellingEWMAT2(score, n_hei_PII, n_wid_PII, ewma_sigma, ewma_wind_len, nugget, FLAGS.mu_train, FLAGS.Sinv_train, FLAGS) for score in ls_score_PII]
    
    # Comp statistics
    
    # EWMA on Hotelling T2
    # 20200706, we don't need this anymore, because we have a better interpretation for deviance.
    # ls_arr_t2_scores_PI = [SpatialHotellingEWMAT2(score, n_hei_PI, n_wid_PI, 1, 0, nugget, FLAGS.mu_train, FLAGS.Sinv_train, FLAGS) for score in ls_score_PI]
    # ls_arr_t2_scores_PII = [SpatialHotellingEWMAT2(score, n_hei_PII, n_wid_PII, 1, 0, nugget, FLAGS.mu_train, FLAGS.Sinv_train, FLAGS) for score in ls_score_PII]
    # ls_ls_arr_comp_spatial_ewma_PI = [[ScoresSpatialEWMA(comp_PI.reshape((-1,1)), n_hei_PI, n_wid_PI, ewma_sigma, ewma_wind_len).squeeze(axis=-1) for comp_PI in ls_arr_t2_scores_PI]]
    # ls_ls_arr_comp_spatial_ewma_PII = [[ScoresSpatialEWMA(comp_PII.reshape((-1,1)), n_hei_PII, n_wid_PII, ewma_sigma, ewma_wind_len).squeeze(axis=-1) for comp_PII in ls_arr_t2_scores_PII]]
    
    # Other comparing metrics
    ls_ls_arr_comp_spatial_ewma_PI = [[ScoresSpatialEWMA(comp_PI[:,np.newaxis], n_hei_PI, n_wid_PI, ewma_sigma, ewma_wind_len).squeeze(axis=-1) for comp_PI in ls_comp_PI] for ls_comp_PI in ls_ls_comp_PI]
    ls_ls_arr_comp_spatial_ewma_PII = [[ScoresSpatialEWMA(comp_PII[:,np.newaxis], n_hei_PII, n_wid_PII, ewma_sigma, ewma_wind_len).squeeze(axis=-1) for comp_PII in ls_comp_PII] for ls_comp_PII in ls_ls_comp_PII]

    # The following are combining ht2_sewma and sewma_ht2. We no longer use this.
    # arr_arr_comb_spatial_ewma_PI, arr_arr_comb_spatial_ewma_PII = Cal_Multi_Chart(ls_arr_t2_scores_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PI[0], 
    #                                                                               ls_arr_t2_scores_spatial_ewma_PII, ls_ls_arr_comp_spatial_ewma_PII[0], 
    #                                                                               lbd_alarm_level=lbd_alarm_level, ubd_alarm_level=ubd_alarm_level, alarm_level=FLAGS.alarm_level, multi_chart_scale_flag=FLAGS.multi_chart_scale_flag)
    # ls_arr_comb_spatial_ewma_PI, ls_arr_comb_spatial_ewma_PII = list(arr_arr_comb_spatial_ewma_PI), list(arr_arr_comb_spatial_ewma_PII)

    # The following are combining ht2_sewma and deviance.
    arr_arr_comb_spatial_ewma_PI, arr_arr_comb_spatial_ewma_PII = Cal_Multi_Chart(ls_arr_t2_scores_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PI[-1], 
                                                                                  ls_arr_t2_scores_spatial_ewma_PII, ls_ls_arr_comp_spatial_ewma_PII[-1], 
                                                                                  lbd_alarm_level=lbd_alarm_level, ubd_alarm_level=ubd_alarm_level, alarm_level=FLAGS.alarm_level, multi_chart_scale_flag=FLAGS.multi_chart_scale_flag)
    ls_arr_comb_spatial_ewma_PI, ls_arr_comb_spatial_ewma_PII = list(arr_arr_comb_spatial_ewma_PI), list(arr_arr_comb_spatial_ewma_PII)

    # Combine all metrics together
    ls_ls_arr_comp_spatial_ewma_PI = [ls_arr_comb_spatial_ewma_PI, ls_arr_t2_scores_spatial_ewma_PI] + ls_ls_arr_comp_spatial_ewma_PI
    ls_ls_arr_comp_spatial_ewma_PII = [ls_arr_comb_spatial_ewma_PII, ls_arr_t2_scores_spatial_ewma_PII] + ls_ls_arr_comp_spatial_ewma_PII

    # Some operation on EWMA for comp statistics
    if ls_comp_oper is not None:
        ls_ls_arr_comp_oper_spatial_ewma_PI = [[comp_oper(comp_ewma_PI) for comp_ewma_PI in ls_comp_ewma_PI] for comp_oper, ls_comp_ewma_PI in zip(ls_comp_oper, ls_ls_arr_comp_spatial_ewma_PI) if comp_oper is not None]
        ls_ls_arr_comp_oper_spatial_ewma_PII = [[comp_oper(comp_ewma_PII) for comp_ewma_PII in ls_comp_ewma_PII] for comp_oper, ls_comp_ewma_PII in zip(ls_comp_oper, ls_ls_arr_comp_spatial_ewma_PII) if comp_oper is not None]
        ls_oper_comp_name = [comp_name for comp_name in ls_comp_oper_name if comp_name is not None]

        ls_ls_arr_comp_spatial_ewma_PI += ls_ls_arr_comp_oper_spatial_ewma_PI
        ls_ls_arr_comp_spatial_ewma_PII += ls_ls_arr_comp_oper_spatial_ewma_PII
        ls_comp_name += ls_oper_comp_name

        FLAGS.ls_comp_name, FLAGS.ls_comp_short_name = ls_comp_name, ls_comp_short_name

    if save_proc_data:
        pickle.dump(ls_arr_t2_scores_spatial_ewma_PI, open(os.path.join(FLAGS.training_res_folder, 'ls_arr_t2_scores_spatial_ewma_PI.h5'), 'wb'))
        pickle.dump(ls_arr_t2_scores_spatial_ewma_PII, open(os.path.join(FLAGS.training_res_folder, 'ls_arr_t2_scores_spatial_ewma_PII.h5'), 'wb'))
        pickle.dump(ls_ls_arr_comp_spatial_ewma_PI, open(os.path.join(FLAGS.training_res_folder, 'ls_ls_arr_comp_spatial_ewma_PI.h5'), 'wb'))
        pickle.dump(ls_ls_arr_comp_spatial_ewma_PII, open(os.path.join(FLAGS.training_res_folder, 'ls_ls_arr_comp_spatial_ewma_PII.h5'), 'wb'))
        pickle.dump(ls_img_arr_PI, open(os.path.join(FLAGS.training_res_folder, 'ls_img_arr_PI.h5'), 'wb'))
        pickle.dump(ls_img_arr_PII, open(os.path.join(FLAGS.training_res_folder, 'ls_img_arr_PII.h5'), 'wb'))
        pickle.dump(FLAGS, open(os.path.join(FLAGS.training_res_folder, 'visu_FLAGS.h5'), 'wb'))

    # ls_img_arr_plot_PI = [ls_img_arr_PI[idx] for idx in FLAGS.plot_img_PI_idx] if FLAGS.plot_img_PI_idx is not None else ls_img_arr_PI
    ls_img_arr_plot_PII = [ls_img_arr_PII[idx] for idx in FLAGS.plot_img_PII_idx] if FLAGS.plot_img_PII_idx is not None else ls_img_arr_PII
    ls_ls_arr_comp_spatial_ewma_plot_PI = [ls_ls_arr_comp_spatial_ewma_PI[idx] for idx in FLAGS.plot_metric_idx] if FLAGS.plot_metric_idx is not None else ls_ls_arr_comp_spatial_ewma_PI 
    print([len(ele) for ele in ls_ls_arr_comp_spatial_ewma_PII])
    ls_ls_arr_comp_spatial_ewma_plot_PII = [[ls_ls_arr_comp_spatial_ewma_PII[idx][img_idx] for img_idx in FLAGS.plot_img_PII_idx] for idx in FLAGS.plot_metric_idx] if FLAGS.plot_metric_idx is not None else ls_ls_arr_comp_spatial_ewma_PII
    print([len(ele) for ele in ls_ls_arr_comp_spatial_ewma_plot_PII])
    ls_comp_name_plot = [ls_comp_name[idx] for idx in FLAGS.plot_metric_idx] if FLAGS.plot_metric_idx is not None else ls_comp_name
    ls_comp_short_name_plot = [ls_comp_short_name[idx] for idx in FLAGS.plot_metric_idx] if FLAGS.plot_metric_idx is not None else ls_comp_short_name
    
    # 3D heatmap
    PlotSaveSpatial2D3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_plot_PII,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PI,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PII,
                                                ls_ls_arr_comp_spatial_ewma_plot_PI,
                                                ls_ls_arr_comp_spatial_ewma_plot_PII,
                                                ls_comp_name_plot,
                                                ls_comp_short_name_plot,
                                                FLAGS.training_res_folder,
                                                '3D_y_'+fig_name,
                                                FLAGS,
                                                alarm_level=FLAGS.alarm_level,
                                                cmap=cmap,
                                                flag_3d=True,
                                                title_size=title_size,
                                                save_sep=save_sep)

    # Heatmap overlapped with the original images
    PlotSaveSpatial2D3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_plot_PII,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PI,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PII,
                                                ls_ls_arr_comp_spatial_ewma_plot_PI,
                                                ls_ls_arr_comp_spatial_ewma_plot_PII,
                                                ls_comp_name_plot,
                                                ls_comp_short_name_plot,
                                                FLAGS.training_res_folder,
                                                'overlap_'+fig_name,
                                                FLAGS,
                                                cmap=cmap,
                                                overlap_flag=True,
                                                title_size=title_size,
                                                save_sep=save_sep)
    
    if plot_flag:
        # Heatmap
        PlotSaveSpatial2D3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_plot_PII,
                                                #  ls_arr_t2_scores_spatial_ewma_plot_PI,
                                                #  ls_arr_t2_scores_spatial_ewma_plot_PII,
                                                    ls_ls_arr_comp_spatial_ewma_plot_PI,
                                                    ls_ls_arr_comp_spatial_ewma_plot_PII,
                                                    ls_comp_name_plot,
                                                    ls_comp_short_name_plot,
                                                    FLAGS.training_res_folder,
                                                    fig_name,
                                                    FLAGS,
                                                    cmap=cmap,
                                                    title_size=title_size,
                                                    save_sep=save_sep)

        # Scaled heatmap
        PlotSaveSpatial2D3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_plot_PII,
                                                #  ls_arr_t2_scores_spatial_ewma_plot_PI,
                                                #  ls_arr_t2_scores_spatial_ewma_plot_PII,
                                                 ls_ls_arr_comp_spatial_ewma_plot_PI,
                                                 ls_ls_arr_comp_spatial_ewma_plot_PII,
                                                 ls_comp_name_plot,
                                                 ls_comp_short_name_plot,
                                                 FLAGS.training_res_folder,
                                                 'scaled_'+fig_name,
                                                 FLAGS,
                                                 cmap=cmap,
                                                 scale_flag=True,
                                                 title_size=title_size,
                                                 save_sep=save_sep)

        # Scaled heatmap overlapped with the original images
        PlotSaveSpatial2D3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_plot_PII,
                                                #  ls_arr_t2_scores_spatial_ewma_plot_PI,
                                                #  ls_arr_t2_scores_spatial_ewma_plot_PII,
                                                 ls_ls_arr_comp_spatial_ewma_plot_PI,
                                                 ls_ls_arr_comp_spatial_ewma_plot_PII,
                                                 ls_comp_name_plot,
                                                 ls_comp_short_name_plot,
                                                 FLAGS.training_res_folder,
                                                 'scaled_overlap_'+fig_name,
                                                 FLAGS,
                                                 cmap=cmap,
                                                 overlap_flag=True,
                                                 scale_flag=True,
                                                 title_size=title_size,
                                                 save_sep=save_sep)

        # # Heatmap
        # PlotSaveSpatialHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_PII,
        #                                          ls_arr_t2_scores_spatial_ewma_PI,
        #                                          ls_arr_t2_scores_spatial_ewma_PII,
        #                                          ls_ls_arr_comp_spatial_ewma_PI,
        #                                          ls_ls_arr_comp_spatial_ewma_PII,
        #                                          ls_comp_name,
        #                                          FLAGS.training_res_folder,
        #                                          fig_name,
        #                                          FLAGS,
        #                                          cmap=cmap,
        #                                          title_size=title_size,
        #                                          save_sep=save_sep)
        
        # # Heatmap overlapped with the original images
        # PlotSaveSpatialHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_PII,
        #                                          ls_arr_t2_scores_spatial_ewma_PI,
        #                                          ls_arr_t2_scores_spatial_ewma_PII,
        #                                          ls_ls_arr_comp_spatial_ewma_PI,
        #                                          ls_ls_arr_comp_spatial_ewma_PII,
        #                                          ls_comp_name,
        #                                          FLAGS.training_res_folder,
        #                                          'overlap_'+fig_name,
        #                                          FLAGS,
        #                                          cmap=cmap,
        #                                          overlap_flag=True,
        #                                          title_size=title_size,
        #                                          save_sep=save_sep)

        # # Scaled heatmap
        # PlotSaveSpatialHeatMap_Other_Score_Multi_Img_Scale_Prosp(ls_img_arr_PI, ls_img_arr_PII,
        #                                          ls_arr_t2_scores_spatial_ewma_PI,
        #                                          ls_arr_t2_scores_spatial_ewma_PII,
        #                                          ls_ls_arr_comp_spatial_ewma_PI,
        #                                          ls_ls_arr_comp_spatial_ewma_PII,
        #                                          ls_comp_name,
        #                                          FLAGS.training_res_folder,
        #                                          'scaled_'+fig_name,
        #                                          FLAGS,
        #                                          cmap=cmap,
        #                                          overlap_flag=False,
        #                                          title_size=title_size,
        #                                          save_sep=save_sep)

        # # Scaled heatmap overlapped with the original images
        # PlotSaveSpatialHeatMap_Other_Score_Multi_Img_Scale_Prosp(ls_img_arr_PI, ls_img_arr_PII,
        #                                          ls_arr_t2_scores_spatial_ewma_PI,
        #                                          ls_arr_t2_scores_spatial_ewma_PII,
        #                                          ls_ls_arr_comp_spatial_ewma_PI,
        #                                          ls_ls_arr_comp_spatial_ewma_PII,
        #                                          ls_comp_name,
        #                                          FLAGS.training_res_folder,
        #                                          'scaled_overlap_'+fig_name,
        #                                          FLAGS,
        #                                          cmap=cmap,
        #                                          overlap_flag=True,
        #                                          title_size=title_size,
        #                                          save_sep=save_sep)
        
        # # 3D heatmap
        # PlotSaveSpatial3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_PII,
        #                                          ls_arr_t2_scores_spatial_ewma_PI,
        #                                          ls_arr_t2_scores_spatial_ewma_PII,
        #                                          ls_ls_arr_comp_spatial_ewma_PI,
        #                                          ls_ls_arr_comp_spatial_ewma_PII,
        #                                          ls_comp_name,
        #                                          FLAGS.training_res_folder,
        #                                          '3D_y_'+fig_name,
        #                                          FLAGS,
        #                                          alarm_level=FLAGS.alarm_level,
        #                                          cmap=cmap,
        #                                          title_size=title_size,
        #                                          save_sep=save_sep)

    return (ls_arr_t2_scores_spatial_ewma_PI, ls_arr_t2_scores_spatial_ewma_PII,
            ls_ls_arr_comp_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PII)

def Plot_SpatialHotellingEWMAT2_Other_Score_Multi_Img_Prosp(
        fig_name,
        FLAGS,
        cmap=CMAP,
        title_size=LAB_SIZE,
        save_sep=False, show_fig=False):
    """ Calculate and plot scores and t2 of scores in 2D spatial image for prospective analysis for multiple images."""
    ls_arr_t2_scores_spatial_ewma_PI = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'ls_arr_t2_scores_spatial_ewma_PI.h5'), 'rb'))
    ls_arr_t2_scores_spatial_ewma_PII = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'ls_arr_t2_scores_spatial_ewma_PII.h5'), 'rb'))
    ls_ls_arr_comp_spatial_ewma_PI = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'ls_ls_arr_comp_spatial_ewma_PI.h5'), 'rb'))
    ls_ls_arr_comp_spatial_ewma_PII = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'ls_ls_arr_comp_spatial_ewma_PII.h5'), 'rb'))
    ls_img_arr_PI = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'ls_img_arr_PI.h5'), 'rb'))
    ls_img_arr_PII = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'ls_img_arr_PII.h5'), 'rb'))
    # FLAGS = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'visu_FLAGS.h5'), 'rb'))

    ls_comp_name, ls_comp_short_name = FLAGS.ls_comp_name, FLAGS.ls_comp_short_name

    # ls_img_arr_plot_PI = [ls_img_arr_PI[idx] for idx in FLAGS.plot_img_PI_idx] if FLAGS.plot_img_PI_idx is not None else ls_img_arr_PI
    ls_img_arr_plot_PII = [ls_img_arr_PII[idx] for idx in FLAGS.plot_img_PII_idx] if FLAGS.plot_img_PII_idx is not None else ls_img_arr_PII
    ls_ls_arr_comp_spatial_ewma_plot_PI = [ls_ls_arr_comp_spatial_ewma_PI[idx] for idx in FLAGS.plot_metric_idx] if FLAGS.plot_metric_idx is not None else ls_ls_arr_comp_spatial_ewma_PI
    logger.info("Num img for different metrics before filtering: {}".format([len(ele) for ele in ls_ls_arr_comp_spatial_ewma_PII]), extra=d)
    ls_ls_arr_comp_spatial_ewma_plot_PII = [[ls_ls_arr_comp_spatial_ewma_PII[idx][img_idx] for img_idx in FLAGS.plot_img_PII_idx] for idx in FLAGS.plot_metric_idx] if FLAGS.plot_metric_idx is not None else ls_ls_arr_comp_spatial_ewma_PII
    logger.info("Num img for different metrics after filtering: {}".format([len(ele) for ele in ls_ls_arr_comp_spatial_ewma_plot_PII]), extra=d)
    ls_comp_name_plot = [ls_comp_name[idx] for idx in FLAGS.plot_metric_idx] if FLAGS.plot_metric_idx is not None else ls_comp_name
    ls_comp_short_name_plot = [ls_comp_short_name[idx] for idx in FLAGS.plot_metric_idx] if FLAGS.plot_metric_idx is not None else ls_comp_short_name
    
    # Heatmap
    PlotSaveSpatial2D3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_plot_PII,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PI,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PII,
                                                ls_ls_arr_comp_spatial_ewma_plot_PI,
                                                ls_ls_arr_comp_spatial_ewma_plot_PII,
                                                ls_comp_name_plot,
                                                ls_comp_short_name_plot,
                                                FLAGS.training_res_folder,
                                                fig_name,
                                                FLAGS,
                                                cmap=cmap,
                                                title_size=title_size,
                                                save_sep=save_sep,
                                                show_fig=show_fig)
    
    # Heatmap overlapped with the original images
    PlotSaveSpatial2D3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_plot_PII,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PI,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PII,
                                                ls_ls_arr_comp_spatial_ewma_plot_PI,
                                                ls_ls_arr_comp_spatial_ewma_plot_PII,
                                                ls_comp_name_plot,
                                                ls_comp_short_name_plot,
                                                FLAGS.training_res_folder,
                                                'overlap_'+fig_name,
                                                FLAGS,
                                                cmap=cmap,
                                                overlap_flag=True,
                                                title_size=title_size,
                                                save_sep=save_sep,
                                                show_fig=show_fig)

    # Scaled heatmap
    PlotSaveSpatial2D3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_plot_PII,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PI,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PII,
                                                ls_ls_arr_comp_spatial_ewma_plot_PI,
                                                ls_ls_arr_comp_spatial_ewma_plot_PII,
                                                ls_comp_name_plot,
                                                ls_comp_short_name_plot,
                                                FLAGS.training_res_folder,
                                                'scaled_'+fig_name,
                                                FLAGS,
                                                cmap=cmap,
                                                scale_flag=True,
                                                title_size=title_size,
                                                save_sep=save_sep,
                                                show_fig=show_fig)

    # Scaled heatmap overlapped with the original images
    PlotSaveSpatial2D3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_plot_PII,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PI,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PII,
                                                ls_ls_arr_comp_spatial_ewma_plot_PI,
                                                ls_ls_arr_comp_spatial_ewma_plot_PII,
                                                ls_comp_name_plot,
                                                ls_comp_short_name_plot,
                                                FLAGS.training_res_folder,
                                                'scaled_overlap_'+fig_name,
                                                FLAGS,
                                                cmap=cmap,
                                                overlap_flag=True,
                                                scale_flag=True,
                                                title_size=title_size,
                                                save_sep=save_sep,
                                                show_fig=show_fig)
    
    # 3D heatmap
    PlotSaveSpatial2D3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_plot_PII,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PI,
                                            #  ls_arr_t2_scores_spatial_ewma_plot_PII,
                                                ls_ls_arr_comp_spatial_ewma_plot_PI,
                                                ls_ls_arr_comp_spatial_ewma_plot_PII,
                                                ls_comp_name_plot,
                                                ls_comp_short_name_plot,
                                                FLAGS.training_res_folder,
                                                '3D_y_'+fig_name,
                                                FLAGS,
                                                alarm_level=FLAGS.alarm_level,
                                                cmap=cmap,
                                                flag_3d=True,
                                                title_size=title_size,
                                                save_sep=save_sep,
                                                show_fig=show_fig)

    # # Heatmap
    # PlotSaveSpatialHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_PII,
    #                                             ls_arr_t2_scores_spatial_ewma_PI,
    #                                             ls_arr_t2_scores_spatial_ewma_PII,
    #                                             ls_ls_arr_comp_spatial_ewma_PI,
    #                                             ls_ls_arr_comp_spatial_ewma_PII,
    #                                             ls_comp_name,
    #                                             FLAGS.training_res_folder,
    #                                             fig_name,
    #                                             FLAGS,
    #                                             cmap=cmap,
    #                                             title_size=title_size,
    #                                             save_sep=save_sep)
    
    # # Heatmap overlapped with the original images
    # PlotSaveSpatialHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_PII,
    #                                             ls_arr_t2_scores_spatial_ewma_PI,
    #                                             ls_arr_t2_scores_spatial_ewma_PII,
    #                                             ls_ls_arr_comp_spatial_ewma_PI,
    #                                             ls_ls_arr_comp_spatial_ewma_PII,
    #                                             ls_comp_name,
    #                                             FLAGS.training_res_folder,
    #                                             'overlap_'+fig_name,
    #                                             FLAGS,
    #                                             cmap=cmap,
    #                                             overlap_flag=True,
    #                                             title_size=title_size,
    #                                             save_sep=save_sep)

    # # Scaled heatmap
    # PlotSaveSpatialHeatMap_Other_Score_Multi_Img_Scale_Prosp(ls_img_arr_PI, ls_img_arr_PII,
    #                                             ls_arr_t2_scores_spatial_ewma_PI,
    #                                             ls_arr_t2_scores_spatial_ewma_PII,
    #                                             ls_ls_arr_comp_spatial_ewma_PI,
    #                                             ls_ls_arr_comp_spatial_ewma_PII,
    #                                             ls_comp_name,
    #                                             FLAGS.training_res_folder,
    #                                             'scaled_'+fig_name,
    #                                             FLAGS,
    #                                             cmap=cmap,
    #                                             overlap_flag=False,
    #                                             title_size=title_size,
    #                                             save_sep=save_sep)

    # # Scaled heatmap overlapped with the original images
    # PlotSaveSpatialHeatMap_Other_Score_Multi_Img_Scale_Prosp(ls_img_arr_PI, ls_img_arr_PII,
    #                                             ls_arr_t2_scores_spatial_ewma_PI,
    #                                             ls_arr_t2_scores_spatial_ewma_PII,
    #                                             ls_ls_arr_comp_spatial_ewma_PI,
    #                                             ls_ls_arr_comp_spatial_ewma_PII,
    #                                             ls_comp_name,
    #                                             FLAGS.training_res_folder,
    #                                             'scaled_overlap_'+fig_name,
    #                                             FLAGS,
    #                                             cmap=cmap,
    #                                             overlap_flag=True,
    #                                             title_size=title_size,
    #                                             save_sep=save_sep)
    
    # # 3D heatmap
    # PlotSaveSpatial3DHeatMap_Other_Score_Multi_Img_Prosp(ls_img_arr_PI, ls_img_arr_PII,
    #                                             ls_arr_t2_scores_spatial_ewma_PI,
    #                                             ls_arr_t2_scores_spatial_ewma_PII,
    #                                             ls_ls_arr_comp_spatial_ewma_PI,
    #                                             ls_ls_arr_comp_spatial_ewma_PII,
    #                                             ls_comp_name,
    #                                             FLAGS.training_res_folder,
    #                                             '3D_y_'+fig_name,
    #                                             FLAGS,
    #                                             alarm_level=FLAGS.alarm_level,
    #                                             cmap=cmap,
    #                                             title_size=title_size,
    #                                             save_sep=save_sep)

    logger.info("The plotting has finished!", extra=d)

    return (ls_arr_t2_scores_spatial_ewma_PI, ls_arr_t2_scores_spatial_ewma_PII,
            ls_ls_arr_comp_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PII)

# Old way of plotting phase-II image in heatmap. It is not very appealling. Should not combine all phase-II together.
def PlotSaveSpatialHeatMap_Other_Score_Multi_Img_Prosp(
        ls_img_arr_PI, ls_img_arr_PII,
        ls_arr_t2_scores_spatial_ewma_PI,
        ls_arr_t2_scores_spatial_ewma_PII,
        ls_ls_arr_comp_spatial_ewma_PI,
        ls_ls_arr_comp_spatial_ewma_PII,
        ls_comp_name,
        folder_path,
        fig_name,
        FLAGS,
        pt_size=PT_SIZE,
        alpha=ALPHA,
        cmap=CMAP,
        overlap_flag=False,
        label_pad = LAB_PAD,
        title_size = LAB_SIZE,
        save_sep = False):
    """ Plot and save spatial heatmap for scores and other metrics."""
    num_comp_metric = len(ls_ls_arr_comp_spatial_ewma_PI)
    num_plt_row = 2+num_comp_metric
    num_img_PI, num_img_PII = len(ls_img_arr_PI), len(ls_img_arr_PII)
    print(num_img_PI, num_img_PII)
    fig = plt.figure(num=None, figsize=((num_img_PI+num_img_PII) * 2.2 * ONE_FIG_HEI + 4, num_plt_row * 2 * ONE_FIG_HEI), dpi=100, facecolor='w', edgecolor='k')
    # gs = fig.add_gridspec(num_plt_row, num_img_PI+num_img_PII)
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = WSPACE)
    # Phase-I and -II image
    ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (0,0), colspan=num_img_PI)
    # ax = fig.add_subplot(gs[0,:num_img_PI])
    ax.imshow(np.concatenate(ls_img_arr_PI, axis=1), cmap = GRAY_CMAP)
    ax.invert_yaxis()
    # ax.set_title('PI image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PI images', rot=-90)
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PI_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (0,num_img_PI), colspan=num_img_PII)
    # ax = fig.add_subplot(gs[0,num_img_PI:num_img_PI+num_img_PII])
    ax.imshow(np.concatenate(ls_img_arr_PII, axis=1), cmap = GRAY_CMAP)
    ax.invert_yaxis()
    # ax.set_title('PII image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PII images', rot=-90)
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PII_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    
    def plot_multi_heatmap(ls_img_arr, ls_metric_arr, cmap_vmin, cmap_vmax, overlap_flag, title, fig_name, plt_row, col):
        num_img = len(ls_img_arr)
        hei, wid = ls_img_arr[0].shape
        n_hei, n_wid = ls_metric_arr[0].shape
        if col == 'PI':
            ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (plt_row,0), colspan=num_img_PI)
        else:
            ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (plt_row,num_img_PI), colspan=num_img_PII)
        # title = 'Hotelling $T^2$ EWMA Score PII'
        if overlap_flag:
            ax.imshow(np.concatenate(ls_img_arr, axis=1), cmap=GRAY_CMAP)
            ls_upperleft_corner = [(idx*wid+(wid-n_wid)//2, (hei-n_hei)//2) for idx in range(num_img)]
            temp_alp = alpha
        else:
            ls_upperleft_corner = [(idx*n_wid, 0) for idx in range(num_img)]
            ax.invert_yaxis()
            temp_alp = 1
        
        for idx, (upperleft_corner, arr) in enumerate(zip(ls_upperleft_corner, ls_metric_arr)):
            if idx==num_img-1:
                PlotSpatialHeatMap(ax, arr, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                                title=title, fig_name=fig_name, title_size=title_size, rot=0, upperleft_corner=upperleft_corner, alpha=temp_alp, save_sep=save_sep)
            else:
                PlotSpatialHeatMap(ax, arr, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                                title=None, fig_name=None, title_size=title_size, rot=0, upperleft_corner=upperleft_corner, config_prop=False, alpha=temp_alp, save_sep=False)
        ax.invert_yaxis()

    # Hotelling T2 scores
    cmap_vmin = min(np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI]), np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    cmap_vmax = max(np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI]), np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    title = 'Hotelling $T^2$ EWMA Score PI'
    plot_multi_heatmap(ls_img_arr_PI, ls_arr_t2_scores_spatial_ewma_PI, cmap_vmin, cmap_vmax, overlap_flag, title, fig_name=fig_name.split('.')[0]+'_PI_score.png', plt_row=1, col='PI')
    title = 'Hotelling $T^2$ EWMA Score PII'
    plot_multi_heatmap(ls_img_arr_PII, ls_arr_t2_scores_spatial_ewma_PII, cmap_vmin, cmap_vmax, overlap_flag, title, fig_name=fig_name.split('.')[0]+'_PII_score.png', plt_row=1, col='PII')

    # Comparing metric
    for idx_comp, (comp_name, ls_arr_comp_spatial_ewma_PI, ls_arr_comp_spatial_ewma_PII) in enumerate(zip(ls_comp_name, ls_ls_arr_comp_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PII)): 
        cmap_vmin = min(np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PI]), np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        cmap_vmax = max(np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PI]), np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        title = ' '.join([comp_name, 'PI'])
        plot_multi_heatmap(ls_img_arr_PI, ls_arr_comp_spatial_ewma_PI, cmap_vmin, cmap_vmax, overlap_flag, title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', plt_row=2+idx_comp, col='PI')
        title = ' '.join([comp_name, 'PII'])
        plot_multi_heatmap(ls_img_arr_PII, ls_arr_comp_spatial_ewma_PII, cmap_vmin, cmap_vmax, overlap_flag, title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', plt_row=2+idx_comp, col='PII')

    # plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.savefig(os.path.join(folder_path, fig_name))
    plt.close()

def PlotSaveSpatialHeatMap_Other_Score_Multi_Img_Scale_Prosp(
        ls_img_arr_PI, ls_img_arr_PII,
        ls_arr_t2_scores_spatial_ewma_PI,
        ls_arr_t2_scores_spatial_ewma_PII,
        ls_ls_arr_comp_spatial_ewma_PI,
        ls_ls_arr_comp_spatial_ewma_PII,
        ls_comp_name,
        folder_path,
        fig_name,
        FLAGS,
        pt_size=PT_SIZE,
        alpha=ALPHA,
        cmap=CMAP,
        overlap_flag=False,
        label_pad = LAB_PAD,
        title_size = LAB_SIZE,
        save_sep = False):
    """ Plot and save spatial heatmap for scores and other metrics."""
    num_comp_metric = len(ls_ls_arr_comp_spatial_ewma_PI)
    num_plt_row = 2+num_comp_metric
    num_img_PI, num_img_PII = len(ls_img_arr_PI), len(ls_img_arr_PII)
    fig = plt.figure(num=None, figsize=((num_img_PI+num_img_PII) * 2.2 * ONE_FIG_HEI + 4, num_plt_row * 2 * ONE_FIG_HEI), dpi=100, facecolor='w', edgecolor='k')
    # gs = fig.add_gridspec(num_plt_row, num_img_PI+num_img_PII)
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = WSPACE)
    # Phase-I and -II image
    ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (0,0), colspan=num_img_PI)
    # ax = plt.subplot2grid((num_plt_row,2), (0,0))
    # ax = fig.add_subplot(gs[0,:num_img_PI])
    ax.imshow(np.concatenate(ls_img_arr_PI, axis=1), cmap = GRAY_CMAP)
    ax.invert_yaxis()
    # ax.set_title('PI image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PI images')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PI_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (0,num_img_PI), colspan=num_img_PII)
    ax.imshow(np.concatenate(ls_img_arr_PII, axis=1), cmap = GRAY_CMAP)
    ax.invert_yaxis()
    # ax.set_title('PII image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PII images')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PII_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    
    def plot_multi_heatmap(ls_img_arr, ls_metric_arr, cmap_vmin, cmap_vmax, overlap_flag, title, fig_name, plt_row, col):
        num_img = len(ls_img_arr)
        hei, wid = ls_img_arr[0].shape
        n_hei, n_wid = ls_metric_arr[0].shape
        # ax = plt.subplot2grid((num_plt_row,2), (plt_row,plt_col))
        if col == 'PI':
            ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (plt_row,0), colspan=num_img_PI)
        else:
            ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (plt_row,num_img_PI), colspan=num_img_PII)
        # title = 'Hotelling $T^2$ EWMA Score PII'
        if overlap_flag:
            ax.imshow(np.concatenate(ls_img_arr, axis=1), cmap=GRAY_CMAP)
            ls_upperleft_corner = [(idx*wid+(wid-n_wid)//2, (hei-n_hei)//2) for idx in range(num_img)]
            temp_alp = alpha
        else:
            ls_upperleft_corner = [(idx*n_wid, 0) for idx in range(num_img)]
            ax.invert_yaxis()
            temp_alp = 1
        
        for idx, (upperleft_corner, arr) in enumerate(zip(ls_upperleft_corner, ls_metric_arr)):
            if idx==num_img-1-1:
                PlotSpatialHeatMap(ax, arr, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                                title=title, fig_name=fig_name, title_size=title_size, rot=0, upperleft_corner=upperleft_corner, alpha=temp_alp, save_sep=save_sep)
            else:
                PlotSpatialHeatMap(ax, arr, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                                title=None, fig_name=None, title_size=title_size, rot=0, upperleft_corner=upperleft_corner, config_prop=False, alpha=temp_alp, save_sep=False)
        ax.invert_yaxis()

    # Hotelling T2 scores
    title = 'Hotelling $T^2$ EWMA Score PI'
    plot_multi_heatmap(ls_img_arr_PI, ls_arr_t2_scores_spatial_ewma_PI, None, None, overlap_flag, title, fig_name=fig_name.split('.')[0]+'_PI_score.png', plt_row=1, col='PI')
    
    PI_min = np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI])
    PI_max = np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI])
    ls_arr_t2_scores_spatial_ewma_PII = [(arr-PI_min)/(PI_max-PI_min) for arr in ls_arr_t2_scores_spatial_ewma_PII]
    # cmap_vmin = min(np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI]), np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    # cmap_vmax = max(np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI]), np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    cmap_vmin = np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII])
    cmap_vmax = np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII])
    title = 'Hotelling $T^2$ EWMA Score PII'
    plot_multi_heatmap(ls_img_arr_PII, ls_arr_t2_scores_spatial_ewma_PII, cmap_vmin, cmap_vmax, overlap_flag, title, fig_name=fig_name.split('.')[0]+'_PII_score.png', plt_row=1, col='PII')

    # Comparing metric
    for idx_comp, (comp_name, ls_arr_comp_spatial_ewma_PI, ls_arr_comp_spatial_ewma_PII) in enumerate(zip(ls_comp_name, ls_ls_arr_comp_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PII)): 
        title = ' '.join([comp_name, 'PI'])
        plot_multi_heatmap(ls_img_arr_PI, ls_arr_comp_spatial_ewma_PI, None, None, overlap_flag, title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', plt_row=2+idx_comp, col='PI')
        
        PI_min = np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII])
        PI_max = np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII])
        ls_arr_comp_spatial_ewma_PII = [(arr-PI_min)/(PI_max-PI_min) for arr in ls_arr_comp_spatial_ewma_PII]
        cmap_vmin = np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII])
        cmap_vmax = np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII])
        title = ' '.join([comp_name, 'PII'])
        plot_multi_heatmap(ls_img_arr_PII, ls_arr_comp_spatial_ewma_PII, cmap_vmin, cmap_vmax, overlap_flag, title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', plt_row=2+idx_comp, col='PII')

    # plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.savefig(os.path.join(folder_path, fig_name))
    plt.close()


def PlotSaveSpatial3DHeatMap_Other_Score_Multi_Img_Prosp(
        ls_img_arr_PI, ls_img_arr_PII,
        ls_arr_t2_scores_spatial_ewma_PI,
        ls_arr_t2_scores_spatial_ewma_PII,
        ls_ls_arr_comp_spatial_ewma_PI,
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
        thre_kwargs = {'linewidth': 2, 'edgecolor': 'k', 'linestyle': '-', 'alpha': 0.3},
        save_sep = False):
    """ Plot and save spatial heatmap for scores and other metrics."""
    num_comp_metric = len(ls_ls_arr_comp_spatial_ewma_PI)
    num_plt_row = 2+num_comp_metric
    num_img_PI, num_img_PII = len(ls_img_arr_PI), len(ls_img_arr_PII)
    fig = plt.figure(num=None, figsize=((num_img_PI+num_img_PII) * 2.2 * ONE_FIG_HEI + 4, num_plt_row * 2 * ONE_FIG_HEI), dpi=100, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = WSPACE)
    # Phase-I and -II image
    for idx, img_arr_PI in enumerate(ls_img_arr_PI):
        ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (0,idx))
        ax.imshow(img_arr_PI, cmap = GRAY_CMAP)
        ax.invert_yaxis()
        # ax.set_title('PI image', size=title_size)
        Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PI image {}'.format(idx), rot=-90)
        if save_sep:
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PI_{}_orig.png'.format(idx)), 
                        bbox_inches=extent.expanded(1.4, 1.4))
    for idx, img_arr_PII in enumerate(ls_img_arr_PII):
        ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (0,num_img_PI+idx))
        ax.imshow(img_arr_PII, cmap = GRAY_CMAP)
        ax.invert_yaxis()
        # ax.set_title('PII image', size=title_size)
        Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PII image {}'.format(idx), rot=-90)
        if save_sep:
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PII_{}_orig.png'.format(idx)), 
                        bbox_inches=extent.expanded(1.4, 1.4))
    
    def plot_3D_heatmap(metric_arr, cmap_vmin, cmap_vmax, title, fig_name, arr_thres, plt_row, plt_col):
        # ax = plt.subplot2grid((num_plt_row,2), (plt_row,plt_col), projection='3d')
        ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (plt_row,plt_col), projection='3d')
        # ls_upperleft_corner = [(idx*n_wid, 0) for idx in range(num_img)]
        PlotSpatial3DHeatMap(ax, metric_arr, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                    title=title, fig_name=fig_name, title_size=title_size, thre_flag=thre_flag, 
                    thre_kwargs=thre_kwargs, arr_thres=arr_thres, z_lim=(cmap_vmin, cmap_vmax), axis_tick_flag=[True, False, True], save_sep=save_sep)
        ax.invert_yaxis()

    def plot_multi_3D_heatmap(ls_metric_arr, cmap_vmin, cmap_vmax, title, fig_name, arr_thres, plt_row, col):
        n_hei, n_wid = ls_metric_arr[0].shape
        for idx, metric_arr in enumerate(ls_metric_arr):
            if col=='PI':
                plot_3D_heatmap(metric_arr, cmap_vmin, cmap_vmax, title+' {}'.format(idx), fig_name=fig_name.split('.')[0]+'_{}.png'.format(idx), arr_thres=arr_thres, plt_row=plt_row, plt_col=idx)
            else: # 'PII'
                plot_3D_heatmap(metric_arr, cmap_vmin, cmap_vmax, title+' {}'.format(idx), fig_name=fig_name.split('.')[0]+'_{}.png'.format(idx), arr_thres=arr_thres, plt_row=plt_row, plt_col=num_img_PI+idx)

    # Hotelling T2 scores
    ucl = np.percentile(np.array(ls_arr_t2_scores_spatial_ewma_PI), (100+alarm_level)/2)
    lcl = np.percentile(np.array(ls_arr_t2_scores_spatial_ewma_PI), (100-alarm_level)/2)
    arr_thres = np.array([lcl, ucl])

    cmap_vmin = min(np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI]), np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    cmap_vmax = max(np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI]), np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    title = 'Hotelling $T^2$ SEWMA Score PI'
    plot_multi_3D_heatmap(ls_arr_t2_scores_spatial_ewma_PI, cmap_vmin, cmap_vmax, title, fig_name=fig_name.split('.')[0]+'_PI_score.png', arr_thres=arr_thres, plt_row=1, col='PI')
    title = 'Hotelling $T^2$ SEWMA Score PII'
    plot_multi_3D_heatmap(ls_arr_t2_scores_spatial_ewma_PII, cmap_vmin, cmap_vmax, title, fig_name=fig_name.split('.')[0]+'_PII_score.png', arr_thres=arr_thres, plt_row=1, col='PII')

    # Comparing metric
    for idx_comp, (comp_name, ls_arr_comp_spatial_ewma_PI, ls_arr_comp_spatial_ewma_PII) in enumerate(zip(ls_comp_name, ls_ls_arr_comp_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PII)):
        ucl = np.percentile(np.array(ls_arr_comp_spatial_ewma_PI), (100+alarm_level)/2)
        lcl = np.percentile(np.array(ls_arr_comp_spatial_ewma_PI), (100-alarm_level)/2)
        arr_thres = np.array([lcl, ucl])        
        cmap_vmin = min(np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PI]), np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        cmap_vmax = max(np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PI]), np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        title = ' '.join([comp_name, 'PI'])
        plot_multi_3D_heatmap(ls_arr_comp_spatial_ewma_PI, cmap_vmin, cmap_vmax, title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', arr_thres=arr_thres, plt_row=2+idx_comp, col='PI')
        title = ' '.join([comp_name, 'PII'])
        plot_multi_3D_heatmap(ls_arr_comp_spatial_ewma_PII, cmap_vmin, cmap_vmax, title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', arr_thres=arr_thres, plt_row=2+idx_comp, col='PII')

    # plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.savefig(os.path.join(folder_path, fig_name))
    plt.close()

def PlotSaveSpatial2D3DHeatMap_Other_Score_Multi_Img_Prosp(
        ls_img_arr_PI, ls_img_arr_PII,
        # ls_arr_t2_scores_spatial_ewma_PI,
        # ls_arr_t2_scores_spatial_ewma_PII,
        ls_ls_arr_comp_spatial_ewma_PI,
        ls_ls_arr_comp_spatial_ewma_PII,
        ls_comp_name,
        ls_comp_short_name,
        folder_path,
        fig_name,
        FLAGS,
        pt_size=PT_SIZE,
        alpha=ALPHA,
        cmap=CMAP,
        flag_3d=False,
        overlap_flag=False,
        scale_flag=False,
        label_pad = LAB_PAD,
        title_size = LAB_SIZE,
        thre_flag = True,
        alarm_level = 99, # percentail
        thre_kwargs = {'linewidth': 2, 'edgecolor': 'k', 'linestyle': '-', 'alpha': 0.3},
        # ls_one_cl_idx = [1], # default r'SEWMA-$\bm{\theta}$'
        save_sep = False,
        show_fig = False):
    """ Plot and save spatial heatmap for scores and other metrics."""
    logger.info("Start plotting.", extra=d)
    num_comp_metric = len(ls_ls_arr_comp_spatial_ewma_PII)
    num_plt_row = 1+num_comp_metric
    num_img_PI, num_img_PII = len(FLAGS.plot_img_PI_idx), len(ls_img_arr_PII)
    fig = plt.figure(num=None, figsize=((num_img_PI+num_img_PII) * 2.2 * ONE_FIG_HEI + 4, num_plt_row * 2 * ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = WSPACE)
    # Phase-I and -II image
    for idx, img_arr_PI in enumerate([ls_img_arr_PI[img_idx] for img_idx in FLAGS.plot_img_PI_idx]):
        ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (0,idx))
        ax.imshow(img_arr_PI, cmap = GRAY_CMAP)
        ax.invert_yaxis()
        # ax.set_title('PI image', size=title_size)
        # Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PI image {}'.format(idx))
        Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='CL-Selection image {}'.format(idx))
        if save_sep:
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PI_{}_orig.png'.format(idx)), 
                        bbox_inches=extent.expanded(1.4, 1.4))
    for idx, img_arr_PII in enumerate(ls_img_arr_PII):
        ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (0,num_img_PI+idx))
        ax.imshow(img_arr_PII, cmap = GRAY_CMAP)
        ax.invert_yaxis()
        # ax.set_title('PII image', size=title_size)
        # Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PII image {}'.format(idx))
        Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='Monitoring image {}'.format(idx))
        if save_sep:
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PII_{}_orig.png'.format(idx)), 
                        bbox_inches=extent.expanded(1.4, 1.4))
    
    def plot_3D_heatmap(metric_arr, cmap_vmin, cmap_vmax, title, fig_name, arr_thres, scatter_plot, plt_row, plt_col):
        # ax = plt.subplot2grid((num_plt_row,2), (plt_row,plt_col), projection='3d')
        print(num_plt_row, num_img_PI, num_img_PII, plt_row, plt_col, FLAGS.plot_img_PI_idx, FLAGS.plot_img_PII_idx, FLAGS.plot_metric_idx)
        ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (plt_row,plt_col), projection='3d')
        # ls_upperleft_corner = [(idx*n_wid, 0) for idx in range(num_img)]
        PlotSpatial3DHeatMap(ax, metric_arr, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                    title=title, fig_name=fig_name, title_size=title_size, thre_flag=thre_flag, 
                    thre_kwargs=thre_kwargs, arr_thres=arr_thres, z_lim=(cmap_vmin, cmap_vmax), 
                    axis_tick_flag=[True, False, True], scatter_plot=scatter_plot, save_sep=save_sep)
        ax.invert_yaxis()

    def plot_multi_3D_heatmap(ls_metric_arr, cmap_vmin, cmap_vmax, title, fig_name, arr_thres, scatter_plot, plt_row, col):
        # n_hei, n_wid = ls_metric_arr[0].shape
        for idx, metric_arr in enumerate(ls_metric_arr):
            print(len(ls_metric_arr), idx)
            if col=='PI':
                plot_3D_heatmap(metric_arr, cmap_vmin, cmap_vmax, title+' {}'.format(idx), fig_name=fig_name.split('.')[0]+'_{}.png'.format(idx), arr_thres=arr_thres, scatter_plot=scatter_plot, plt_row=plt_row, plt_col=idx)
            else: # 'PII'
                plot_3D_heatmap(metric_arr, cmap_vmin, cmap_vmax, title+' {}'.format(idx), fig_name=fig_name.split('.')[0]+'_{}.png'.format(idx), arr_thres=arr_thres, scatter_plot=scatter_plot, plt_row=plt_row, plt_col=num_img_PI+idx)

    def plot_2D_heatmap(img_arr, metric_arr, cmap_vmin, cmap_vmax, title, fig_name, alpha, overlap_flag, plt_row, plt_col):
        hei, wid = img_arr.shape
        n_hei, n_wid = metric_arr.shape
        # ax = plt.subplot2grid((num_plt_row,2), (plt_row,plt_col), projection='3d')
        ax = plt.subplot2grid((num_plt_row,num_img_PI+num_img_PII), (plt_row,plt_col))
        # ls_upperleft_corner = [(idx*n_wid, 0) for idx in range(num_img)]
        if overlap_flag:
            ax.imshow(img_arr, cmap=GRAY_CMAP)
            upperleft_corner = ((wid-n_wid)//2, (hei-n_hei)//2)
            tmp_alp = alpha
        else:
            upperleft_corner = (0, 0)
            tmp_alp = 1

        PlotSpatialHeatMap(ax, metric_arr, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                        title=title, fig_name=fig_name, title_size=title_size, rot=0, 
                        upperleft_corner=upperleft_corner, alpha=tmp_alp, save_sep=save_sep)
        ax.invert_yaxis() 

    def plot_multi_2D_heatmap(ls_img_arr, ls_metric_arr, cmap_vmin, cmap_vmax, overlap_flag, title, fig_name, plt_row, col):
        # num_img = len(ls_img_arr)
        # hei, wid = ls_img_arr[0].shape
        # n_hei, n_wid = ls_metric_arr[0].shape
        for idx, (img_arr, metric_arr) in enumerate(zip(ls_img_arr, ls_metric_arr)):
            if col=='PI':
                plot_2D_heatmap(img_arr, metric_arr, cmap_vmin, cmap_vmax, title+' {}'.format(idx), fig_name=fig_name.split('.')[0]+'_{}.png'.format(idx), alpha=alpha, overlap_flag=overlap_flag, plt_row=plt_row, plt_col=idx)
            else: # 'PII'
                plot_2D_heatmap(img_arr, metric_arr, cmap_vmin, cmap_vmax, title+' {}'.format(idx), fig_name=fig_name.split('.')[0]+'_{}.png'.format(idx), alpha=alpha, overlap_flag=overlap_flag, plt_row=plt_row, plt_col=num_img_PI+idx)

    # # Hotelling T2 scores
    # ucl = np.percentile(np.array(ls_arr_t2_scores_spatial_ewma_PI), (100+alarm_level)/2)
    # lcl = np.percentile(np.array(ls_arr_t2_scores_spatial_ewma_PI), (100-alarm_level)/2)
    # arr_thres = np.array([lcl, ucl])

    # if scale_flag:
    #     PI_min = np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI])
    #     PI_max = np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI])
    #     ls_arr_t2_scores_spatial_ewma_PII = [(arr-PI_min)/(PI_max-PI_min) for arr in ls_arr_t2_scores_spatial_ewma_PII]
    #     # cmap_vmin = min(np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI]), np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    #     # cmap_vmax = max(np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI]), np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    #     cmap_vmin = np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII])
    #     cmap_vmax = np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII])
    # else:
    #     cmap_vmin = min(np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI]), np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    #     cmap_vmax = max(np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PI]), np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    
    # title = 'Hotelling $T^2$ SEWMA Score PI'
    # if flag_3d:
    #     plot_multi_3D_heatmap(ls_arr_t2_scores_spatial_ewma_PI, cmap_vmin, cmap_vmax, title, fig_name=fig_name.split('.')[0]+'_PI_score.png', arr_thres=arr_thres, plt_row=1, col='PI')
    # else:
    #     plot_multi_2D_heatmap(ls_img_arr_PI, ls_arr_t2_scores_spatial_ewma_PI, None if scale_flag else cmap_vmin, None if scale_flag else cmap_vmax, overlap_flag, title, fig_name=fig_name.split('.')[0]+'_PI_score.png', plt_row=1, col='PI')
    # title = 'Hotelling $T^2$ SEWMA Score PII'
    # if flag_3d:
    #     plot_multi_3D_heatmap(ls_arr_t2_scores_spatial_ewma_PII, cmap_vmin, cmap_vmax, title, fig_name=fig_name.split('.')[0]+'_PII_score.png', arr_thres=arr_thres, plt_row=1, col='PII')
    # else:
    #     plot_multi_2D_heatmap(ls_img_arr_PII, ls_arr_t2_scores_spatial_ewma_PII, cmap_vmin, cmap_vmax, overlap_flag, title, fig_name=fig_name.split('.')[0]+'_PII_score.png', plt_row=1, col='PII')

    # Comparing metric
    for idx_comp, (comp_name, comp_short_name, ls_arr_comp_spatial_ewma_PI, ls_arr_comp_spatial_ewma_PII) in enumerate(zip(ls_comp_name, ls_comp_short_name, ls_ls_arr_comp_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PII)):
        # if comp_name == 'REWMA' or FLAGS.multi_chart_scale_flag == 'mid':
        if comp_short_name!='ht2_ewma_score' and FLAGS.multi_chart_scale_flag == 'mid':
            ucl = np.percentile(np.array(ls_arr_comp_spatial_ewma_PI), (100+alarm_level)/2)
            lcl = np.percentile(np.array(ls_arr_comp_spatial_ewma_PI), (100-alarm_level)/2)
            arr_thres = np.array([ucl, lcl])
        else:
            ucl = np.percentile(np.array(ls_arr_comp_spatial_ewma_PI), alarm_level)
            arr_thres = np.array([ucl])
        
        if scale_flag:
            PI_min = np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII])
            PI_max = np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII])
            ls_arr_comp_spatial_ewma_PII = [(arr-PI_min)/(PI_max-PI_min) for arr in ls_arr_comp_spatial_ewma_PII]
            cmap_vmin = np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII])
            cmap_vmax = np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII])
        else:
            cmap_vmin = min(np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PI]), np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
            cmap_vmax = max(np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PI]), np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        
        # title = ' '.join([comp_name, 'PI'])
        # title = ' '.join([comp_name, 'CL-Selection'])
        title = comp_name
        if flag_3d:
            # plot_multi_3D_heatmap(ls_arr_comp_spatial_ewma_PI, cmap_vmin, cmap_vmax, title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', 
            #                       arr_thres=arr_thres if idx_comp>0 else None, scatter_plot=False if idx_comp>0 else True, plt_row=1+idx_comp, col='PI')
            plot_multi_3D_heatmap([ls_arr_comp_spatial_ewma_PI[img_idx] for img_idx in FLAGS.plot_img_PI_idx], cmap_vmin, cmap_vmax, title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', 
                                  arr_thres=arr_thres, scatter_plot=False, plt_row=1+idx_comp, col='PI')
        else:
            plot_multi_2D_heatmap([ls_img_arr_PI[img_idx] for img_idx in FLAGS.plot_img_PI_idx], [ls_arr_comp_spatial_ewma_PI[img_idx] for img_idx in FLAGS.plot_img_PI_idx], None if scale_flag else cmap_vmin, None if scale_flag else cmap_vmax, overlap_flag, title, 
                                  fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', plt_row=1+idx_comp, col='PI')
        # title = ' '.join([comp_name, 'PII'])
        # title = ' '.join([comp_name, 'Monitoring'])
        title = comp_name
        if flag_3d:
            # plot_multi_3D_heatmap(ls_arr_comp_spatial_ewma_PII, cmap_vmin, cmap_vmax, title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', 
            #                       arr_thres=arr_thres if idx_comp>0 else None, scatter_plot=False if idx_comp>0 else True, plt_row=1+idx_comp, col='PII')
            plot_multi_3D_heatmap(ls_arr_comp_spatial_ewma_PII, cmap_vmin, cmap_vmax, title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', 
                                  arr_thres=arr_thres, scatter_plot=False, plt_row=1+idx_comp, col='PII')
        else:
            plot_multi_2D_heatmap(ls_img_arr_PII, ls_arr_comp_spatial_ewma_PII, cmap_vmin, cmap_vmax, overlap_flag, title, 
                                  fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', plt_row=1+idx_comp, col='PII')

    # plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    if not show_fig:
        plt.savefig(os.path.join(folder_path, fig_name))
        plt.close()
    else:
        plt.show()

def SpatialHotellingEWMAT2_Other_Score_Prosp(
        img_arr_PI, 
        ls_img_arr_PII,
        score_PI,
        comb_score_PII,
        ls_comp_PI,
        ls_comb_comp_PII,
        n_hei_PI, n_wid_PI,
        n_hei_PII, n_wid_PII,
        ewma_sigma,
        ewma_wind_len,
        nugget,
        fig_name,
        ls_comp_name,
        FLAGS,
        mu_train, Sinv_train,
        cmap=CMAP,
        title_size=LAB_SIZE,
        save_proc_data=False,
        save_sep=False):
    """ Calculate and plot scores and t2 of scores in 2D spatial image for prospective analysis."""
    # Spatial EWMA-T2 of scores.
    arr_t2_scores_spatial_ewma_PI = SpatialHotellingEWMAT2(score_PI, n_hei_PI, n_wid_PI, ewma_sigma, ewma_wind_len, nugget, mu_train, Sinv_train, FLAGS)
    num_rows_one_PII = comb_score_PII.shape[0]//FLAGS.num_PII
    ls_arr_t2_scores_spatial_ewma_PII = [SpatialHotellingEWMAT2(comb_score_PII[idx*num_rows_one_PII:(idx+1)*num_rows_one_PII], n_hei_PII, n_wid_PII, ewma_sigma, ewma_wind_len, nugget, mu_train, Sinv_train, FLAGS) for idx in range(FLAGS.num_PII)]
    # Comp statistics
    ls_arr_comp_spatial_ewma_PI = [ScoresSpatialEWMA(comp_PI[:,np.newaxis], n_hei_PI, n_wid_PI, ewma_sigma, ewma_wind_len).squeeze(axis=-1) for comp_PI in ls_comp_PI]
    ls_ls_arr_comp_spatial_ewma_PII = [[ScoresSpatialEWMA(comb_comp_PII[idx*num_rows_one_PII:(idx+1)*num_rows_one_PII,np.newaxis], n_hei_PII, n_wid_PII, ewma_sigma, ewma_wind_len).squeeze(axis=-1) for idx in range(FLAGS.num_PII)] for comb_comp_PII in ls_comb_comp_PII]

    if save_proc_data:
        pickle.dump(arr_t2_scores_spatial_ewma_PI, open(os.path.join(FLAGS.training_res_folder, 't2_scores_spatial_ewma_PI.h5'), 'wb'))
        pickle.dump(ls_arr_t2_scores_spatial_ewma_PII, open(os.path.join(FLAGS.training_res_folder, 'ls_t2_scores_spatial_ewma_PII.h5'), 'wb'))
        pickle.dump(ls_arr_comp_spatial_ewma_PI, open(os.path.join(FLAGS.training_res_folder, 'ls_metrics_spatial_ewma_PI.h5'), 'wb'))
        pickle.dump(ls_ls_arr_comp_spatial_ewma_PII, open(os.path.join(FLAGS.training_res_folder, 'ls_ls_metrics_spatial_ewma_PII.h5'), 'wb'))
        pickle.dump(img_arr_PI, open(os.path.join(FLAGS.training_res_folder, 'img_arr_PI.h5'), 'wb'))
        pickle.dump(ls_img_arr_PII, open(os.path.join(FLAGS.training_res_folder, 'ls_img_arr_PII.h5'), 'wb'))
        pickle.dump(FLAGS, open(os.path.join(FLAGS.training_res_folder, 'visu_FLAGS.h5'), 'wb'))

    # Heatmap
    PlotSaveSpatialHeatMap_Other_Score_Prosp(img_arr_PI, ls_img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             ls_arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             title_size=title_size,
                                             save_sep=save_sep)
    
    # Heatmap overlapped with the original images
    PlotSaveSpatialHeatMap_Other_Score_Prosp(img_arr_PI, ls_img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             ls_arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             'overlap_'+fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             overlap_flag=True,
                                             title_size=title_size,
                                             save_sep=save_sep)
    
    # Scaled heatmap
    PlotSaveSpatialHeatMap_Other_Score_Scale_Prosp(img_arr_PI, ls_img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             ls_arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             'scaled_'+fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             overlap_flag=False,
                                             title_size=title_size,
                                             save_sep=save_sep)

    # Scaled heatmap overlapped with the original images
    PlotSaveSpatialHeatMap_Other_Score_Scale_Prosp(img_arr_PI, ls_img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             ls_arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             'scaled_overlap_'+fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             overlap_flag=True,
                                             title_size=title_size,
                                             save_sep=save_sep)
    
    # 3D heatmap
    PlotSaveSpatial3DHeatMap_Other_Score_Prosp(img_arr_PI, ls_img_arr_PII,
                                               arr_t2_scores_spatial_ewma_PI,
                                               ls_arr_t2_scores_spatial_ewma_PII,
                                               ls_arr_comp_spatial_ewma_PI,
                                               ls_ls_arr_comp_spatial_ewma_PII,
                                               ls_comp_name,
                                               FLAGS.training_res_folder,
                                               '3D_y_'+fig_name,
                                               FLAGS,
                                               cmap=cmap,
                                               title_size=title_size,
                                               save_sep=save_sep)

    return (arr_t2_scores_spatial_ewma_PI,
            ls_arr_t2_scores_spatial_ewma_PII,
            ls_arr_comp_spatial_ewma_PI,
            ls_ls_arr_comp_spatial_ewma_PII)


def Plot_SpatialHotellingEWMAT2_Other_Score_Prosp(
        fig_name,
        FLAGS,
        cmap=CMAP,
        title_size=LAB_SIZE,
        save_sep=False):
    """ Calculate and plot scores and t2 of scores in 2D spatial image for prospective analysis."""
    arr_t2_scores_spatial_ewma_PI = pickle.load(open(os.path.join(FLAGS.training_res_folder, 't2_scores_spatial_ewma_PI.h5'), 'rb'))
    ls_arr_t2_scores_spatial_ewma_PII = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'ls_t2_scores_spatial_ewma_PII.h5'), 'rb'))
    ls_arr_comp_spatial_ewma_PI = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'ls_metrics_spatial_ewma_PI.h5'), 'rb'))
    ls_ls_arr_comp_spatial_ewma_PII = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'ls_ls_metrics_spatial_ewma_PII.h5'), 'rb'))
    img_arr_PI = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'img_arr_PI.h5'), 'rb'))
    ls_img_arr_PII = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'ls_img_arr_PII.h5'), 'rb'))
    FLAGS = pickle.load(open(os.path.join(FLAGS.training_res_folder, 'visu_FLAGS.h5'), 'rb'))

    n_hei_PI, n_wid_PI = FLAGS.moni_stat_hei_PI, FLAGS.moni_stat_wid_PI 
    n_hei_PII, n_wid_PII = FLAGS.moni_stat_hei_PII, FLAGS.moni_stat_wid_PII
    sigma, wind_len = FLAGS.spatial_ewma_sigma, FLAGS.spatial_ewma_wind_len
    nugget, ls_comp_name = FLAGS.nugget, FLAGS.ls_comp_name
    mu_train, Sinv_train = FLAGS.mu_train, FLAGS.Sinv_train

    # Heatmap
    PlotSaveSpatialHeatMap_Other_Score_Prosp(img_arr_PI, ls_img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             ls_arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             title_size=title_size,
                                             save_sep=save_sep)
    
    # Heatmap overlapped with the original images
    PlotSaveSpatialHeatMap_Other_Score_Prosp(img_arr_PI, ls_img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             ls_arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             'overlap_'+fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             overlap_flag=True,
                                             title_size=title_size,
                                             save_sep=save_sep)
    
    # Scaled heatmap
    PlotSaveSpatialHeatMap_Other_Score_Scale_Prosp(img_arr_PI, ls_img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             ls_arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             'scaled_'+fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             overlap_flag=False,
                                             title_size=title_size,
                                             save_sep=save_sep)

    # Scaled heatmap overlapped with the original images
    PlotSaveSpatialHeatMap_Other_Score_Scale_Prosp(img_arr_PI, ls_img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             ls_arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             'scaled_overlap_'+fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             overlap_flag=True,
                                             title_size=title_size,
                                             save_sep=save_sep)
    
    # 3D heatmap
    PlotSaveSpatial3DHeatMap_Other_Score_Prosp(img_arr_PI, ls_img_arr_PII,
                                               arr_t2_scores_spatial_ewma_PI,
                                               ls_arr_t2_scores_spatial_ewma_PII,
                                               ls_arr_comp_spatial_ewma_PI,
                                               ls_ls_arr_comp_spatial_ewma_PII,
                                               ls_comp_name,
                                               FLAGS.training_res_folder,
                                               '3D_y_'+fig_name,
                                               FLAGS,
                                               cmap=cmap,
                                               title_size=title_size,
                                               save_sep=save_sep)

    return (arr_t2_scores_spatial_ewma_PI, ls_arr_t2_scores_spatial_ewma_PII,
            ls_arr_comp_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PII)


def PlotSaveSpatialHeatMap_Other_Score_Prosp(
        img_arr_PI, ls_img_arr_PII,
        arr_t2_scores_spatial_ewma_PI,
        ls_arr_t2_scores_spatial_ewma_PII,
        ls_arr_comp_spatial_ewma_PI,
        ls_ls_arr_comp_spatial_ewma_PII,
        ls_comp_name,
        folder_path,
        fig_name,
        FLAGS,
        pt_size=PT_SIZE,
        alpha=ALPHA,
        cmap=CMAP,
        overlap_flag=False,
        scale_PII_flag=False,
        label_pad = LAB_PAD,
        title_size = LAB_SIZE,
        save_sep = False):
    """ Plot and save spatial heatmap for scores and other metrics."""
    num_comp_metric = len(ls_arr_comp_spatial_ewma_PI)
    num_plt_row = 2+num_comp_metric
    fig = plt.figure(num=None, figsize=(2.4 * 2 * ONE_FIG_HEI, num_plt_row * 2 * ONE_FIG_HEI), dpi=100, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = WSPACE)
    # Phase-I and -II image
    ax = plt.subplot2grid((num_plt_row,2), (0,0))
    ax.imshow(img_arr_PI, cmap = GRAY_CMAP)
    ax.invert_yaxis()
    # ax.set_title('PI image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PI image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PI_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    ax = plt.subplot2grid((num_plt_row,2), (0,1))
    comb_img_arr_PII = np.block([[ls_img_arr_PII[0], ls_img_arr_PII[1]],[ls_img_arr_PII[2], ls_img_arr_PII[3]]])
    ax.imshow(comb_img_arr_PII, cmap = GRAY_CMAP)
    ax.invert_yaxis()
    # ax.set_title('PII image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PII image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PII_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    
    # Hotelling T2 scores
    hei, wid = img_arr_PI.shape
    n_hei, n_wid = arr_t2_scores_spatial_ewma_PI.shape
    cmap_vmin = min(np.min(arr_t2_scores_spatial_ewma_PI), np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    cmap_vmax = max(np.max(arr_t2_scores_spatial_ewma_PI), np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    ax = plt.subplot2grid((num_plt_row,2), (1,0))
    title = 'Hotelling $T^2$ EWMA Score PI'
    if overlap_flag:
        ax.imshow(img_arr_PI, cmap=GRAY_CMAP)
        upperleft_corner = ((wid-n_wid)//2, (hei-n_hei)//2)
        temp_alp = alpha
    else:
        upperleft_corner = (0,0)
        ax.invert_yaxis()
        temp_alp = 1
    PlotSpatialHeatMap(ax, arr_t2_scores_spatial_ewma_PI, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                        title=title, fig_name=fig_name.split('.')[0]+'_PI_score.png', title_size=title_size, rot=0, upperleft_corner=upperleft_corner, alpha=temp_alp, save_sep=save_sep)
    ax.invert_yaxis()

    hei, wid = ls_img_arr_PII[0].shape
    n_hei, n_wid = ls_arr_t2_scores_spatial_ewma_PII[0].shape
    ax = plt.subplot2grid((num_plt_row,2), (1,1))
    title = 'Hotelling $T^2$ EWMA Score PII'
    if overlap_flag:
        ax.imshow(comb_img_arr_PII, cmap=GRAY_CMAP)
        ls_upperleft_corner = [((wid-n_wid)//2, (hei-n_hei)//2),
                               (wid+(wid-n_wid)//2, (hei-n_hei)//2),
                               ((wid-n_wid)//2, hei+(hei-n_hei)//2),
                               (wid+(wid-n_wid)//2, hei+(hei-n_hei)//2)]
        temp_alp = alpha
    else:
        ls_upperleft_corner = [(0, 0),
                               (n_wid, 0),
                               (0, n_hei),
                               (n_wid, n_hei)]
        ax.invert_yaxis()
        temp_alp = 1
    
    for idx, (upperleft_corner, arr_t2_scores_spatial_ewma_PII) in enumerate(zip(ls_upperleft_corner, ls_arr_t2_scores_spatial_ewma_PII)):
        if idx==FLAGS.num_PII-1:
            PlotSpatialHeatMap(ax, arr_t2_scores_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                            title=title, fig_name=fig_name.split('.')[0]+'_PII_score.png', title_size=title_size, rot=0, upperleft_corner=upperleft_corner, alpha=temp_alp, save_sep=save_sep)
        else:
            PlotSpatialHeatMap(ax, arr_t2_scores_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                            title=None, fig_name=None, title_size=title_size, rot=0, upperleft_corner=upperleft_corner, config_prop=False, alpha=temp_alp, save_sep=False)
    ax.invert_yaxis()

    # Comparing metric
    for idx_comp, (comp_name, arr_comp_spatial_ewma_PI, ls_arr_comp_spatial_ewma_PII) in enumerate(zip(ls_comp_name, ls_arr_comp_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PII)): 
        cmap_vmin = min(np.min(arr_comp_spatial_ewma_PI), np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        cmap_vmax = max(np.max(arr_comp_spatial_ewma_PI), np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        
        hei, wid = img_arr_PI.shape
        n_hei, n_wid = arr_comp_spatial_ewma_PI.shape
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,0))
        title = ' '.join([comp_name, 'PI'])
        if overlap_flag:
            ax.imshow(img_arr_PI, cmap=GRAY_CMAP)
            upperleft_corner = ((wid-n_wid)//2, (hei-n_hei)//2)
            temp_alp=alpha
        else:
            upperleft_corner = (0,0)
            ax.invert_yaxis()
            temp_alp=1
        PlotSpatialHeatMap(ax, arr_comp_spatial_ewma_PI, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                            title=title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, rot=0, upperleft_corner=upperleft_corner, alpha=temp_alp, save_sep=save_sep)
        ax.invert_yaxis()

        hei, wid = ls_img_arr_PII[0].shape
        n_hei, n_wid = ls_arr_comp_spatial_ewma_PII[0].shape
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,1))
        title = ' '.join([comp_name, 'PII'])
        if overlap_flag:
            ax.imshow(comb_img_arr_PII, cmap=GRAY_CMAP)
            ls_upperleft_corner = [((wid-n_wid)//2, (hei-n_hei)//2),
                                   (wid+(wid-n_wid)//2, (hei-n_hei)//2),
                                   ((wid-n_wid)//2, hei+(hei-n_hei)//2),
                                   (wid+(wid-n_wid)//2, hei+(hei-n_hei)//2)]
            temp_alp = alpha
        else:
            ls_upperleft_corner = [(0, 0),
                                   (n_wid, 0),
                                   (0, n_hei),
                                   (n_wid, n_hei)]
            ax.invert_yaxis()
            temp_alp = 1

        for idx, (upperleft_corner, arr_comp_spatial_ewma_PII) in enumerate(zip(ls_upperleft_corner, ls_arr_comp_spatial_ewma_PII)):
            if idx==FLAGS.num_PII-1:
                PlotSpatialHeatMap(ax, arr_comp_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                                title=title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, rot=0, upperleft_corner=upperleft_corner, alpha=temp_alp, save_sep=save_sep)
            else:
                PlotSpatialHeatMap(ax, arr_comp_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                                title=None, fig_name=None, title_size=title_size, rot=0, upperleft_corner=upperleft_corner, config_prop=False, alpha=temp_alp, save_sep=False)
        ax.invert_yaxis()

    # plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.savefig(os.path.join(folder_path, fig_name))
    plt.close()


def PlotSaveSpatialHeatMap_Other_Score_Scale_Prosp(
        img_arr_PI, ls_img_arr_PII,
        arr_t2_scores_spatial_ewma_PI,
        ls_arr_t2_scores_spatial_ewma_PII,
        ls_arr_comp_spatial_ewma_PI,
        ls_ls_arr_comp_spatial_ewma_PII,
        ls_comp_name,
        folder_path,
        fig_name,
        FLAGS,
        pt_size=PT_SIZE,
        alpha=ALPHA,
        cmap=CMAP,
        overlap_flag=False,
        scale_PII_flag=False,
        label_pad = LAB_PAD,
        title_size = LAB_SIZE,
        save_sep = False):
    """ Plot and save spatial heatmap for scores and other metrics."""
    num_comp_metric = len(ls_arr_comp_spatial_ewma_PI)
    num_plt_row = 2+num_comp_metric
    fig = plt.figure(num=None, figsize=(2.4 *  2 * ONE_FIG_HEI, num_plt_row * 2 * ONE_FIG_HEI), dpi=100, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = WSPACE)
    # Phase-I and -II image
    ax = plt.subplot2grid((num_plt_row,2), (0,0))
    ax.imshow(img_arr_PI, cmap = GRAY_CMAP)
    ax.invert_yaxis()
    # ax.set_title('PI image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PI image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PI_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    ax = plt.subplot2grid((num_plt_row,2), (0,1))
    comb_img_arr_PII = np.block([[ls_img_arr_PII[0], ls_img_arr_PII[1]],[ls_img_arr_PII[2], ls_img_arr_PII[3]]])
    ax.imshow(comb_img_arr_PII, cmap = GRAY_CMAP)
    ax.invert_yaxis()
    # ax.set_title('PII image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PII image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PII_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    
    # Hotelling T2 scores
    hei, wid = img_arr_PI.shape
    n_hei, n_wid = arr_t2_scores_spatial_ewma_PI.shape
    ax = plt.subplot2grid((num_plt_row,2), (1,0))
    title = 'Hotelling $T^2$ EWMA Score PI'
    if overlap_flag:
        ax.imshow(img_arr_PI, cmap=GRAY_CMAP)
        upperleft_corner = ((wid-n_wid)//2, (hei-n_hei)//2)
        temp_alp = alpha
    else:
        upperleft_corner = (0,0)
        ax.invert_yaxis()
        temp_alp = 1
    PlotSpatialHeatMap(ax, arr_t2_scores_spatial_ewma_PI, fig, FLAGS, vmin=None, vmax=None, cmap=cmap,
                        title=title, fig_name=fig_name.split('.')[0]+'_PI_score.png', title_size=title_size, rot=0, upperleft_corner=upperleft_corner, alpha=temp_alp, save_sep=save_sep)
    ax.invert_yaxis()

    hei, wid = ls_img_arr_PII[0].shape
    n_hei, n_wid = ls_arr_t2_scores_spatial_ewma_PII[0].shape 
    ax = plt.subplot2grid((num_plt_row,2), (1,1))
    title = 'Hotelling $T^2$ EWMA Score PII'
    if overlap_flag:
        ax.imshow(comb_img_arr_PII, cmap=GRAY_CMAP)
        ls_upperleft_corner = [((wid-n_wid)//2, (hei-n_hei)//2),
                               (wid+(wid-n_wid)//2, (hei-n_hei)//2),
                               ((wid-n_wid)//2, hei+(hei-n_hei)//2),
                               (wid+(wid-n_wid)//2, hei+(hei-n_hei)//2)]
        temp_alp = alpha
    else:
        ls_upperleft_corner = [(0, 0),
                               (n_wid, 0),
                               (0, n_hei),
                               (n_wid, n_hei)]
        ax.invert_yaxis()
        temp_alp = 1

    print("Score PI: ({:.4f},{:.4f}).".format(np.min(arr_t2_scores_spatial_ewma_PI), np.max(arr_t2_scores_spatial_ewma_PI)))
    print("Score before scaling: ({:.4f},{:.4f}), ({:.4f},{:.4f}), ({:.4f},{:.4f}), ({:.4f},{:.4f}).".format(*(np.array([[np.min(arr), np.max(arr)] for arr in ls_arr_t2_scores_spatial_ewma_PII]).reshape(-1,))))
    ls_arr_t2_scores_spatial_ewma_PII = [(arr-np.min(arr_t2_scores_spatial_ewma_PI))/(np.max(arr_t2_scores_spatial_ewma_PI)-np.min(arr_t2_scores_spatial_ewma_PI)) for arr in ls_arr_t2_scores_spatial_ewma_PII]
    print("Score after scaling: ({:.4f},{:.4f}), ({:.4f},{:.4f}), ({:.4f},{:.4f}), ({:.4f},{:.4f}).".format(*(np.array([[np.min(arr), np.max(arr)] for arr in ls_arr_t2_scores_spatial_ewma_PII]).reshape(-1,))))
    cmap_vmin = np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII])
    cmap_vmax = np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII])

    for idx, (upperleft_corner, arr_t2_scores_spatial_ewma_PII) in enumerate(zip(ls_upperleft_corner, ls_arr_t2_scores_spatial_ewma_PII)):
        if idx==FLAGS.num_PII-1:
            PlotSpatialHeatMap(ax, arr_t2_scores_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                            title=title, fig_name=fig_name.split('.')[0]+'_PII_score.png', title_size=title_size, rot=0, upperleft_corner=upperleft_corner, alpha=temp_alp, save_sep=False)
        else:
            PlotSpatialHeatMap(ax, arr_t2_scores_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                            title=None, fig_name=None, title_size=title_size, rot=0, upperleft_corner=upperleft_corner, config_prop=False, alpha=temp_alp, save_sep=False)
    ax.invert_yaxis()

    # Comparing metric
    for idx_comp, (comp_name, arr_comp_spatial_ewma_PI, ls_arr_comp_spatial_ewma_PII) in enumerate(zip(ls_comp_name, ls_arr_comp_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PII)): 
        hei, wid = img_arr_PI.shape
        n_hei, n_wid = arr_comp_spatial_ewma_PI.shape
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,0))
        title = ' '.join([comp_name, 'PI'])
        if overlap_flag:
            ax.imshow(img_arr_PI, cmap=GRAY_CMAP)
            upperleft_corner = ((wid-n_wid)//2, (hei-n_hei)//2)
            temp_alp=alpha
        else:
            upperleft_corner = (0,0)
            ax.invert_yaxis()
            temp_alp=1
        PlotSpatialHeatMap(ax, arr_comp_spatial_ewma_PI, fig, FLAGS, vmin=None, vmax=None, cmap=cmap,
                            title=title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, rot=0, upperleft_corner=upperleft_corner, alpha=temp_alp, save_sep=save_sep)
        ax.invert_yaxis()

        hei, wid = ls_img_arr_PII[0].shape
        n_hei, n_wid = ls_arr_comp_spatial_ewma_PII[0].shape
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,1))
        title = ' '.join([comp_name, 'PII'])
        if overlap_flag:
            ax.imshow(comb_img_arr_PII, cmap=GRAY_CMAP)
            ls_upperleft_corner = [((wid-n_wid)//2, (hei-n_hei)//2),
                                   (wid+(wid-n_wid)//2, (hei-n_hei)//2),
                                   ((wid-n_wid)//2, hei+(hei-n_hei)//2),
                                   (wid+(wid-n_wid)//2, hei+(hei-n_hei)//2)]
            temp_alp = alpha
        else:
            ls_upperleft_corner = [(0, 0),
                                   (n_wid, 0),
                                   (0, n_hei),
                                   (n_wid, n_hei)]
            ax.invert_yaxis()
            temp_alp = 1
        print(comp_name + " PI: ({:.4f},{:.4f}).".format(np.min(arr_comp_spatial_ewma_PI), np.max(arr_comp_spatial_ewma_PI)))
        print(comp_name + " before scaling: ({:.4f},{:.4f}), ({:.4f},{:.4f}), ({:.4f},{:.4f}), ({:.4f},{:.4f}).".format(*(np.array([[np.min(arr), np.max(arr)] for arr in ls_arr_comp_spatial_ewma_PII]).reshape(-1,))))
        ls_arr_comp_spatial_ewma_PII = [(arr-np.min(arr_comp_spatial_ewma_PI))/(np.max(arr_comp_spatial_ewma_PI)-np.min(arr_comp_spatial_ewma_PI)) for arr in ls_arr_comp_spatial_ewma_PII]
        print(comp_name + " after scaling: ({:.4f},{:.4f}), ({:.4f},{:.4f}), ({:.4f},{:.4f}), ({:.4f},{:.4f}).".format(*(np.array([[np.min(arr), np.max(arr)] for arr in ls_arr_comp_spatial_ewma_PII]).reshape(-1,))))
        cmap_vmin = np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII])
        cmap_vmax = np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII])
        
        for idx, (upperleft_corner, arr_comp_spatial_ewma_PII) in enumerate(zip(ls_upperleft_corner, ls_arr_comp_spatial_ewma_PII)):
            if idx==FLAGS.num_PII-1:
                PlotSpatialHeatMap(ax, arr_comp_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                                title=title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, rot=0, upperleft_corner=upperleft_corner, alpha=temp_alp, save_sep=False)
            else:
                PlotSpatialHeatMap(ax, arr_comp_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                                title=None, fig_name=None, title_size=title_size, rot=0, upperleft_corner=upperleft_corner, config_prop=False, alpha=temp_alp, save_sep=False)
        ax.invert_yaxis()

    # plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.savefig(os.path.join(folder_path, fig_name))
    plt.close()


def PlotSaveSpatial3DHeatMap_Other_Score_Prosp(
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
        thre_kwargs = {'linewidth': 2, 'edgecolor': 'k', 'linestyle': '-', 'alpha': 0.3},
        save_sep = False):
    """ Plot and save spatial heatmap for scores and other metrics."""
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

    hei, wid = img_arr_PI.shape
    n_hei, n_wid = arr_t2_scores_spatial_ewma_PI.shape
    cmap_vmin = min(np.min(arr_t2_scores_spatial_ewma_PI), np.min([np.min(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    cmap_vmax = max(np.max(arr_t2_scores_spatial_ewma_PI), np.max([np.max(arr) for arr in ls_arr_t2_scores_spatial_ewma_PII]))
    ax = plt.subplot2grid((num_plt_row,2), (1,0), projection='3d')
    title = 'Hotelling $T^2$ EWMA Score PI'
    PlotSpatial3DHeatMap(ax, arr_t2_scores_spatial_ewma_PI, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                         title=title, fig_name=fig_name.split('.')[0]+'_PI_score.png', title_size=title_size, 
                         thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=[lcl, ucl], save_sep=save_sep)
    
    hei, wid = ls_img_arr_PII[0].shape
    n_hei, n_wid = ls_arr_t2_scores_spatial_ewma_PII[0].shape
    ax = plt.subplot2grid((num_plt_row,2), (1,1), projection='3d')
    title = 'Hotelling $T^2$ EWMA Score PII'
    ls_upperleft_corner = [(0, 0),
                           (n_wid, 0),
                           (0, n_hei),
                           (n_wid, n_hei)]
    for idx, (upperleft_corner, arr_t2_scores_spatial_ewma_PII) in enumerate(zip(ls_upperleft_corner, ls_arr_t2_scores_spatial_ewma_PII)):
        if idx==FLAGS.num_PII-1:
            PlotSpatial3DHeatMap(ax, arr_t2_scores_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                         title=title, fig_name=fig_name.split('.')[0]+'_PII_score.png', title_size=title_size, upperleft_corner=upperleft_corner, 
                         thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=[lcl, ucl], z_lim=(cmap_vmin, cmap_vmax), save_sep=save_sep)
        else:
            PlotSpatial3DHeatMap(ax, arr_t2_scores_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                         title=None, fig_name=None, title_size=title_size, upperleft_corner=upperleft_corner, config_prop=False, 
                         thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=[lcl, ucl], save_sep=False)

    # Comparing metric
    for idx_comp, (comp_name, arr_comp_spatial_ewma_PI, ls_arr_comp_spatial_ewma_PII) in enumerate(zip(ls_comp_name, ls_arr_comp_spatial_ewma_PI, ls_ls_arr_comp_spatial_ewma_PII)):
        ucl = np.percentile(arr_comp_spatial_ewma_PI, (100+alarm_level)/2)
        lcl = np.percentile(arr_comp_spatial_ewma_PI, (100-alarm_level)/2)

        hei, wid = img_arr_PI.shape
        n_hei, n_wid = arr_comp_spatial_ewma_PI.shape
        cmap_vmin = min(np.min(arr_comp_spatial_ewma_PI), np.min([np.min(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        cmap_vmax = max(np.max(arr_comp_spatial_ewma_PI), np.max([np.max(arr) for arr in ls_arr_comp_spatial_ewma_PII]))
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,0), projection='3d')
        title = ' '.join([comp_name, 'PI'])
        PlotSpatial3DHeatMap(ax, arr_comp_spatial_ewma_PI, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                             title=title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, 
                             thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=[lcl, ucl], save_sep=save_sep)
        
        hei, wid = ls_img_arr_PII[0].shape
        n_hei, n_wid = ls_arr_comp_spatial_ewma_PII[0].shape
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,1), projection='3d')
        title = ' '.join([comp_name, 'PII'])
        ls_upperleft_corner = [(0, 0),
                               (n_wid, 0),
                               (0, n_hei),
                               (n_wid, n_hei)]
        for idx, (upperleft_corner, arr_comp_spatial_ewma_PII) in enumerate(zip(ls_upperleft_corner, ls_arr_comp_spatial_ewma_PII)):
            if idx==FLAGS.num_PII-1:
                PlotSpatial3DHeatMap(ax, arr_comp_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                             title=title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, upperleft_corner=upperleft_corner, 
                             thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=[lcl, ucl], z_lim=(cmap_vmin, cmap_vmax), save_sep=save_sep)
            else:
                PlotSpatial3DHeatMap(ax, arr_comp_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                             title=None, fig_name=None, title_size=title_size, upperleft_corner=upperleft_corner, config_prop=False,
                             thre_flag=thre_flag, thre_kwargs=thre_kwargs, arr_thres=[lcl, ucl], save_sep=False)
    
    # plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.savefig(os.path.join(folder_path, fig_name))
    plt.close()


def SpatialHotellingEWMAT2_Other_Score_PIPII(
        img_arr_PI, 
        img_arr_PII,
        score_PI,
        score_PII,
        ls_comp_PI,
        ls_comp_PII,
        n_hei,
        n_wid,
        sigma,
        wind_len,
        nugget,
        fig_name,
        ls_comp_name,
        FLAGS,
        mu_train, Sinv_train,
        cmap=CMAP,
        title_size=LAB_SIZE,
        save_sep=False):
    """ Calculate and plot scores and t2 of scores in 2D spatial image."""
    # Spatial EWMA-T2 of scores.
    arr_t2_scores_spatial_ewma_PI = SpatialHotellingEWMAT2(score_PI, n_hei, n_wid, sigma, wind_len, nugget, mu_train, Sinv_train, FLAGS)
    arr_t2_scores_spatial_ewma_PII = SpatialHotellingEWMAT2(score_PII, n_hei, n_wid, sigma, wind_len, nugget, mu_train, Sinv_train, FLAGS)
    print("The first 10 score for PI is: {}.\n".format(score_PI[:10],))
    print("The first 10 score for PII is: {}.\n".format(score_PII[:10],))
    print("The first 10 hotelling T2 for PI is {}.\n".format(arr_t2_scores_spatial_ewma_PI[:10],))
    print("The first 10 hotelling T2 for PII is {}.\n".format(arr_t2_scores_spatial_ewma_PII[:10],))
    # Comp statistics
    ls_arr_comp_spatial_ewma_PI = [ScoresSpatialEWMA(comp_PI[:,np.newaxis], n_hei, n_wid, sigma, wind_len).squeeze(axis=-1) for comp_PI in ls_comp_PI]
    ls_arr_comp_spatial_ewma_PII = [ScoresSpatialEWMA(comp_PII[:,np.newaxis], n_hei, n_wid, sigma, wind_len).squeeze(axis=-1) for comp_PII in ls_comp_PII]

    # Heatmap
    PlotSaveSpatialHeatMap_Other_Score_PIPII(img_arr_PI, img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             title_size=title_size,
                                             save_sep=save_sep)
    
    # Heatmap overlapped with the original images
    PlotSaveSpatialHeatMap_Other_Score_PIPII(img_arr_PI, img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             'overlap_'+fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             overlap_flag=True,
                                             title_size=title_size,
                                             save_sep=save_sep)
    
    # Scaled heatmap
    PlotSaveSpatialHeatMap_Other_Score_Scale_PIPII(img_arr_PI, img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             'scaled_'+fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             overlap_flag=False,
                                             title_size=title_size,
                                             save_sep=save_sep)

    # Scaled heatmap overlapped with the original images
    PlotSaveSpatialHeatMap_Other_Score_Scale_PIPII(img_arr_PI, img_arr_PII,
                                             arr_t2_scores_spatial_ewma_PI,
                                             arr_t2_scores_spatial_ewma_PII,
                                             ls_arr_comp_spatial_ewma_PI,
                                             ls_arr_comp_spatial_ewma_PII,
                                             ls_comp_name,
                                             FLAGS.training_res_folder,
                                             'scaled_overlap_'+fig_name,
                                             FLAGS,
                                             cmap=cmap,
                                             overlap_flag=True,
                                             title_size=title_size,
                                             save_sep=save_sep)
    
    # 3D heatmap
    PlotSaveSpatial3DHeatMap_Other_Score_PIPII(img_arr_PI, img_arr_PII,
                                               arr_t2_scores_spatial_ewma_PI,
                                               arr_t2_scores_spatial_ewma_PII,
                                               ls_arr_comp_spatial_ewma_PI,
                                               ls_arr_comp_spatial_ewma_PII,
                                               ls_comp_name,
                                               FLAGS.training_res_folder,
                                               '3D_y_'+fig_name,
                                               FLAGS,
                                               cmap=cmap,
                                               title_size=title_size,
                                               save_sep=save_sep)

    return (arr_t2_scores_spatial_ewma_PI,
            arr_t2_scores_spatial_ewma_PII,
            ls_arr_comp_spatial_ewma_PI,
            ls_arr_comp_spatial_ewma_PII)
    
    
def PlotSaveSpatialHeatMap_Other_Score_PIPII(
        img_arr_PI, img_arr_PII,
        arr_t2_scores_spatial_ewma_PI,
        arr_t2_scores_spatial_ewma_PII,
        ls_arr_comp_spatial_ewma_PI,
        ls_arr_comp_spatial_ewma_PII,
        ls_comp_name,
        folder_path,
        fig_name,
        FLAGS,
        pt_size=PT_SIZE,
        alpha=ALPHA,
        cmap=CMAP,
        overlap_flag=False,
        scale_PII_flag=False,
        label_pad = LAB_PAD,
        title_size = LAB_SIZE,
        save_sep = False):
    """ Plot and save spatial heatmap for scores and other metrics."""
    num_comp_metric = len(ls_arr_comp_spatial_ewma_PI)
    num_plt_row = 2+num_comp_metric
    fig = plt.figure(num=None, figsize=(2.4 *  2 * ONE_FIG_HEI, num_plt_row * 2 * ONE_FIG_HEI), dpi=100, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = WSPACE)
    # Phase-I and -II image
    ax = plt.subplot2grid((num_plt_row,2), (0,0))
    ax.imshow(img_arr_PI, cmap = GRAY_CMAP)
    # ax.set_title('PI image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PI image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PI_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    ax = plt.subplot2grid((num_plt_row,2), (0,1))
    ax.imshow(img_arr_PII, cmap = GRAY_CMAP)
    # ax.set_title('PII image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PII image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PII_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    # Hotelling T2 scores
    cmap_vmin = min(np.min(arr_t2_scores_spatial_ewma_PI), np.min(arr_t2_scores_spatial_ewma_PII))
    cmap_vmax = max(np.max(arr_t2_scores_spatial_ewma_PI), np.max(arr_t2_scores_spatial_ewma_PII))
    ax = plt.subplot2grid((num_plt_row,2), (1,0))
    title = 'Hotelling $T^2$ EWMA Score PI'
    if overlap_flag:
        PlotSpatialHeatMapOverlap(ax, arr_t2_scores_spatial_ewma_PI, img_arr_PI, fig, FLAGS, cmap_vmin, cmap_vmax, pt_size=pt_size, alpha=alpha, cmap=cmap,
                                  title=title, fig_name=fig_name.split('.')[0]+'_PI_score.png', title_size=title_size, save_sep=save_sep)
    else:
        PlotSpatialHeatMap(ax, arr_t2_scores_spatial_ewma_PI, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                           title=title, fig_name=fig_name.split('.')[0]+'_PI_score.png', title_size=title_size, rot=0, save_sep=save_sep)
    ax = plt.subplot2grid((num_plt_row,2), (1,1))
    title = 'Hotelling $T^2$ EWMA Score PII'
    if overlap_flag:
        PlotSpatialHeatMapOverlap(ax, arr_t2_scores_spatial_ewma_PII, img_arr_PII, fig, FLAGS, cmap_vmin, cmap_vmax, pt_size=pt_size, alpha=alpha, cmap=cmap,
                           title=title, fig_name=fig_name.split('.')[0]+'_PII_score.png', title_size=title_size, save_sep=save_sep)
    else:
        PlotSpatialHeatMap(ax, arr_t2_scores_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                           title=title, fig_name=fig_name.split('.')[0]+'_PII_score.png', title_size=title_size, rot=0, save_sep=save_sep)
    # Comparing metric
    for idx_comp, (comp_name, arr_comp_spatial_ewma_PI, arr_comp_spatial_ewma_PII) in enumerate(zip(ls_comp_name, ls_arr_comp_spatial_ewma_PI, ls_arr_comp_spatial_ewma_PII)): 
        cmap_vmin = min(np.min(arr_comp_spatial_ewma_PI), np.min(arr_comp_spatial_ewma_PII))
        cmap_vmax = max(np.max(arr_comp_spatial_ewma_PI), np.max(arr_comp_spatial_ewma_PII))
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,0))
        title = ' '.join([comp_name, 'PI'])
        if overlap_flag:
            PlotSpatialHeatMapOverlap(ax, arr_comp_spatial_ewma_PI, img_arr_PI, fig, FLAGS, cmap_vmin, cmap_vmax, pt_size=pt_size, alpha=alpha, cmap=cmap,
                                      title=title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, save_sep=save_sep)
        else:
            PlotSpatialHeatMap(ax, arr_comp_spatial_ewma_PI, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                               title=title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, rot=0, save_sep=save_sep)
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,1))
        title = ' '.join([comp_name, 'PII'])
        if overlap_flag:
            PlotSpatialHeatMapOverlap(ax, arr_comp_spatial_ewma_PII, img_arr_PII, fig, FLAGS, cmap_vmin, cmap_vmax, pt_size=pt_size, alpha=alpha, cmap=cmap,
                                      title=title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, save_sep=save_sep)
        else:
            PlotSpatialHeatMap(ax, arr_comp_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                               title=title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, rot=0, save_sep=save_sep)

    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.close()


def PlotSaveSpatialHeatMap_Other_Score_Scale_PIPII(
        img_arr_PI, img_arr_PII,
        arr_t2_scores_spatial_ewma_PI,
        arr_t2_scores_spatial_ewma_PII,
        ls_arr_comp_spatial_ewma_PI,
        ls_arr_comp_spatial_ewma_PII,
        ls_comp_name,
        folder_path,
        fig_name,
        FLAGS,
        pt_size=PT_SIZE,
        alpha=ALPHA,
        cmap=CMAP,
        overlap_flag=False,
        label_pad = LAB_PAD,
        title_size = LAB_SIZE,
        save_sep = False):
    """ Plot and save spatial heatmap for scores and other metrics."""
    num_comp_metric = len(ls_arr_comp_spatial_ewma_PI)
    num_plt_row = 2+num_comp_metric
    fig = plt.figure(num=None, figsize=(2.4 *  2 * ONE_FIG_HEI, num_plt_row * 2 * ONE_FIG_HEI), dpi=100, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = WSPACE)
    # Phase-I and -II image
    ax = plt.subplot2grid((num_plt_row,2), (0,0))
    ax.imshow(img_arr_PI, cmap = GRAY_CMAP)
    # ax.set_title('PI image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PI image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PI_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    ax = plt.subplot2grid((num_plt_row,2), (0,1))
    ax.imshow(img_arr_PII, cmap = GRAY_CMAP)
    # ax.set_title('PII image', size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PII image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PII_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    # Hotelling T2 scores
    cmap_vmin, cmap_vmax = None, None
    ax = plt.subplot2grid((num_plt_row,2), (1,0))
    title = 'Hotelling $T^2$ EWMA Score PI'
    if overlap_flag:
        PlotSpatialHeatMapOverlap(ax, arr_t2_scores_spatial_ewma_PI, img_arr_PI, fig, FLAGS, cmap_vmin, cmap_vmax, pt_size=pt_size, alpha=alpha, cmap=cmap,
                                  title=title, fig_name=fig_name.split('.')[0]+'_PI_score.png', title_size=title_size, save_sep=save_sep)
    else:
        PlotSpatialHeatMap(ax, arr_t2_scores_spatial_ewma_PI, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                           title=title, fig_name=fig_name.split('.')[0]+'_PI_score.png', title_size=title_size, rot=0, save_sep=save_sep)
    ax = plt.subplot2grid((num_plt_row,2), (1,1))
    title = 'Hotelling $T^2$ EWMA Score PII'
    arr_t2_scores_spatial_ewma_PII = (arr_t2_scores_spatial_ewma_PII-np.min(arr_t2_scores_spatial_ewma_PI))/(np.max(arr_t2_scores_spatial_ewma_PI)-np.min(arr_t2_scores_spatial_ewma_PI))
    if overlap_flag:
        PlotSpatialHeatMapOverlap(ax, arr_t2_scores_spatial_ewma_PII, img_arr_PII, fig, FLAGS, cmap_vmin, cmap_vmax, pt_size=pt_size, alpha=alpha, cmap=cmap,
                           title=title, fig_name=fig_name.split('.')[0]+'_PII_score.png', title_size=title_size, save_sep=save_sep)
    else:
        PlotSpatialHeatMap(ax, arr_t2_scores_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                           title=title, fig_name=fig_name.split('.')[0]+'_PII_score.png', title_size=title_size, rot=0, save_sep=save_sep)
    # Comparing metric
    for idx_comp, (comp_name, arr_comp_spatial_ewma_PI, arr_comp_spatial_ewma_PII) in enumerate(zip(ls_comp_name, ls_arr_comp_spatial_ewma_PI, ls_arr_comp_spatial_ewma_PII)): 
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,0))
        title = ' '.join([comp_name, 'PI'])
        if overlap_flag:
            PlotSpatialHeatMapOverlap(ax, arr_comp_spatial_ewma_PI, img_arr_PI, fig, FLAGS, cmap_vmin, cmap_vmax, pt_size=pt_size, alpha=alpha, cmap=cmap,
                                      title=title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, save_sep=save_sep)
        else:
            PlotSpatialHeatMap(ax, arr_comp_spatial_ewma_PI, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                               title=title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, rot=0, save_sep=save_sep)
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,1))
        title = ' '.join([comp_name, 'PII'])
        arr_comp_spatial_ewma_PII = (arr_comp_spatial_ewma_PII-np.min(arr_comp_spatial_ewma_PI))/(np.max(arr_comp_spatial_ewma_PI)-np.min(arr_comp_spatial_ewma_PI))
        if overlap_flag:
            PlotSpatialHeatMapOverlap(ax, arr_comp_spatial_ewma_PII, img_arr_PII, fig, FLAGS, cmap_vmin, cmap_vmax, pt_size=pt_size, alpha=alpha, cmap=cmap,
                                      title=title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, save_sep=save_sep)
        else:
            PlotSpatialHeatMap(ax, arr_comp_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                               title=title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, rot=0, save_sep=save_sep)

    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.close()


def PlotSaveSpatial3DHeatMap_Other_Score_PIPII(
        img_arr_PI, img_arr_PII,
        arr_t2_scores_spatial_ewma_PI,
        arr_t2_scores_spatial_ewma_PII,
        ls_arr_comp_spatial_ewma_PI,
        ls_arr_comp_spatial_ewma_PII,
        ls_comp_name,
        folder_path,
        fig_name,
        FLAGS,
        cmap=CMAP,
        label_pad = LAB_PAD,
        title_size = LAB_SIZE,
        save_sep = False):
    """ Plot and save spatial heatmap for scores and other metrics."""
    num_comp_metric = len(ls_arr_comp_spatial_ewma_PI)
    num_plt_row = 2+num_comp_metric
    fig = plt.figure(num=None, figsize=(2.4 *  2 * ONE_FIG_HEI, num_plt_row * 2 * ONE_FIG_HEI), dpi=100, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = WSPACE)
    # Phase-I and -II image
    ax = plt.subplot2grid((num_plt_row,2), (0,0))
    ax.imshow(img_arr_PI, cmap = GRAY_CMAP)
    # ax.set_title('PI image', size = title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PI image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PI_orig.png'), 
                    bbox_inches=extent.expanded(1.4, 1.4))
    ax = plt.subplot2grid((num_plt_row,2), (0,1))
    ax.imshow(img_arr_PII, cmap = GRAY_CMAP)
    # ax.set_title('PII image', size = title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title='PII image')
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name.split('.')[0]+'_PII_orig.png'),
                    bbox_inches=extent.expanded(1.4, 1.4))
    # Hotelling T2 scores
    cmap_vmin = min(np.min(arr_t2_scores_spatial_ewma_PI), np.min(arr_t2_scores_spatial_ewma_PII))
    cmap_vmax = max(np.max(arr_t2_scores_spatial_ewma_PI), np.max(arr_t2_scores_spatial_ewma_PII))
    ax = plt.subplot2grid((num_plt_row,2), (1,0), projection='3d')
    title = 'Hotelling $T^2$ EWMA Score PI'
    PlotSpatial3DHeatMap(ax, arr_t2_scores_spatial_ewma_PI, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                         title=title, fig_name=fig_name.split('.')[0]+'_PI_score.png', title_size=title_size, save_sep=save_sep)
    ax = plt.subplot2grid((num_plt_row,2), (1,1), projection='3d')
    title = 'Hotelling $T^2$ EWMA Score PII'
    PlotSpatial3DHeatMap(ax, arr_t2_scores_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                         title=title, fig_name=fig_name.split('.')[0]+'_PII_score.png', title_size=title_size, save_sep=save_sep)
    # Comparing metric
    for idx_comp, (comp_name, arr_comp_spatial_ewma_PI, arr_comp_spatial_ewma_PII) in enumerate(zip(ls_comp_name, ls_arr_comp_spatial_ewma_PI, ls_arr_comp_spatial_ewma_PII)):
        cmap_vmin = min(np.min(arr_comp_spatial_ewma_PI), np.min(arr_comp_spatial_ewma_PII))
        cmap_vmax = max(np.max(arr_comp_spatial_ewma_PI), np.max(arr_comp_spatial_ewma_PII))
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,0), projection='3d')
        title = ' '.join([comp_name, 'PI'])
        PlotSpatial3DHeatMap(ax, arr_comp_spatial_ewma_PI, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                             title=title, fig_name=fig_name.split('.')[0]+'_PI_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, save_sep=save_sep)
        ax = plt.subplot2grid((num_plt_row,2), (2+idx_comp,1), projection='3d')
        title = ' '.join([comp_name, 'PII'])
        PlotSpatial3DHeatMap(ax, arr_comp_spatial_ewma_PII, fig, FLAGS, cmap_vmin, cmap_vmax, cmap=cmap,
                             title=title, fig_name=fig_name.split('.')[0]+'_PII_'+'_'.join(comp_name.split(' '))+'.png', title_size=title_size, save_sep=save_sep)
    
    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.close()


def PlotARSeries(ts_PI, ts_PII, N_PIIs, folder_path, fig_name, time_step):
    label_size = LAB_SIZE
    plt.figure(num=None, figsize=(10, 0.8 * ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE)
    ax1 = plt.subplot(111)
    xlabel_name = "Phase-I&-II Observation Index"
    ylabel_name = "Time Series Data"
    N_PII = np.sum(N_PIIs)
    PlotMonStatOneAxisPIPII(ax1, ts_PI, ts_PII, xlabel_name, ylabel_name, label_size, [], [], np.arange(N_PII), N_PIIs, time_step)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.close()


def PlotSpatialHeatMap(ax, arr_2D, fig=None, FLAGS=None, vmin=None, vmax=None, cmap=CMAP, 
                       title='', fig_name='', title_size=LAB_SIZE, label_pad=LAB_PAD, rot=0, upperleft_corner=(0,0), config_prop=True, alpha=1, as_ratio=1, save_sep=False):
    # ax.imshow(arr_2D, cmap='hot', interpolation='nearest')
    
    # # Get colorbar from sns.heatmap: https://stackoverflow.com/a/53095480/4307919
    # hm_ax = sns.heatmap(arr_2D, linewidth=0.0, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, cbar_kws={"shrink": 1.0,"aspect": 20.0})
    # cbar = hm_ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=0.65*title_size)
    
    # Use pcolormesh to plot heatmap so that we can control the colorbar easily: https://stackoverflow.com/a/54088910/4307919
    X, Y = np.meshgrid(upperleft_corner[0]+np.arange(arr_2D.shape[1]), upperleft_corner[1]+np.arange(arr_2D.shape[0]))
    hm = ax.pcolormesh(X, Y, arr_2D, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

    if config_prop:
        cbar = plt.colorbar(hm, ax=ax, shrink=CB_SHRINK, aspect=20.0, pad=CB_PAD)
        cbar.ax.tick_params(labelsize=0.65*title_size)

        # ax.set_aspect(1.0)
        # ax.set_xlabel('$X$', size=0.75*title_size, labelpad=LAB_PAD)
        # ax.set_ylabel('$Y$', size=0.75*title_size, labelpad=LAB_PAD)
        # ax.xaxis.set_tick_params(labelsize=0.65*title_size)
        # ax.yaxis.set_tick_params(labelsize=0.65*title_size)
        # y_len, x_len = arr_2D.shape
        # ax.set_xticks(list(range(0,y_len,y_len//TICK_NUM)))
        # ax.set_yticks(list(range(0,x_len,x_len//TICK_NUM)))
        # if title!='':
        #     ax.set_title(title, size=title_size)
        Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title=title, rot=rot, as_ratio=as_ratio)
        if save_sep:
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name), 
                        bbox_inches=extent.expanded(1.4, 1.4))


def PlotSpatialHeatMapOverlap(ax, arr_2D, arr_img, fig=None, FLAGS=None, vmin=None, vmax=None, pt_size=1.0, alpha=0.5, cmap=CMAP, 
                              title='', fig_name='', title_size=LAB_SIZE, label_pad=LAB_PAD, save_sep=False):
    ax.imshow(arr_img, cmap=GRAY_CMAP)
    hei, wid = arr_img.shape
    n_hei, n_wid = arr_2D.shape
    coord_x, coord_y = np.meshgrid(np.arange((wid-n_wid)//2, (wid+n_wid)//2),np.arange((hei-n_hei)//2, (hei+n_hei)//2))
    coord_x, coord_y = coord_x.ravel(order='C'), coord_y.ravel(order='C')
    scat_ax = ax.scatter(coord_x, coord_y, c=(arr_2D.ravel(order='C')-np.min(arr_2D))/(np.max(arr_2D)-np.min(arr_2D)), cmap=CMAP, marker='o', s=pt_size, alpha=alpha)
    cbar = plt.colorbar(scat_ax, shrink=CB_SHRINK, aspect=20.0, pad=CB_PAD)
    cbar.ax.tick_params(labelsize=0.65*title_size)
    # ax.set_aspect(1.0)
    # ax.set_xlabel('$X$', size=0.75*title_size, labelpad=LAB_PAD)
    # ax.set_ylabel('$Y$', size=0.75*title_size, labelpad=LAB_PAD)
    # ax.xaxis.set_tick_params(labelsize=0.65*title_size)
    # ax.yaxis.set_tick_params(labelsize=0.65*title_size)
    # # Set positions of axis tick labels: https://stackoverflow.com/a/17426515/4307919
    # y_len, x_len = arr_img.shape
    # ax.set_xticks(list(range(0,y_len,y_len//TICK_NUM)))
    # ax.set_yticks(list(range(0,x_len,x_len//TICK_NUM)))
    # if title!='':
    #     ax.set_title(title, size=title_size)
    Set_Axis_Prop(ax, ['$X$','$Y$'], labelsize=title_size, labelpad=label_pad, title=title)
    if save_sep:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name), 
                    bbox_inches=extent.expanded(1.4, 1.4))


def PlotSpatial3DHeatMap(ax, arr_2D, fig=None, FLAGS=None, vmin=None, vmax=None, cmap=CMAP, 
                         title='', fig_name='', title_size=LAB_SIZE, label_pad=LAB_PAD, upperleft_corner=(0,0), 
                         config_prop=True, thre_flag=False, thre_kwargs={}, arr_thres=None, z_lim=None, axis_tick_flag=[True]*3, scatter_plot=False, save_sep=False):
    X, Y = np.meshgrid(upperleft_corner[0]+np.arange(arr_2D.shape[1]), upperleft_corner[1]+np.arange(arr_2D.shape[0]))

    # Plot the surface.
    if scatter_plot:
        surf = ax.scatter(X.reshape((-1,)), Y.reshape((-1,)), arr_2D.astype(np.float).reshape((-1,)), 
                          c=arr_2D.astype(np.float).reshape((-1,)), cmap=cmap, vmin=0, vmax=2, 
                          marker=marker[0], s=PT_SIZE)
    else:
        surf = ax.plot_surface(X, Y, arr_2D.astype(np.float), cmap=cmap,
                            vmin=vmin, vmax=vmax,
                            linewidth=0, antialiased=False)
    if thre_flag and arr_thres is not None:
        ls_fc = ['g', 'y']
        for idx in range(len(arr_thres)):
            bd = patches.Rectangle(upperleft_corner,arr_2D.shape[1],arr_2D.shape[0],facecolor=ls_fc[idx],**thre_kwargs)
            ax.add_patch(bd)
            art3d.pathpatch_2d_to_3d(bd, z=arr_thres[idx], zdir="z")

    if config_prop:
        if not scatter_plot:
            # Add a color bar which maps values to colors.
            cbar = plt.colorbar(surf, shrink=CB_SHRINK, aspect=20.0, pad=CB_PAD)
            cbar.ax.tick_params(labelsize=0.65*title_size)
        
        # def axisEqual3D(ax):
        #     """ Handle the incompatability of matplotlib3.
        #     https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio/12371373 """
        #     extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        #     sz = extents[:,1] - extents[:,0]
        #     centers = np.mean(extents, axis=1)
        #     maxsize = max(abs(sz))
        #     r = maxsize/2
        #     for ctr, dim in zip(centers, 'xyz'):
        #         getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

        # # ax.set_aspect(1.0)
        # axisEqual3D(ax)

        # scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        # ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
        
        # https://stackoverflow.com/questions/10326371/setting-aspect-ratio-of-3d-plot/19026019
        # ax.pbaspect = [1.0, 1.0, 0.2]
        # ax.set_xlabel('$X$', size=0.75*title_size, labelpad=LAB_PAD)
        # ax.set_ylabel('$Y$', size=0.75*title_size, labelpad=LAB_PAD)
        # ax.set_zlabel('$Z$', size=0.75*title_size, labelpad=LAB_PAD)
        # y_len, x_len = arr_2D.shape
        # ax.set_xticks(list(range(*ax.get_xlim(),y_len//TICK_NUM)))
        # ax.set_yticks(list(range(0,x_len,x_len//TICK_NUM)))
        # ax.set_zticks(list(range(0,x_len,x_len//TICK_NUM)))
        # ax.xaxis.set_tick_params(labelsize=0.65*title_size)
        # ax.yaxis.set_tick_params(labelsize=0.65*title_size)
        # ax.zaxis.set_tick_params(labelsize=0.65*title_size)
        # ax.set_title(title, size=title_size)
        Set_Axis_Prop(ax, ['$X$','$Y$','$Z$'], labelsize=title_size, labelpad=label_pad, title=title, axis_tick_flag=axis_tick_flag)
        if z_lim is not None:
            ax.set_zlim(*z_lim)
        if save_sep:
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(FLAGS.training_res_folder, fig_name), 
                        bbox_inches=extent.expanded(1.4, 1.4))


def PlotSaveSpatialHeatMap(arr_2D, folder_path, fig_name, cmap=CMAP, title=''):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.figure(num=None, figsize=(8, 0.7*2*ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    PlotSpatialHeatMap(ax, arr_2D, cmap=cmap, title=title)
    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.close()

    # plt.plot(np.arange(len(ewmaI)), np.repeat(ucl, len(ewmaI)))
    # plt.plot(np.arange(len(ewmaI)), np.repeat(lcl, len(ewmaI)))
    # plt.plot(np.arange(len(ewmaI)), np.repeat(0, len(ewmaI)), 'k')

def PlotShowSpatialHeatMap(arr_2D, title=""):
    plt.figure(num=None, figsize=(8, 0.7*2*ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    PlotSpatialHeatMap(ax, arr_2D, title=title)
    plt.show()
    # plt.close()

def PlotSaveSpatial3DHeatMap(arr_2D, folder_path, fig_name, title=''):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.figure(num=None, figsize=(8, 0.7*2*ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    ax = plt.subplot(111, projection='3d')
    PlotSpatial3DHeatMap(ax, arr_2D, title=title)
    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    plt.close()


# def HotellingT2Single(
#         score,
#         time_stamp,
#         gamma,
#         alarm_level,
#         nugget,
#         fig_name,
#         FLAGS,
#         time_step=1,
#         date_index_flag=False):
#     """Calculate and plot hotelling control chart."""
#     # Phase I data
#     N = score.shape[0]
#     Sinv = InversedCov(score, nugget)  # nnet needs 0.15
#     mu = np.mean(score, axis=0)
#
#     # gamma = 0.001
#     score_ewma =  Scores_ewma(
#         score, gamma, np.mean(score, 0))  # starting point
#     # score_ewma =  Scores_ewma(score, gamma, np.zeros(Sinv.shape[0])) #
#     # starting point
#     Sinv2 = Sinv * (2 - gamma) / gamma
#     t2ewma = np.zeros((N, ))
#     for t in range(N):
#         # Statistical quality control-(11.32)
#         t2ewma[t] = HotellingT2(
#             score_ewma[ t, :], mu, Sinv2 / (1 - (1 - gamma)**(2 * (1 + t))))
#
#     # Control Chart
#     # alarm_level = 99.99
#     offset = OFFSET
#     PlotHotellingT2Single(t2ewma[offset:],
#                           alarm_level,
#                           time_stamp[offset:],
#                           FLAGS.training_res_folder,
#                           fig_name,
#                           time_step,
#                           date_index_flag)

def cal_ewma_t2_PI_PII(
        offset,
        alarm_level,
        score_PI,
        score_PII,
        gamma,
        nugget,
        eff_wind_len_factor,
        mu_train, Sinv_train,
        start_PI):
    # Phase-I
    ext_len, start_PII, _, t2ewmaI = EwmaT2PI(score_PI, mu_train, Sinv_train,
        gamma, eff_wind_len_factor, nugget, start_PI)
    t2ewmaI = t2ewmaI[offset+ext_len:]

    # Phase-II
    _, t2ewmaII = EwmaT2PII(score_PII, mu_train, Sinv_train, gamma, start_PII)

    return (t2ewmaI, t2ewmaII)

def cal_ewma_PI_PII(
        offset,
        alarm_level,
        comp_PI,
        comp_PII,
        gamma,
        nugget,
        eff_wind_len_factor,
        start_PI):
    # Phase-I
    ext_len, start_PII, ewma_comp_PI = EwmaPI(comp_PI, gamma, eff_wind_len_factor, start_PI)
    ewma_comp_PI = ewma_comp_PI[offset+ext_len:]

    # Phase-II
    ewma_comp_PII = EwmaPII(comp_PII, gamma, start_PII)

    return (ewma_comp_PI, ewma_comp_PII)

def find_ewma_t2_statistics_PI_PII(
        offset,
        alarm_level,
        score_PI,
        score_PII,
        gamma,
        nugget,
        eff_wind_len_factor,
        mu_train, Sinv_train,
        start_score_PI,
        ucl_score_ls, fp_score_PI_ls, rl_fp_score_PI_ls, len_after_detect_score_PI_ls,
        signal_ratio_score_PI_ls, flag_rl_score_PI_ls,
        fp_score_PII_ls, rl_fp_score_PII_ls, len_after_detect_score_PII_ls,
        signal_ratio_score_PII_ls, flag_rl_score_PII_ls):
    # Phase-I statistics for simulation
    (ewma_score_PI, start_score_PII, ucl_score,
     fp_score_PI, rl_fp_score_PI, len_after_detect_score_PI,
     signal_ratio_score_PI, flag_rl_score_PI) = calEwmaT2StatisticsPI(
        offset, alarm_level, score_PI,
        mu_train, Sinv_train, gamma,
        eff_wind_len_factor, nugget, start_score_PI)

    ucl_score_ls.append(ucl_score)
    fp_score_PI_ls.append(fp_score_PI)
    rl_fp_score_PI_ls.append(rl_fp_score_PI)
    len_after_detect_score_PI_ls.append(len_after_detect_score_PI)
    signal_ratio_score_PI_ls.append(signal_ratio_score_PI)
    flag_rl_score_PI_ls.append(flag_rl_score_PI)

    # Phase-II statistics for simulation
    (ewma_score_PII, fp_score_PII, rl_fp_score_PII,
     len_after_detect_score_PII, signal_ratio_score_PII,
     flag_rl_score_PII) = calEwmaT2StatisticsPII(
        score_PII, mu_train, Sinv_train,
        gamma, start_score_PII, ucl_score)

    fp_score_PII_ls.append(fp_score_PII)
    rl_fp_score_PII_ls.append(rl_fp_score_PII)
    len_after_detect_score_PII_ls.append(len_after_detect_score_PII)
    signal_ratio_score_PII_ls.append(signal_ratio_score_PII)
    flag_rl_score_PII_ls.append(flag_rl_score_PII)

    return ewma_score_PI, ewma_score_PII


def find_ewma_statistics_PI_PII(
        offset,
        alarm_level,
        comp_PI,
        comp_PII,
        gamma,
        nugget,
        eff_wind_len_factor,
        start_comp_PI,
        lcl_comp_ls, ucl_comp_ls, fp_comp_PI_ls,
        fp_comp_l_PI_ls, fp_comp_u_PI_ls, rl_fp_comp_PI_ls,
        len_after_detect_comp_PI_ls, signal_ratio_comp_PI_ls, flag_rl_comp_PI_ls,
        fp_comp_PII_ls, fp_comp_l_PII_ls, fp_comp_u_PII_ls,
        rl_fp_comp_PII_ls, len_after_detect_comp_PII_ls,
        signal_ratio_comp_PII_ls, flag_rl_comp_PII_ls):
    # Phase-I statistics for simulation
    (ewma_comp_PI, start_comp_PII, lcl_comp, ucl_comp, fp_comp_PI,
     fp_comp_l_PI, fp_comp_u_PI, rl_fp_comp_PI, len_after_detect_comp_PI,
     signal_ratio_comp_PI, flag_rl_comp_PI) = calEwmaStatisticsPI(
        offset, alarm_level, comp_PI,
        gamma, eff_wind_len_factor, start_comp_PI)

    lcl_comp_ls.append(lcl_comp)
    ucl_comp_ls.append(ucl_comp)
    fp_comp_PI_ls.append(fp_comp_PI)
    fp_comp_l_PI_ls.append(fp_comp_l_PI)
    fp_comp_u_PI_ls.append(fp_comp_u_PI)
    rl_fp_comp_PI_ls.append(rl_fp_comp_PI)
    len_after_detect_comp_PI_ls.append(len_after_detect_comp_PI)
    signal_ratio_comp_PI_ls.append(signal_ratio_comp_PI)
    flag_rl_comp_PI_ls.append(flag_rl_comp_PI)

    # Phase-II statistics for simulation
    (ewma_comp_PII, fp_comp_PII, fp_comp_l_PII, fp_comp_u_PII,
    rl_fp_comp_PII, len_after_detect_comp_PII, signal_ratio_comp_PII,
    flag_rl_comp_PII) = calEwmaStatisticsPII(
      comp_PII, gamma, start_comp_PII, lcl_comp, ucl_comp)

    fp_comp_PII_ls.append(fp_comp_PII)
    fp_comp_l_PII_ls.append(fp_comp_l_PII)
    fp_comp_u_PII_ls.append(fp_comp_u_PII)
    rl_fp_comp_PII_ls.append(rl_fp_comp_PII)
    len_after_detect_comp_PII_ls.append(len_after_detect_comp_PII)
    signal_ratio_comp_PII_ls.append(signal_ratio_comp_PII)
    flag_rl_comp_PII_ls.append(flag_rl_comp_PII)

    return ewma_comp_PI, ewma_comp_PII


# def HotellingCC(
#         score_PI,
#         score_PII,
#         time_stamp,
#         idx_PII,
#         gamma,
#         alarm_level,
#         fig_name,
#         FLAGS,
#         time_step=1):
#     """ Calculate and plot hotelling control chart. A single plot with a single line."""
#     # # Phase I data
#     # N_PI = score_PI.shape[0]
#     # # Sinv = InversedCov(score_PI, 0) # nnet needs 0.15
#     # mu = np.mean(score_PI, axis=0)
#     #
#     # # gamma = 0.001
#     # score_ewma =  Scores_ewma(
#     #     score_PI, gamma, np.mean(score_PI, 0))  # starting point
#     #
#     # # Sinv2 = Sinv * (2 - gamma) / gamma
#     # ewmaI = np.zeros((N_PI, ))
#     # for t in range(N_PI):
#     #     # Statistical quality control-(11.32)
#     #     # * (Sinv2 / (1 - (1 - gamma)**(2 * (1 + t))))**0.5
#     #     ewmaI[t] = (score_ewma[ t, :] - mu)
#     #
#     # # Phase II data:
#     # N_PII = score_PII.shape[0]
#     # # scoreII = score(model, X_test, y_test, reg_val)
#     # score_ewma = Scores_ewma(score_PII, gamma, score_ewma[ -1, :])
#     # ewmaII = np.zeros((N_PII,))
#     # for t in range(N_PII):
#     #     ewmaII[t] = (score_ewma[t, :] - mu)  # * Sinv2**0.5

#     # Control Chart
#     # alarm_level = 99.99
#     offset = OFFSET
#     # EWMA-T2
#     # Phase-I
#     ext_len, start_score_PII, _, t2ewmaI = EwmaT2PI(score_PI, mu_train, Sinv_train,
#         gamma, eff_wind_len_factor, nugget, start_score_PI)
#     t2ewmaI = t2ewmaI[offset+ext_len:]

#     # Phase-II
#     _, t2ewmaII = EwmaT2PII(score_PII, mu_train, Sinv_train, gamma, start_score_PII)

#     PlotHotellingCC(ewmaI[offset:],
#                     ewmaII,
#                     alarm_level,
#                     time_stamp[offset:],
#                     idx_PII - offset,
#                     os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder),
#                     fig_name,
#                     time_step)


def PlotHotellingCC(ewmaI, ewmaII, alarm_level, time_stamp, idx_PII,
                    folder_path, fig_name, time_step):
    """Based on Phase-I&-II ewma hotelling time_stamp to plot control chart."""
    ucl = np.percentile(ewmaI, (100 + alarm_level) / 2)
    lcl = np.percentile(ewmaI, (100 - alarm_level) / 2)
    plt.figure(num=None, figsize=(8, 2*ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    plt.subplot(211)
    plt.plot(np.arange(len(ewmaI)), ewmaI)
    plt.plot(np.arange(len(ewmaI)), np.repeat(ucl, len(ewmaI)))
    plt.plot(np.arange(len(ewmaI)), np.repeat(lcl, len(ewmaI)))
    if lcl < 0 < ucl:
        plt.plot(np.arange(len(ewmaI)), np.repeat(0, len(ewmaI)), 'k')
    tmp = np.min(ewmaI)
    for i in np.arange(time_stamp[0], time_stamp[idx_PII - 1] + 1, time_step):
        j = np.argmax(time_stamp >= i)
        plt.text(j, tmp, time_stamp[j], rotation=45)
    plt.subplot(212)
    plt.plot(np.arange(len(ewmaII)), ewmaII)
    plt.plot(np.arange(len(ewmaII)), np.repeat(ucl, len(ewmaII)))
    plt.plot(np.arange(len(ewmaII)), np.repeat(lcl, len(ewmaII)))
    if lcl < 0 < ucl:
        plt.plot(np.arange(len(ewmaII)), np.repeat(0, len(ewmaII)), 'k')
    tmp = np.min(ewmaII)
    for i in np.arange(time_stamp[idx_PII], max(time_stamp) + 1, time_step):
        j = np.argmax(time_stamp >= i)
        plt.text(j - idx_PII, tmp, time_stamp[j], rotation=45)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    # plt.show()
    plt.close()



def HotellingT2CC(
        score_PI,
        score_PII,
        time_stamp,
        N_PIIs,
        gamma,
        alarm_level,
        nugget,
        fig_name,
        FLAGS,
        comp_score_offset,
        mu_train, Sinv_train,
        eff_wind_len_factor,
        start_score_PI,
        xlabel_name='Phase-I&-II Observation Index',
        time_step=1,
        thre_flag=True,
        date_index_flag=False,
        to_decode=False):
    """ Calculate and plot hotelling control chart. A single plot with a single line."""
    # # Phase I data
    # N_PI = score_PI.shape[0]
    # Sinv = InversedCov(score_PI, nugget)  # nnet needs 0.15
    # mu = np.mean(score_PI, axis=0)
    #
    # # gamma = 0.001
    # score_ewma =  Scores_ewma(
    #     score_PI, gamma, np.mean(score_PI, 0))  # starting point
    # # score_ewma =  Scores_ewma(score, gamma, np.zeros(Sinv.shape[0])) #
    # # starting point
    # Sinv2 = Sinv * (2 - gamma) / gamma
    # t2ewmaI = np.zeros((N_PI, ))
    # for t in range(N_PI):
    #     # Statistical quality control-(11.32)
    #     t2ewmaI[t] = HotellingT2(
    #         score_ewma[ t, :], mu, Sinv2 / (1 - (1 - gamma)**(2 * (1 + t))))
    #
    # # Phase II data:
    # N_PII = score_PII.shape[0]
    # # scoreII = score(model, X_test, y_test, reg_val)
    # score_ewma = Scores_ewma(score_PII, gamma, score_ewma[ -1, :])
    # t2ewmaII = np.zeros((N_PII,))
    # for t in range(N_PII):
    #     t2ewmaII[t] = HotellingT2(score_ewma[t, :], mu, Sinv2)

    # Control Chart
    # alarm_level = 99.99
    offset = OFFSET
    # EWMA-T2
    # Phase-I
    ext_len, start_score_PII, _, t2ewmaI = EwmaT2PI(score_PI, mu_train, Sinv_train,
        gamma, eff_wind_len_factor, nugget, start_score_PI)
    t2ewmaI = t2ewmaI[offset+ext_len:]

    # Phase-II
    _, t2ewmaII = EwmaT2PII(score_PII, mu_train, Sinv_train, gamma, start_score_PII)

    PlotHotellingT2CC(t2ewmaI,
                      t2ewmaII,
                      alarm_level,
                      time_stamp,
                      N_PIIs,
                      os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder),
                      fig_name,
                      time_step,
                      xlabel_name=xlabel_name,
                      comp_score_offset=comp_score_offset,
                      sign_factor=1,
                      thre_flag=thre_flag,
                      date_index_flag=date_index_flag, 
                      to_decode=to_decode)

    # PlotHotellingT2CC(t2ewmaI,
    #                   t2ewmaII,
    #                   alarm_level,
    #                   time_stamp,
    #                   N_PIIs,
    #                   os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder),
    #                   fig_name,
    #                   time_step,
    #                   comp_score_offset,
    #                   -1,
    #                   date_index_flag)


def PlotHotellingT2CC(
        t2ewmaI,
        t2ewmaII,
        alarm_level,
        time_stamp,
        N_PIIs,
        folder_path,
        fig_name,
        time_step,
        xlabel_name = "Phase-I&-II Observation Index",
        comp_score_offset=0,
        sign_factor=1,
        thre_flag=True,
        date_index_flag=False, 
        to_decode=False):
    """Based on Phase-I&-II ewma hotelling time_stamp to plot control chart."""
    ucl5 = np.percentile(t2ewmaI, alarm_level)

    label_size = LAB_SIZE
    # line_width = 1.5
    # marker_size = 5
    plt.figure(num=None, figsize=(10, 0.8*ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE)
    ax1 = plt.subplot(111)
    ylabel_name = "MEWMA Multivariate\nScore Vector"
    thre_ls = [ucl5, 0]
    thre_style_ls = ['b--', 'k--']
    PlotMonStatOneAxisPIPII(ax1, t2ewmaI, t2ewmaII, xlabel_name, ylabel_name,
            label_size, thre_ls, thre_style_ls, time_stamp, N_PIIs, time_step,
            comp_score_offset=comp_score_offset, sign_factor=sign_factor, thre_flag=thre_flag, 
            date_index_flag=date_index_flag, to_decode=to_decode)

    # ax1 = plt.subplot(211)
    # ax1.set_xlabel('Phase-I Observation Index', size=label_size)
    # ax1.set_ylabel(
    #     'MEWMA Multivariate\nScore Vector',
    #     color='k',
    #     size=label_size)
    # ax1.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
    # ax1.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    # ax1.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    #
    # ax1.plot(np.arange(len(t2ewmaI)), t2ewmaI)
    # ax1.plot(np.arange(len(t2ewmaI)), np.repeat(ucl5, len(t2ewmaI)))
    # ax1.plot(np.arange(len(t2ewmaI)), np.repeat(0, len(t2ewmaI)), 'k', lw=0.5)
    # tmp = np.min(t2ewmaI)
    # for i in np.arange(time_stamp[0], time_stamp[idx_PII - 1] + 1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax1.text(j,
    #              ax1.get_ylim()[0] + 0.1 * (ax1.get_ylim()
    #                                         [1] - ax1.get_ylim()[0]),
    #              MonIdx2Date(time_stamp[j],
    #                          date_index_flag),
    #              rotation=45)
    #
    # ax1 = plt.subplot(212)
    # ax1.set_xlabel('Phase-II Observation Index', size=label_size)
    # ax1.set_ylabel(
    #     'MEWMA Multivariate\nScore Vector',
    #     color='k',
    #     size=label_size)
    # ax1.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
    # ax1.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    # ax1.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    #
    # ax1.plot(np.arange(len(t2ewmaII)), t2ewmaII)
    # ax1.plot(np.arange(len(t2ewmaII)), np.repeat(ucl5, len(t2ewmaII)), 'k')
    # ax1.plot(
    #     np.arange(
    #         len(t2ewmaII)), np.repeat(
    #         0, len(t2ewmaII)), 'k', lw=0.5)
    # tmp = np.min(t2ewmaII)
    # for i in np.arange(time_stamp[idx_PII], max(time_stamp) + 1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax1.text(j - idx_PII,
    #              ax1.get_ylim()[0] + 0.1 * (ax1.get_ylim()
    #                                         [1] - ax1.get_ylim()[0]),
    #              MonIdx2Date(time_stamp[j],
    #                          date_index_flag),
    #              rotation=45)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    # plt.show()
    plt.close()



def HotellingT2CC_Split(
        score_PI,
        score_PII,
        time_stamp,
        N_PIIs,
        gamma,
        alarm_level,
        nugget,
        fig_name,
        FLAGS,
        comp_score_offset,
        mu_train, Sinv_train,
        eff_wind_len_factor,
        start_score_PI,
        time_step=1,
        thre_flag=True,
        date_index_flag=False):
    """ Calculate and plot hotelling control chart. A single plot with a single line,
        but split plots for Phase-I&-II.
    """
    # # Phase I data
    # N_PI = score_PI.shape[0]
    # Sinv = InversedCov(score_PI, nugget)  # nnet needs 0.15
    # mu = np.mean(score_PI, axis=0)
    #
    # # gamma = 0.001
    # score_ewma =  Scores_ewma(
    #     score_PI, gamma, np.mean(score_PI, 0))  # starting point
    # # score_ewma =  Scores_ewma(score, gamma, np.zeros(Sinv.shape[0])) #
    # # starting point
    # Sinv2 = Sinv * (2 - gamma) / gamma
    # t2ewmaI = np.zeros((N_PI, ))
    # for t in range(N_PI):
    #     # Statistical quality control-(11.32)
    #     t2ewmaI[t] = HotellingT2(
    #         score_ewma[ t, :], mu, Sinv2 / (1 - (1 - gamma)**(2 * (1 + t))))
    #
    # # Phase II data:
    # N_PII = score_PII.shape[0]
    # # scoreII = score(model, X_test, y_test, reg_val)
    # score_ewma = Scores_ewma(score_PII, gamma, score_ewma[ -1, :])
    # t2ewmaII = np.zeros((N_PII,))
    # for t in range(N_PII):
    #     t2ewmaII[t] = HotellingT2(score_ewma[t, :], mu, Sinv2)

    # Control Chart
    # alarm_level = 99.99
    offset = OFFSET
    # EWMA-T2
    # Phase-I
    ext_len, start_score_PII, _, t2ewmaI = EwmaT2PI(score_PI, mu_train, Sinv_train,
        gamma, eff_wind_len_factor, nugget, start_score_PI)
    t2ewmaI = t2ewmaI[offset+ext_len:]

    # Phase-II
    _, t2ewmaII = EwmaT2PII(score_PII, mu_train, Sinv_train, gamma, start_score_PII)

    PlotHotellingT2CC_Split(t2ewmaI,
                      t2ewmaII,
                      alarm_level,
                      time_stamp,
                      N_PIIs,
                      os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder),
                      fig_name,
                      time_step,
                      comp_score_offset=comp_score_offset,
                      sign_factor=1,
                      thre_flag=thre_flag,
                      date_index_flag=date_index_flag)


def PlotHotellingT2CC_Split(
        t2ewmaI,
        t2ewmaII,
        alarm_level,
        time_stamp,
        N_PIIs,
        folder_path,
        fig_name,
        time_step,
        comp_score_offset=0,
        sign_factor=1,
        thre_flag=True,
        date_index_flag=False):
    """ Based on Phase-I&-II ewma hotelling time_stamp to plot control chart."""
    ucl5 = np.percentile(t2ewmaI, alarm_level)

    label_size = LAB_SIZE
    # line_width = 1.5
    # marker_size = 5
    plt.figure(num=None, figsize=(2*ONE_FIG_HEI, 0.6*ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = 0.3)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    # xlabel_name1 = "Phase-I Observation Index"
    # xlabel_name2 = "Phase-II Observation Index"
    # ylabel_name = "MEWMA Multivariate\nScore Vector"
    xlabel_name1 = ""
    xlabel_name2 = ""
    ylabel_name = ""
    thre_ls = [ucl5, 0]
    thre_style_ls = ['b--', 'k--']
    PlotMonStatOneAxisPIPII_Split(ax1, ax2, t2ewmaI, t2ewmaII, xlabel_name1, xlabel_name2,
            ylabel_name, label_size, thre_ls, thre_style_ls, time_stamp, N_PIIs, time_step,
            col_label="", comp_score_offset=comp_score_offset, sign_factor=sign_factor, 
            thre_flag=thre_flag, date_index_flag=date_index_flag)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    # plt.show()
    plt.close()

    fig_name_post = fig_name

    PlotMonStatOneAxisPIPII_Separated(ax1, ax2, t2ewmaI, t2ewmaII, xlabel_name1, xlabel_name2,
            ylabel_name, label_size, thre_ls, thre_style_ls, time_stamp, N_PIIs, time_step, folder_path,
            fig_name_post, col_label="", comp_score_offset=comp_score_offset, sign_factor=sign_factor, 
            thre_flag=thre_flag, date_index_flag=date_index_flag)


def HotellingT2CC_Other_Score_Thre_PII(
        score_PI,
        score_PII,
        comp1_PI,
        comp1_PII,
        comp2_PI,
        comp2_PII,
        thresholds,
        thre_colors,
        time_stamp,
        idx_PII,
        N_PIIs,
        gamma,
        alarm_level,
        nugget,
        fig_name,
        comp1_name,
        comp2_name,
        FLAGS,
        comp_score_offset,
        mu_train, Sinv_train,
        eff_wind_len_factor,
        start_score_PI,
        start_comp1_PI,
        start_comp2_PI,
        time_step=1,
        date_index_flag=False,
        fplot=True):
    """ Calculate and plot hotelling control chart with another ewma plot.
    """
    # Control Chart
    # alarm_level = 99.99
    offset = OFFSET

    # EWMA-T2
    # Phase-I
    ext_len, start_score_PII, _, t2ewmaI = EwmaT2PI(score_PI, mu_train, Sinv_train,
        gamma, eff_wind_len_factor, nugget, start_score_PI)
    t2ewmaI = t2ewmaI[offset+ext_len:]

    # Phase-II
    _, t2ewmaII = EwmaT2PII(score_PII, mu_train, Sinv_train, gamma, start_score_PII)

    # Comp1 statistics
    # Phase-I
    _, start_comp1_PII, ewma_comp1_PI = EwmaPI(comp1_PI, gamma, eff_wind_len_factor, start_comp1_PI)
    ewma_comp1_PI = ewma_comp1_PI[offset+ext_len:]

    # Phase-II
    ewma_comp1_PII = EwmaPII(comp1_PII, gamma, start_comp1_PII)

    # Comp2 statistics
    # Phase-I
    _, start_comp2_PII, ewma_comp2_PI = EwmaPI(comp2_PI, gamma, eff_wind_len_factor, start_comp2_PI)
    ewma_comp2_PI = ewma_comp2_PI[offset+ext_len:]

    # Phase-II
    ewma_comp2_PII = EwmaPII(comp2_PII, gamma, start_comp2_PII)

    # # Phase I data
    # N_PI = score_PI.shape[0]
    # Sinv = InversedCov(score_PI, nugget)  # nnet needs 0.15
    # mu = np.mean(score_PI, axis=0)
    #
    # # gamma = 0.001
    # score_ewma = Scores_ewma(
    #     score_PI, gamma, np.mean(score_PI, 0))  # starting point
    # # score_ewma = Scores_ewma(score, gamma, np.zeros(Sinv.shape[0])) #
    # # starting point
    # Sinv2 = Sinv * (2 - gamma) / gamma
    # t2ewmaI = np.zeros((N_PI, ))
    # for t in range(N_PI):
    #     # Statistical quality control-(11.32)
    #     t2ewmaI[t] = HotellingT2(
    #         score_ewma[t, :], mu, Sinv2 / (1 - (1 - gamma)**(2 * (1 + t))))
    #
    # # Phase II data:
    # N_PII = score_PII.shape[0]
    # # scoreII = score(model, X_test, y_test, reg_val)
    # score_ewmaII = Scores_ewma(score_PII, gamma, score_ewma[-1, :])
    # t2ewmaII = np.zeros((N_PII,))
    # for t in range(N_PII):
    #     t2ewmaII[t] = HotellingT2(score_ewmaII[t, :], mu, Sinv2)

    # # Control Chart
    # # alarm_level = 99.99
    # offset = OFFSET

    if fplot:
        PlotHotellingT2CC_Other_Score_Thre_PII(
            t2ewmaI,
            t2ewmaII,
            ewma_comp1_PI,
            ewma_comp1_PII,
            ewma_comp2_PI,
            ewma_comp2_PII,
            thresholds,
            thre_colors,
            alarm_level,
            time_stamp,
            idx_PII,
            N_PIIs,
            FLAGS.training_res_folder,
            fig_name,
            comp1_name,
            comp2_name,
            comp_score_offset,
            time_step,
            sign_factor = 1,
            date_index_flag = False)

        # PlotHotellingT2CC_Other_Score_Thre_PII(
        #     t2ewmaI,
        #     t2ewmaII,
        #     comp_score1_PI,
        #     comp_score1_PII,
        #     comp_score2_PI,
        #     comp_score2_PII,
        #     thresholds,
        #     thre_colors,
        #     alarm_level,
        #     time_stamp,
        #     idx_PII,
        #     N_PIIs,
        #     folder_path,
        #     fig_name,
        #     comp_name1,
        #     comp_name2,
        #     comp_score_offset,
        #     time_step,
        #     sign_factor = -1,
        #     date_index_flag = False)


def PlotMonStatOneAxisPII(
        ax,
        statII,
        xlabel_name,
        ylabel_name,
        label_size,
        thre_ls,
        thre_style_ls,
        time_stamp,
        N_PIIs,
        time_step,
        sign_factor=1,
        comp_score_offset=-1,
        date_index_flag=False,
        to_decode=False,
        xlabel_flag=True,
        ylims=None,
        xlims=None):
    """Plot one single axis for a big plot."""
    if xlabel_flag:
        ax.get_xaxis().set_label_coords(0.5, XLAB_YPOS)
        ax.set_xlabel(xlabel_name, size=1.5*label_size)
    ax.set_ylabel(ylabel_name, color='b',size=label_size)
    ax.get_yaxis().set_label_coords(YLAB_XPOS,0.5)
    n_sample_PII = len(statII)
    if comp_score_offset>0:
        yrange = max(statII[comp_score_offset:]) - min(statII[comp_score_offset:])
        ymax = max(statII[comp_score_offset:])
        ymin = min(statII[comp_score_offset:])
    else:
        yrange = max(statII) - min(statII)
        ymax = max(statII)
        ymin = min(statII)
    if sign_factor == -1:
        ax.set_ylim([ymax + yrange / 10,
                     ymin - yrange / 10])
    else:
        ax.set_ylim([ymin - yrange / 10,
                     ymax + yrange / 10])
    ax.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    ax.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

    ax.plot(np.arange(n_sample_PII), statII, 'b')
    for thre, sty in zip(thre_ls, thre_style_ls):
        ax.plot(np.arange(n_sample_PII),
                np.repeat(thre, n_sample_PII), sty, lw=1)
    for idx in np.cumsum(N_PIIs[:-1]):
        ax.axvline(x=idx, color='g') # Separation b/w different stages of concept drift
    if time_stamp.shape[0]>0 and date_index_flag:
        set_xticks_xticklabels(ax, time_stamp, time_step, label_size=label_size, rotation=ROTATION, to_decode=to_decode)
        
    # lines1, labels1 = ax.get_legend_handles_labels()
    # ax.legend(lines1, labels1, loc='upper left',
    #           bbox_to_anchor=(0,0.95), prop={'size':label_size}, ncol=2)


def PlotHotellingT2CC_Other_Score_Thre_PII(
        t2ewmaI,
        t2ewmaII,
        comp_score1_PI,
        comp_score1_PII,
        comp_score2_PI,
        comp_score2_PII,
        thresholds,
        thre_colors,
        alarm_level,
        time_stamp,
        idx_PII,
        N_PIIs,
        folder_path,
        fig_name,
        comp_name1,
        comp_name2,
        comp_score_offset,
        time_step,
        sign_factor = 1,
        date_index_flag = False):
    ucl5 = np.percentile(t2ewmaI, alarm_level)
    ucl_comp1 = np.percentile(
        comp_score1_PI[comp_score_offset:], (100.0 + alarm_level) / 2)
    lcl_comp1 = np.percentile(
        comp_score1_PI[comp_score_offset:], (100.0 - alarm_level) / 2)
    ucl_comp2 = np.percentile(
        comp_score2_PI[comp_score_offset:], (100.0 + alarm_level) / 2)
    lcl_comp2 = np.percentile(
        comp_score2_PI[comp_score_offset:], (100.0 - alarm_level) / 2)

    label_size = LAB_SIZE
    # line_width = 1.5
    # marker_size = 5
    plt.figure(num=None, figsize=(10, 3*0.7*ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace=0.3)
    ax1 = plt.subplot(311)
    xlabel_name = "Phase-II Observation Index"
    ylabel_name = "MEWMA Multivariate\nScore Vector"
    thre_ls = [ucl5, 0]
    thre_style_ls = ['b--', 'k--']
    PlotMonStatOneAxisPII(ax1, t2ewmaII, xlabel_name, ylabel_name, label_size,
            thre_ls, thre_style_ls, time_stamp[idx_PII:], N_PIIs, time_step, sign_factor=sign_factor)
    # ax1.set_xlabel('Phase-I&-II Observation Index', size=label_size)
    # ax1.set_ylabel(
    #     'MEWMA Multivariate\nScore Vector',
    #     color='b',
    #     size=label_size)
    # ax1.get_yaxis().set_label_coords(-0.06, 0.5)
    # t2ewmaI_II = np.hstack((t2ewmaI, t2ewmaII))
    # yrange = max(t2ewmaI_II) - min(t2ewmaI_II)
    # if sign_factor == -1:
    #     ax1.set_ylim([max(t2ewmaI_II) + yrange / 10,
    #                   min(t2ewmaI_II) - yrange / 10])
    # else:
    #     ax1.set_ylim([min(t2ewmaI_II) - yrange / 10,
    #                   max(t2ewmaI_II) + yrange / 10])
    # ax1.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    # ax1.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    #
    # ax1.plot(np.arange(len(t2ewmaI_II)), t2ewmaI_II, 'b')
    # ax1.plot(
    #     np.arange(
    #         len(t2ewmaI_II)), np.repeat(
    #         ucl5, len(t2ewmaI_II)), 'b--')
    # ax1.plot(
    #     np.arange(
    #         len(t2ewmaI_II)), np.repeat(
    #         0, len(t2ewmaI_II)), 'k--', lw=0.5)
    # ax1.axvline(x=len(t2ewmaI))
    # for idx_PII in len(t2ewmaI) + np.cumsum(N_PIIs[:-1]):
    #     ax1.axvline(x=idx_PII, color='g')
    # for i in np.arange(time_stamp[0], time_stamp[idx_PII - 1] + 1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax1.text(j,
    #              ax1.get_ylim()[0] + 0.1 * (ax1.get_ylim()
    #                                         [1] - ax1.get_ylim()[0]),
    #              MonIdx2Date(time_stamp[j],
    #                          date_index_flag),
    #              rotation=45,
    #              fontsize=label_size * AX_LAB_SCALE)

    # ax2 = ax1.twinx()
    ax2 = plt.subplot(312)
    ylabel_name = comp_name1
    thre_ls = [ucl_comp1, lcl_comp1]
    thre_style_ls = ['k--', 'k--']
    PlotMonStatOneAxisPII(ax2, comp_score1_PII, xlabel_name, ylabel_name, label_size,
            thre_ls, thre_style_ls, time_stamp[idx_PII:], N_PIIs, time_step, sign_factor=sign_factor)

    # ax2.set_xlabel('Phase-I&-II Observation Index', size=label_size)
    # ax2.set_ylabel(comp_name, color='k', size=label_size)
    # comp_score_PI_PII = np.hstack((comp_score_PI, comp_score_PII))
    # margin = (max(comp_score1_PI_PII[comp_score_offset:]) -
    #     min(comp_score1_PI_PII[comp_score_offset:])) / 10
    # if sign_factor == -1:
    #     ax2.set_ylim([max(comp_score_PI_PII[comp_score_offset:]) + margin,
    #                   min(comp_score_PI_PII[comp_score_offset:]) - margin])
    # else:
    #     ax2.set_ylim([min(comp_score_PI_PII[comp_score_offset:]) - margin,
    #                   max(comp_score_PI_PII[comp_score_offset:]) + margin])
    # ax2.get_yaxis().set_label_coords(-0.06, 0.5)
    # ax2.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    # ax2.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    #
    # ax2.plot(np.arange(comp_score_offset, len(comp_score_PI_PII)),
    #          comp_score_PI_PII[comp_score_offset:], 'k')
    # ax2.plot(
    #     np.arange(
    #         len(comp_score_PI_PII)), np.repeat(
    #         thresholds[0], len(comp_score_PI_PII)), thre_colors[0])
    # ax2.plot(
    #     np.arange(
    #         len(comp_score_PI_PII)), np.repeat(
    #         thresholds[1], len(comp_score_PI_PII)), thre_colors[1])
    # ax2.axvline(x=len(comp_score_PI))
    # for idx_PII in len(comp_score_PI) + np.cumsum(N_PIIs[:-1]):
    #     ax2.axvline(x=idx_PII, color='g')
    # for i in np.arange(time_stamp[0], time_stamp[idx_PII - 1] + 1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax2.text(j,
    #              (ax2.get_ylim()[0] + 0.1 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])),
    #              MonIdx2Date(time_stamp[j],
    #                          date_index_flag),
    #              rotation=45,
    #              fontsize=label_size * AX_LAB_SCALE)

    ax3 = plt.subplot(313)
    ylabel_name = comp_name2
    thre_ls = [ucl_comp2, lcl_comp2]
    thre_style_ls = ['k--', 'k--']
    PlotMonStatOneAxisPII(ax3, comp_score2_PII, xlabel_name, ylabel_name, label_size,
            thre_ls, thre_style_ls, time_stamp[idx_PII:], N_PIIs, time_step, sign_factor=sign_factor)

    # ax1 = plt.subplot(222)
    # ax1.set_xlabel('Phase-II Observation Index',size=label_size)
    # ax1.set_ylabel('MEWMA Multivariate\nScore Vector', color='b',size=label_size)
    # ax1.get_yaxis().set_label_coords(-0.12,0.5)
    # ax1.xaxis.set_tick_params(labelsize=label_size*0.5)
    # ax1.yaxis.set_tick_params(labelsize=label_size*0.5)
    #
    # # ax1.plot(np.arange(len(t2ewmaII)), t2ewmaII, 'b', label='Score Function\n(Hotelling $T^2$ of EWMA)')
    # ax1.plot(np.arange(len(t2ewmaII)), t2ewmaII, 'b')
    # ax1.plot(np.arange(len(t2ewmaII)), np.repeat(ucl5, len(t2ewmaII)), 'b--')
    # ax1.plot(np.arange(len(t2ewmaII)), np.repeat(0, len(t2ewmaII)), 'k--', lw=0.5)
    # tmp = np.min(t2ewmaII)
    # for i in np.arange(time_stamp[idx_PII], max(time_stamp)+1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax1.text(j - idx_PII, ax1.get_ylim()[0]+0.1*(ax1.get_ylim()[1]-ax1.get_ylim()[0]),
    #              MonIdx2Date(time_stamp[j], date_index_flag), rotation=45, fontsize=label_size*0.5)
    # # lines1, labels1 = ax1.get_legend_handles_labels()
    # # ax1.legend(lines1, labels1, loc='upper left',
    # #            bbox_to_anchor=(0,0.95), prop={'size':label_size}, ncol=2)
    #
    # # ax2 = ax1.twinx()
    # ax2 = plt.subplot(224)
    # ax2.set_xlabel('Phase-II Observation Index',size=label_size)
    # ax2.set_ylabel(comp_name, color='k',size=label_size)
    # if sign_factor==-1:
    #     ax2.set_ylim([max(comp_score_PII)+(max(comp_score_PII)-min(comp_score_PII))/10,
    #                  min(comp_score_PII)-(max(comp_score_PII)-min(comp_score_PII))/10])
    # else:
    #     ax2.set_ylim([min(comp_score_PII)-(max(comp_score_PII)-min(comp_score_PII))/10,
    #                  max(comp_score_PII)+(max(comp_score_PII)-min(comp_score_PII))/10])
    # ax2.get_yaxis().set_label_coords(-0.12,0.5)
    # ax2.xaxis.set_tick_params(labelsize=label_size*0.5)
    # ax2.yaxis.set_tick_params(labelsize=label_size*0.5)
    #
    # # ax2.plot(np.arange(len(comp_score_PII)), comp_score_PII, 'k' ,label=comp_name)
    # ax2.plot(np.arange(len(comp_score_PII)), comp_score_PII, 'k')
    # ax2.plot(np.arange(len(comp_score_PII)), np.repeat(thresholds[0], len(comp_score_PII)), thre_colors[0])
    # ax2.plot(np.arange(len(comp_score_PII)), np.repeat(thresholds[1], len(comp_score_PII)), thre_colors[1])
    # tmp = np.min(t2ewmaII)
    # for i in np.arange(time_stamp[idx_PII], max(time_stamp)+1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax2.text(j - idx_PII, ax2.get_ylim()[0]+0.1*(ax2.get_ylim()[1]-ax2.get_ylim()[0]),
    #              MonIdx2Date(time_stamp[j], date_index_flag), rotation=45, fontsize=label_size*0.5)
    # # lines2, labels2 = ax2.get_legend_handles_labels()
    # # # ax2.legend(lines1+lines2, labels1+labels2, loc=0)
    # # ax2.legend(lines2, labels2, loc='upper left',
    # #            bbox_to_anchor=(0,0.95), prop={'size':label_size}, ncol=2)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    # plt.show()
    plt.close()


def HotellingT2CC_Other_Comp_Score(
        score_PI,
        score_PII,
        comp_score_PI,
        comp_score_PII,
        comp_thre_ls,
        comp_thre_sty_ls,
        time_stamp,
        idx_PII,
        N_PIIs,
        gamma,
        alarm_level,
        nugget,
        fig_name,
        comp_name,
        FLAGS,
        comp_score_offset,
        mu_train, Sinv_train,
        eff_wind_len_factor,
        start_score_PI,
        time_step=1,
        thre_flag=True,
        date_index_flag=False,
        to_decode=False,
        fplot=True):
    """ Calculate and plot hotelling control chart with another kind of score
        (not score function) for comparison. The other score can be some metrics.
        There is a possibility that the threshold (control limits) calculation
        is not based on empirical distribution. Here, the control limits of
        other score are passed in as parameters.
    """
    # Control Chart
    # alarm_level = 99.99
    offset = OFFSET

    # EWMA-T2
    # Phase-I
    ext_len, start_score_PII, _, t2ewmaI = EwmaT2PI(score_PI, mu_train, Sinv_train,
        gamma, eff_wind_len_factor, nugget, start_score_PI)
    t2ewmaI = t2ewmaI[offset+ext_len:]

    # Phase-II
    _, t2ewmaII = EwmaT2PII(score_PII, mu_train, Sinv_train, gamma, start_score_PII)

    if fplot:
        PlotHotellingT2CC_Other_Comp_Score(
                t2ewmaI,
                t2ewmaII,
                comp_score_PI,
                comp_score_PII,
                comp_thre_ls,
                comp_thre_sty_ls,
                alarm_level,
                time_stamp,
                idx_PII,
                N_PIIs,
                FLAGS.training_res_folder,
                fig_name,
                comp_name,
                comp_score_offset,
                time_step,
                sign_factor = 1,
                thre_flag = thre_flag,
                date_index_flag = date_index_flag, 
                to_decode=to_decode)

        # PlotHotellingT2CC_Other_Comp_Score(
        #         t2ewmaI,
        #         t2ewmaII,
        #         comp_score_PI,
        #         comp_score_PII,
        #         comp_thre_ls,
        #         comp_thre_sty_ls,
        #         alarm_level,
        #         time_stamp,
        #         idx_PII,
        #         N_PIIs,
        #         FLAGS.training_res_folder,
        #         fig_name,
        #         comp_name,
        #         comp_score_offset,
        #         time_step,
        #         sign_factor = -1,
        #         date_index_flag = False)


def PlotHotellingT2CC_Other_Comp_Score(
        t2ewmaI,
        t2ewmaII,
        comp_score_PI,
        comp_score_PII,
        comp_thre_ls,
        comp_thre_sty_ls,
        alarm_level,
        time_stamp,
        idx_PII,
        N_PIIs,
        folder_path,
        fig_name,
        comp_name,
        comp_score_offset,
        time_step,
        sign_factor = 1,
        thre_flag = True,
        date_index_flag = False, 
        to_decode=False):
    """Based on Phase-I&-II ewma hotelling time_stamp to plot control chart."""
    # print(comp_score1_PI.shape)
    ucl5 = np.percentile(t2ewmaI, alarm_level)
    if not comp_thre_ls:
        ucl_comp = np.percentile(
            comp_score_PI[comp_score_offset:], (100.0 + alarm_level) / 2)
        lcl_comp = np.percentile(
            comp_score_PI[comp_score_offset:], (100.0 - alarm_level) / 2)
        comp_thre_ls = [ucl_comp, lcl_comp]
        comp_thre_sty_ls = ['k--', 'k--']

    label_size = LAB_SIZE
    # line_width = 1.5
    # marker_size = 5
    plt.figure(num=None, figsize=(10, 2*0.8*ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace=0.3)
    ax1 = plt.subplot(211)
    xlabel_name = "Phase-I&-II Observation Index"
    ylabel_name = "MEWMA Multivariate\nScore Vector"
    thre_ls = [ucl5, 0]
    thre_style_ls = ['b--', 'k--']
    PlotMonStatOneAxisPIPII(ax1, t2ewmaI, t2ewmaII, xlabel_name, ylabel_name,
            label_size, thre_ls, thre_style_ls, time_stamp, N_PIIs, time_step,
            comp_score_offset=comp_score_offset,
            sign_factor=sign_factor, thre_flag=thre_flag, 
            date_index_flag=date_index_flag, to_decode=to_decode, xlabel_flag=False)

    ax2 = plt.subplot(212)
    ylabel_name = comp_name
    PlotMonStatOneAxisPIPII(ax2, comp_score_PI, comp_score_PII, xlabel_name,
            ylabel_name, label_size, comp_thre_ls, comp_thre_sty_ls,
            time_stamp, N_PIIs, time_step, comp_score_offset=comp_score_offset,
            sign_factor=sign_factor, thre_flag=thre_flag, 
            date_index_flag=date_index_flag, to_decode=to_decode)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    # plt.show()
    plt.close()


def HotellingT2CC_Other_Score2(
        score_PI,
        score_PII,
        comp1_PI,
        comp1_PII,
        comp2_PI,
        comp2_PII,
        time_stamp,
        idx_PII,
        N_PIIs,
        gamma,
        alarm_level,
        nugget,
        fig_name,
        comp_name1,
        comp_name2,
        FLAGS,
        comp_score_offset,
        mu_train, Sinv_train,
        eff_wind_len_factor,
        start_score_PI,
        start_comp1_PI,
        start_comp2_PI,
        time_step=1,
        thre_flag=True,
        date_index_flag=False,
        to_decode=False,
        fplot=True,
        xlabel_name="Phase-I&-II Observation Index",
        ylims=None,
        comp_score1_ylims=None,
        comp_score2_ylims=None,
        xlims=None,
        comp_score1_xlims=None,
        comp_score2_xlims=None,
        phase='PIPII'):
    """ Calculate and plot hotelling control chart with classification error/
        regression residual and deviance."""
    # offset = OFFSET
    # # Phase I data
    # (t2ewmaI, start_score_PII, mu, Sinv2, ucl_score,
    #  fp_score_PI, rl_fp_score_PI, len_after_detect_score_PI,
    #  signal_ratio_score_PI, flag_rl_score_PI) = calEwmaT2StatisticsPI(
    #     offset, alarm_level, score_PI, gamma, nugget, start_score_PI)
    #
    # # Phase II data:
    # N_PII = score_PII.shape[0]
    # # scoreII = score(model, X_test, y_test, reg_val)
    # score_ewma = Scores_ewma(score_PII, gamma, score_ewma[ -1, :])
    # t2ewmaII = np.zeros((N_PII,))
    # for t in range(N_PII):
    #     t2ewmaII[t] = HotellingT2(score_ewma[t, :], mu, Sinv2)
    #
    # # Control Chart
    # # alarm_level = 99.99
    # offset = OFFSET
    # ewma_comp1_PI = Scores_ewma(np.reshape(comp1_PI, (-1, 1)), gamma, np.mean(comp1_PI))
    # ewma_comp1_PI = np.reshape(ewma_comp1_PI, (-1,))
    # ewma_comp2_PI = Scores_ewma(np.reshape(comp2_PI, (-1, 1)), gamma, np.mean(comp2_PI))
    # ewma_comp2_PI = np.reshape(ewma_comp2_PI, (-1,))
    #
    # ewma_comp1_PII = Scores_ewma(np.reshape(comp1_PII, (-1, 1)), gamma, ewma_comp1_PI[-1])
    # ewma_comp1_PII = np.reshape(ewma_comp1_PII, (-1,))
    # ewma_comp2_PII = Scores_ewma(np.reshape(comp2_PII, (-1, 1)), gamma, ewma_comp2_PI[-1])
    # ewma_comp2_PII = np.reshape(ewma_comp2_PII, (-1,))

    offset = OFFSET

    # EWMA-T2
    # Phase-I
    ext_len, start_score_PII, _, t2ewmaI = EwmaT2PI(score_PI, mu_train, Sinv_train,
        gamma, eff_wind_len_factor, nugget, start_score_PI)
    # I append a part of data at the end of Phase I towards the beginning of the Phase I data 
    # so that the beginning of calculated ewma of Phase I is somewhat stable, in distribution.
    # In this way, the empirical control limits is trustworthy. Otherwise, control limits 
    # will be too large, and the detection becomes not very sensitive.
    t2ewmaI = t2ewmaI[offset+ext_len:] # Rule-out the extended length at the beginning.

    # Phase-II
    if score_PII is not None and score_PII.shape[0]:
        _, t2ewmaII = EwmaT2PII(score_PII, mu_train, Sinv_train, gamma, start_score_PII)
    else:
        t2ewmaII = np.zeros((0,))

    # Comp1 statistics
    # Phase-I
    _, start_comp1_PII, ewma_comp1_PI = EwmaPI(comp1_PI, gamma, eff_wind_len_factor, start_comp1_PI)
    ewma_comp1_PI = ewma_comp1_PI[offset+ext_len:]

    # Phase-II
    if comp1_PII is not None and comp1_PII.shape[0]:
        ewma_comp1_PII = EwmaPII(comp1_PII, gamma, start_comp1_PII)
    else:
        ewma_comp1_PII = np.zeros((0,))

    # Comp2 statistics
    # Phase-I
    _, start_comp2_PII, ewma_comp2_PI = EwmaPI(comp2_PI, gamma, eff_wind_len_factor, start_comp2_PI)
    ewma_comp2_PI = ewma_comp2_PI[offset+ext_len:]

    # Phase-II
    if comp2_PII is not None and comp2_PII.shape[0]:
        ewma_comp2_PII = EwmaPII(comp2_PII, gamma, start_comp2_PII)
    else:
        ewma_comp2_PII = np.zeros((0,))

    # (t2ewmaI, start_score_PII, ucl_score,
    #  fp_score_PI, rl_fp_score_PI, len_after_detect_score_PI,
    #  signal_ratio_score_PI, flag_rl_score_PI) = calEwmaT2StatisticsPI(
    #     offset, alarm_level, score_PI, mu_train, Sinv_train, gamma,
    #     eff_wind_len_factor, nugget, start_score_PI)
    #
    # (t2ewmaII, fp_score_PII, rl_fp_score_PII,
    #  len_after_detect_score_PII, signal_ratio_score_PII,
    #  flag_rl_score_PII) = calEwmaT2StatisticsPII(
    #     score_PII, mu_train, Sinv_train, gamma, start_score_PII, ucl_score)
    #
    # (ewma_comp1_PI, start_comp1_PII, lcl_comp1, ucl_comp1, fp_comp1_PI,
    #  fp_comp_l1_PI, fp_comp_u1_PI, rl_fp_comp1_PI, len_after_detect_comp1_PI,
    #  signal_ratio_comp1_PI, flag_rl_comp1_PI) = calEwmaStatisticsPI(
    #     offset, alarm_level, comp1_PI, gamma, eff_wind_len_factor, start_comp1_PI)
    #
    # (ewma_comp1_PII, fp_comp1_PII, fp_comp_l1_PII, fp_comp_u1_PII,
    #  rl_fp_comp1_PII, len_after_detect_comp1_PII, signal_ratio_comp1_PII,
    #  flag_rl_comp1_PII) = calEwmaStatisticsPII(
    #     comp1_PII, gamma, start_comp1_PII, lcl_comp1, ucl_comp1)
    #
    # (ewma_comp2_PI, start_comp2_PII, lcl_comp2, ucl_comp2, fp_comp2_PI,
    #  fp_comp_l2_PI, fp_comp_u2_PI, rl_fp_comp2_PI, len_after_detect_comp2_PI,
    #  signal_ratio_comp2_PI, flag_rl_comp2_PI) = calEwmaStatisticsPI(
    #     offset, alarm_level, comp2_PI, gamma, eff_wind_len_factor, start_comp2_PI)
    #
    # (ewma_comp2_PII, fp_comp2_PII, fp_comp_l2_PII, fp_comp_u1_PII,
    #  rl_fp_comp2_PII, len_after_detect_comp2_PII, signal_ratio_comp2_PII,
    #  flag_rl_comp2_PII) = calEwmaStatisticsPII(
    #     comp2_PII, gamma, start_comp2_PII, lcl_comp2, ucl_comp2)

    print(t2ewmaI.shape, t2ewmaII.shape, 
          ewma_comp1_PI.shape, ewma_comp1_PII.shape,
          ewma_comp2_PI.shape, ewma_comp2_PII.shape)

    if fplot:
        print("From HotellingT2CC_Other_Score2----------\n")
        # print comp_score1_PI, comp_score1_PII
        PlotHotellingT2CC_Other_Score2(t2ewmaI,
                                       t2ewmaII,
                                       ewma_comp1_PI,
                                       ewma_comp1_PII,
                                       ewma_comp2_PI,
                                       ewma_comp2_PII,
                                       alarm_level,
                                       time_stamp[offset:],
                                       idx_PII - offset,
                                       N_PIIs,
                                       FLAGS.training_res_folder,
                                       fig_name,
                                       comp_name1,
                                       comp_name2,
                                       time_step,
                                       comp_score_offset,
                                       sign_factor=1,
                                       thre_flag=thre_flag,
                                       date_index_flag=date_index_flag,
                                       to_decode=to_decode,
                                       xlabel_name=xlabel_name,
                                       ylims=ylims,
                                       comp_score1_ylims=comp_score1_ylims,
                                       comp_score2_ylims=comp_score2_ylims,
                                       xlims=xlims,
                                       comp_score1_xlims=comp_score1_xlims,
                                       comp_score2_xlims=comp_score2_xlims,
                                       phase=phase)

        # PlotHotellingT2CC_Other_Score2(t2ewmaI,
        #                                t2ewmaII,
        #                                ewma_comp1_PI,
        #                                ewma_comp1_PII,
        #                                ewma_comp2_PI,
        #                                ewma_comp2_PII,
        #                                alarm_level,
        #                                time_stamp[offset:],
        #                                idx_PII - offset,
        #                                N_PIIs,
        #                                FLAGS.training_res_folder,
        #                                fig_name,
        #                                comp_name1,
        #                                comp_name2,
        #                                time_step,
        #                                comp_score_offset,
        #                                -1,
        #                                date_index_flag)

    return (t2ewmaI, t2ewmaII,
            ewma_comp1_PI, ewma_comp1_PII,
            ewma_comp2_PI, ewma_comp2_PII)


def PlotMonStatOneAxis_Split(
        ax,
        stat,
        xlabel_name,
        ylabel_name,
        label_size,
        thre_ls,
        thre_style_ls,
        time_stamp,
        idx_start,
        idx_end,
        time_step,
        col_label="",
        comp_score_offset=0,
        sign_factor=1,
        thre_flag=True,
        date_index_flag=False):
    """Plot two axes for Phase-I&-II separately."""
    ax.set_xlabel(xlabel_name, size=label_size)
    ax.set_ylabel(
        ylabel_name,
        color='k',
        size=label_size)
    ax.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
    n_sample = len(stat)
    if comp_score_offset>0:
        yrange = max(stat[comp_score_offset:]) - min(stat[comp_score_offset:])
        ymax = max(stat[comp_score_offset:])
        ymin = min(stat[comp_score_offset:])
    else:
        yrange = max(stat) - min(stat)
        ymax = max(stat)
        ymin = min(stat)
    if sign_factor == -1:
        ax.set_ylim([ymax + yrange / 10,
                     ymin - yrange / 10])
    else:
        ax.set_ylim([ymin - yrange / 10,
                     ymax + yrange / 10])
    ax.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    ax.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

    ax.plot(np.arange(len(stat)), stat, 'b')
    if thre_flag:
        for thre, sty in zip(thre_ls, thre_style_ls):
            ax.plot(np.arange(n_sample), np.repeat(thre, n_sample), sty, lw=1)

    # for i in np.arange(time_stamp[idx_start], time_stamp[idx_end - 1] + 1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax.text(j - idx_start, ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
    #              MonIdx2Date(time_stamp[j], date_index_flag), rotation=45)


def PlotMonStatOneAxisPIPII_Separated(
        ax1,
        ax2,
        statI,
        statII,
        xlabel_name1,
        xlabel_name2,
        ylabel_name,
        label_size,
        thre_ls,
        thre_style_ls,
        time_stamp,
        N_PIIs,
        time_step,
        folder_path,
        fig_name_post,
        col_label="",
        comp_score_offset=0,
        sign_factor=1,
        thre_flag=True,
        date_index_flag=False):
    """Plot two axes for Phase-I&-II separately each in one figure."""
    # Left plot
    label_size = LAB_SIZE
    # line_width = 1.5
    # marker_size = 5
    plt.figure(num=None, figsize=(ONE_FIG_HEI, 0.6*ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = 0.3)
    ax1 = plt.subplot(111)
    PlotMonStatOneAxis_Split(
            ax1,
            statI,
            xlabel_name1,
            ylabel_name,
            label_size,
            thre_ls,
            thre_style_ls,
            time_stamp,
            0,
            N_PIIs[0],
            time_step,
            col_label=col_label,
            comp_score_offset=comp_score_offset,
            sign_factor=sign_factor,
            thre_flag=thre_flag,
            date_index_flag=date_index_flag)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, '_'.join(['PI', fig_name_post])), bbox_inches='tight')
    # plt.show()
    plt.close()

    # Right plot
    # line_width = 1.5
    # marker_size = 5
    plt.figure(num=None, figsize=(ONE_FIG_HEI, 0.6*ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace = 0.3)
    ax2 = plt.subplot(111)
    PlotMonStatOneAxis_Split(
            ax2,
            statII,
            xlabel_name2,
            ylabel_name,
            label_size,
            thre_ls,
            thre_style_ls,
            time_stamp,
            N_PIIs[0],
            N_PIIs[1],
            time_step,
            col_label=col_label,
            comp_score_offset=comp_score_offset,
            sign_factor=sign_factor,
            thre_flag=thre_flag,
            date_index_flag=date_index_flag)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, '_'.join(['PII', fig_name_post])), bbox_inches='tight')
    # plt.show()
    plt.close()


def PlotMonStatOneAxisPIPII_Split(
        ax1,
        ax2,
        statI,
        statII,
        xlabel_name1,
        xlabel_name2,
        ylabel_name,
        label_size,
        thre_ls,
        thre_style_ls,
        time_stamp,
        N_PIIs,
        time_step,
        col_label="",
        comp_score_offset=0,
        sign_factor=1,
        thre_flag=True,
        date_index_flag=False):
    """Plot two axes for Phase-I&-II separately in two axes."""
    # Left plot
    PlotMonStatOneAxis_Split(
            ax1,
            statI,
            xlabel_name1,
            ylabel_name,
            label_size,
            thre_ls,
            thre_style_ls,
            time_stamp,
            0,
            N_PIIs[0],
            time_step,
            col_label=col_label,
            comp_score_offset=comp_score_offset,
            sign_factor=sign_factor,
            thre_flag=thre_flag,
            date_index_flag=date_index_flag)
    # Right plot
    PlotMonStatOneAxis_Split(
            ax2,
            statII,
            xlabel_name2,
            ylabel_name,
            label_size,
            thre_ls,
            thre_style_ls,
            time_stamp,
            N_PIIs[0],
            sum(N_PIIs),
            time_step,
            col_label=col_label,
            comp_score_offset=0,
            sign_factor=sign_factor,
            thre_flag=thre_flag,
            date_index_flag=date_index_flag)


def PlotMonStatOneAxisPIPII(
        ax,
        statI,
        statII,
        xlabel_name,
        ylabel_name,
        label_size,
        thre_ls,
        thre_style_ls,
        time_stamp,
        N_PIIs,
        time_step,
        col_label="",
        comp_score_offset=0,
        sign_factor=1,
        thre_flag=True,
        date_index_flag=False,
        to_decode=False, 
        xlabel_flag=True,
        ylims=None,
        xlims=None):
    """Plot one single axis for a big plot of Phase-I&-II."""
    if xlabel_flag:
        ax.get_xaxis().set_label_coords(0.5, XLAB_YPOS)
        ax.set_xlabel(xlabel_name, size=1.5*label_size)
    ax.set_ylabel(ylabel_name, color='b', size=label_size)
    ax.get_yaxis().set_label_coords(YLAB_XPOS, 0.5) # Control the space between axis label and axis ticks
    ax.get_xaxis().set_label_coords(0.5, XLAB_YPOS)
    statI_II = np.hstack((statI, statII))
    n_sample = len(statI_II)
    if comp_score_offset>0:
        yrange = max(statI_II[comp_score_offset:]) - min(statI_II[comp_score_offset:])
        ymax = max(statI_II[comp_score_offset:])
        ymin = min(statI_II[comp_score_offset:])
    else:
        yrange = max(statI_II) - min(statI_II)
        ymax = max(statI_II)
        ymin = min(statI_II)
    if ylims is None:
        if sign_factor == -1:
            ax.set_ylim([ymax + yrange / 10, ymin - yrange / 10])
        else:
            ax.set_ylim([ymin - yrange / 10, ymax + yrange / 10])
    else:
        ax.set_ylim(ylims)
        # ax.set_ylim([-50, 200])
        # ax.set_ylim([ymin - yrange / 10, min(800, ymax + yrange / 10)])
        # ax.set_ylim([0, 1000])
    if xlims is not None:
        ax.set_xlim(xlims)
    ax.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    ax.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

    ax.plot(np.arange(len(statI_II)), statI_II, 'b')
    if thre_flag:
        for thre, sty in zip(thre_ls, thre_style_ls):
            ax.plot(np.arange(n_sample), np.repeat(
                    thre, n_sample), sty, lw=1)
    if len(statI)*len(statII)>0:
        ax.axvline(x=len(statI)) # Separation b/w Phase-I&-II
    for idx in len(statI) + np.cumsum(N_PIIs[:-1]):
        ax.axvline(x=idx, color='g') # Separation b/w different stages of concept drift in Phase-II
    if time_stamp.shape[0]>0 and date_index_flag:
        set_xticks_xticklabels(ax, time_stamp, time_step, label_size=label_size, rotation=ROTATION, to_decode=to_decode)


def PlotHotellingT2CC_Other_Score2(
        t2ewmaI,
        t2ewmaII,
        comp_score1_PI,
        comp_score1_PII,
        comp_score2_PI,
        comp_score2_PII,
        alarm_level,
        time_stamp,
        idx_PII,
        N_PIIs,
        folder_path,
        fig_name,
        comp_name1,
        comp_name2,
        time_step,
        comp_score_offset,
        sign_factor=1,
        thre_flag=True,
        date_index_flag=False,
        to_decode=False,
        xlabel_name = "Phase-I&-II Observation Index",
        ylims=None,
        comp_score1_ylims=None,
        comp_score2_ylims=None,
        xlims=None,
        comp_score1_xlims=None,
        comp_score2_xlims=None,
        phase='PIPII'):
    """Based on Phase-I&-II ewma hotelling time_stamp to plot control chart."""
    # print(comp_score1_PI.shape)
    ucl5 = np.percentile(t2ewmaI[comp_score_offset:], alarm_level)
    ucl_comp1 = np.percentile(
        comp_score1_PI[comp_score_offset:], (100.0 + alarm_level) / 2)
    lcl_comp1 = np.percentile(
        comp_score1_PI[comp_score_offset:], (100.0 - alarm_level) / 2)
    ucl_comp2 = np.percentile(
        comp_score2_PI[comp_score_offset:], (100.0 + alarm_level) / 2)
    lcl_comp2 = np.percentile(
        comp_score2_PI[comp_score_offset:], (100.0 - alarm_level) / 2)
    # comp_score = np.hstack((comp_score_PI, comp_score_PII))

    label_size = LAB_SIZE
    # line_width = 1.5
    # marker_size = 5
    # Only plot score and prediction error/residual
    plt.figure(num=None, figsize=(15, 2*0.95*ONE_FIG_HEI), dpi=DPI, facecolor='w', edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=2*AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE, wspace=0.3)
    # ax1 = plt.subplot(311)
    ax1 = plt.subplot(211)
    ylabel_name = "MEWMA Multivariate\nScore Vector"
    thre_ls = [ucl5, 0]
    thre_style_ls = ['b--', 'k--']
    if phase=='PIPII':
        PlotMonStatOneAxisPIPII(ax1, t2ewmaI, t2ewmaII, xlabel_name, ylabel_name, label_size,
                thre_ls, thre_style_ls, time_stamp, N_PIIs, time_step, comp_score_offset=comp_score_offset,
                sign_factor=sign_factor, thre_flag=thre_flag, date_index_flag=date_index_flag, to_decode=to_decode, xlabel_flag=True, ylims=ylims, xlims=xlims)
    else:
        PlotMonStatOneAxisPII(ax1, t2ewmaII, xlabel_name, ylabel_name, label_size,
                thre_ls, thre_style_ls, time_stamp[idx_PII:], N_PIIs, time_step, comp_score_offset=comp_score_offset,
                sign_factor=sign_factor, date_index_flag=date_index_flag, to_decode=to_decode, xlabel_flag=True, ylims=ylims, xlims=xlims)

    # ax1.set_xlabel('Phase-I&-II Observation Index', size=label_size)
    # ax1.set_ylabel(
    #     'MEWMA Multivariate\nScore Vector',
    #     color='b',
    #     size=label_size)
    # ax1.get_yaxis().set_label_coords(-0.06, 0.5)
    # t2ewmaI_II = np.hstack((t2ewmaI, t2ewmaII))
    # yrange = max(t2ewmaI_II) - min(t2ewmaI_II)
    # if sign_factor == -1:
    #     ax1.set_ylim([max(t2ewmaI_II) + yrange / 10,
    #                   min(t2ewmaI_II) - yrange / 10])
    # else:
    #     ax1.set_ylim([min(t2ewmaI_II) - yrange / 10,
    #                   max(t2ewmaI_II) + yrange / 10])
    # ax1.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    # ax1.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    #
    # ax1.plot(np.arange(len(t2ewmaI_II)), t2ewmaI_II, 'b')
    # ax1.plot(
    #     np.arange(
    #         len(t2ewmaI_II)), np.repeat(
    #         ucl5, len(t2ewmaI_II)), 'b--')
    # ax1.plot(
    #     np.arange(
    #         len(t2ewmaI_II)), np.repeat(
    #         0, len(t2ewmaI_II)), 'k--', lw=0.5)
    # ax1.axvline(x=len(t2ewmaI))
    # for idx_PII in len(t2ewmaI) + np.cumsum(N_PIIs[:-1]):
    #     ax1.axvline(x=idx_PII, color='g')
    # for i in np.arange(time_stamp[0], time_stamp[idx_PII - 1] + 1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax1.text(j,
    #              ax1.get_ylim()[0] + 0.1 * (ax1.get_ylim()
    #                                         [1] - ax1.get_ylim()[0]),
    #              MonIdx2Date(time_stamp[j],
    #                          date_index_flag),
    #              rotation=45,
    #              fontsize=label_size * AX_LAB_SCALE)
    # ax2 = ax1.twinx()

    ax2 = plt.subplot(212)
    ylabel_name = comp_name1
    thre_ls = [ucl_comp1, lcl_comp1]
    thre_style_ls = ['k--', 'k--']
    if phase=='PIPII':
        PlotMonStatOneAxisPIPII(ax2, comp_score1_PI, comp_score1_PII, xlabel_name,
                ylabel_name, label_size, thre_ls, thre_style_ls, time_stamp,
                N_PIIs, time_step, comp_score_offset=comp_score_offset,
                sign_factor=sign_factor, thre_flag=thre_flag, date_index_flag=date_index_flag, to_decode=to_decode, xlabel_flag=True, ylims=comp_score1_ylims, xlims=comp_score1_xlims)
    else:
        PlotMonStatOneAxisPII(ax2, comp_score1_PII, xlabel_name, ylabel_name, label_size,
                thre_ls, thre_style_ls, time_stamp[idx_PII:], N_PIIs, time_step, comp_score_offset=comp_score_offset,
                sign_factor=sign_factor, date_index_flag=date_index_flag, to_decode=to_decode, xlabel_flag=True, ylims=ylims, xlims=xlims)

    # ax2.set_xlabel('Phase-I&-II Observation Index', size=label_size)
    # ax2.set_ylabel(comp_name1, color='k', size=label_size)
    # ax2.get_yaxis().set_label_coords(-0.06, 0.5)
    # comp_score1_PI_PII = np.hstack((comp_score1_PI, comp_score1_PII))
    # margin1 = (max(comp_score1_PI_PII[comp_score_offset:]) -
    #     min(comp_score1_PI_PII[comp_score_offset:])) / 10
    # if sign_factor == -1:
    #     ax2.set_ylim([max(comp_score1_PI_PII[comp_score_offset:]) + margin1,
    #                   min(comp_score1_PI_PII[comp_score_offset:]) - margin1])
    # else:
    #     ax2.set_ylim([min(comp_score1_PI_PII[comp_score_offset:]) - margin1,
    #                   max(comp_score1_PI_PII[comp_score_offset:]) + margin1])
    # ax2.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    # ax2.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    #
    # ax2.plot(np.arange(comp_score_offset, len(comp_score1_PI_PII)),
    #          comp_score1_PI_PII[comp_score_offset:], 'k')
    # ax2.plot(
    #     np.arange(
    #         len(comp_score1_PI_PII)), np.repeat(
    #         ucl_comp1, len(comp_score1_PI_PII)), 'k--')
    # ax2.plot(
    #     np.arange(
    #         len(comp_score1_PI_PII)), np.repeat(
    #         lcl_comp1, len(comp_score1_PI_PII)), 'k--')
    # ax2.axvline(x=len(comp_score1_PI))
    # for idx_PII in len(comp_score1_PI) + np.cumsum(N_PIIs[:-1]):
    #     ax2.axvline(x=idx_PII, color='g')
    # for i in np.arange(time_stamp[0], time_stamp[idx_PII - 1] + 1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax2.text(j,
    #              (ax2.get_ylim()[0] + 0.1 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])),
    #              MonIdx2Date(time_stamp[j],
    #                          date_index_flag),
    #              rotation=45,
    #              fontsize=label_size * AX_LAB_SCALE)

    # ax3 = plt.subplot(313)
    # ylabel_name = comp_name2
    # thre_ls = [ucl_comp2, lcl_comp2]
    # thre_style_ls = ['k--', 'k--']
    # PlotMonStatOneAxisPIPII(ax3, comp_score2_PI, comp_score2_PII, xlabel_name,
    #         ylabel_name, label_size, thre_ls, thre_style_ls, time_stamp,
    #         N_PIIs, time_step, comp_score_offset=comp_score_offset,
    #         sign_factor=sign_factor, thre_flag=thre_flag ,date_index_flag=date_index_flag, xlabel_flag=True)

    # ax2.set_xlabel('Phase-I&-II Observation Index', size=label_size)
    # ax2.set_ylabel(comp_name2, color='k', size=label_size)
    # ax2.get_yaxis().set_label_coords(-0.06, 0.5)
    # comp_score2_PI_PII = np.hstack((comp_score2_PI, comp_score2_PII))
    # margin2 = (max(comp_score2_PI_PII[comp_score_offset:]) -
    #     min(comp_score2_PI_PII[comp_score_offset:])) / 10
    # if sign_factor == -1:
    #     ax2.set_ylim([max(comp_score2_PI_PII[comp_score_offset:]) + margin2,
    #                   min(comp_score2_PI_PII[comp_score_offset:]) - margin2])
    # else:
    #     ax2.set_ylim([min(comp_score2_PI_PII[comp_score_offset:]) - margin2,
    #                   max(comp_score2_PI_PII[comp_score_offset:]) + margin2])
    # ax2.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    # ax2.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    #
    # ax2.plot(np.arange(comp_score_offset, len(comp_score2_PI_PII)),
    #          comp_score2_PI_PII[comp_score_offset:], 'k')
    # ax2.plot(
    #     np.arange(
    #         len(comp_score2_PI_PII)), np.repeat(
    #         ucl_comp2, len(comp_score2_PI_PII)), 'k--')
    # ax2.plot(
    #     np.arange(
    #         len(comp_score2_PI_PII)), np.repeat(
    #         lcl_comp2, len(comp_score2_PI_PII)), 'k--')
    # ax2.axvline(x=len(comp_score2_PI))
    # for idx_PII in len(comp_score2_PI) + np.cumsum(N_PIIs[:-1]):
    #     ax2.axvline(x=idx_PII, color='g')
    # for i in np.arange(time_stamp[0], time_stamp[idx_PII - 1] + 1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax2.text(j,
    #              ax2.get_ylim()[0] + 0.1 * (ax2.get_ylim()
    #                                         [1] - ax2.get_ylim()[0]),
    #              MonIdx2Date(time_stamp[j],
    #                          date_index_flag),
    #              rotation=45,
    #              fontsize=label_size * AX_LAB_SCALE)

    # ax1 = plt.subplot(322)
    # ax1.set_xlabel('Phase-II Observation Index',size=label_size)
    # ax1.set_ylabel('MEWMA Multivariate\nScore Vector', color='b',size=label_size)
    # ax1.get_yaxis().set_label_coords(-0.12,0.5)
    # ax1.xaxis.set_tick_params(labelsize=label_size*0.5)
    # ax1.yaxis.set_tick_params(labelsize=label_size*0.5)
    #
    # # ax1.plot(np.arange(len(t2ewmaII)), t2ewmaII, 'b', label='Score Function\n(Hotelling $T^2$ of EWMA)')
    # ax1.plot(np.arange(len(t2ewmaII)), t2ewmaII, 'b')
    # ax1.plot(np.arange(len(t2ewmaII)), np.repeat(ucl5, len(t2ewmaII)), 'b--')
    # ax1.plot(np.arange(len(t2ewmaII)), np.repeat(0, len(t2ewmaII)), 'k--', lw=0.5)
    # tmp = np.min(t2ewmaII)
    # for i in np.arange(time_stamp[idx_PII], max(time_stamp)+1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax1.text(j - idx_PII, ax1.get_ylim()[0]+0.1*(ax1.get_ylim()[1]-ax1.get_ylim()[0]),
    #              MonIdx2Date(time_stamp[j], date_index_flag), rotation=45, fontsize=label_size*0.5)
    # # lines1, labels1 = ax1.get_legend_handles_labels()
    # # ax1.legend(lines1, labels1, loc='upper left',
    # #            bbox_to_anchor=(0,0.95), prop={'size':label_size}, ncol=2)
    #
    # # ax2 = ax1.twinx()
    # ax2 = plt.subplot(324)
    # ax2.set_xlabel('Phase-II Observation Index',size=label_size)
    # ax2.set_ylabel(comp_name1, color='k',size=label_size)
    # if sign_factor==-1:
    #     ax2.set_ylim([max(comp_score1_PII)+(max(comp_score1_PII)-min(comp_score1_PII))/10,
    #                  min(comp_score1_PII)-(max(comp_score1_PII)-min(comp_score1_PII))/10])
    # else:
    #     ax2.set_ylim([min(comp_score1_PII)-(max(comp_score1_PII)-min(comp_score1_PII))/10,
    #                  max(comp_score1_PII)+(max(comp_score1_PII)-min(comp_score1_PII))/10])
    # ax2.get_yaxis().set_label_coords(-0.12,0.5)
    # ax2.xaxis.set_tick_params(labelsize=label_size*0.5)
    # ax2.yaxis.set_tick_params(labelsize=label_size*0.5)
    #
    # # ax2.plot(np.arange(len(comp_score_PII)), comp_score_PII, 'k' ,label=comp_name)
    # ax2.plot(np.arange(len(comp_score1_PII)), comp_score1_PII, 'k')
    # ax2.plot(np.arange(len(comp_score1_PII)), np.repeat(ucl_comp1, len(comp_score1_PII)), 'k--')
    # ax2.plot(np.arange(len(comp_score1_PII)), np.repeat(lcl_comp1, len(comp_score1_PII)), 'k--')
    # tmp = np.min(t2ewmaII)
    # for i in np.arange(time_stamp[idx_PII], max(time_stamp)+1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax2.text(j - idx_PII, ax2.get_ylim()[0]+0.1*(ax2.get_ylim()[1]-ax2.get_ylim()[0]),
    #              MonIdx2Date(time_stamp[j], date_index_flag), rotation=45, fontsize=label_size*0.5)
    #
    # ax2 = plt.subplot(326)
    # ax2.set_xlabel('Phase-II Observation Index',size=label_size)
    # ax2.set_ylabel(comp_name2, color='k',size=label_size)
    # if sign_factor==-1:
    #     ax2.set_ylim([max(comp_score2_PII)+(max(comp_score2_PII)-min(comp_score2_PII))/10,
    #                  min(comp_score2_PII)-(max(comp_score2_PII)-min(comp_score2_PII))/10])
    # else:
    #     ax2.set_ylim([min(comp_score2_PII)-(max(comp_score2_PII)-min(comp_score2_PII))/10,
    #                  max(comp_score2_PII)+(max(comp_score2_PII)-min(comp_score2_PII))/10])
    # ax2.get_yaxis().set_label_coords(-0.12,0.5)
    # ax2.xaxis.set_tick_params(labelsize=label_size*0.5)
    # ax2.yaxis.set_tick_params(labelsize=label_size*0.5)
    #
    # # ax2.plot(np.arange(len(comp_score_PII)), comp_score_PII, 'k' ,label=comp_name)
    # ax2.plot(np.arange(len(comp_score2_PII)), comp_score2_PII, 'k')
    # ax2.plot(np.arange(len(comp_score2_PII)), np.repeat(ucl_comp2, len(comp_score2_PII)), 'k--')
    # ax2.plot(np.arange(len(comp_score2_PII)), np.repeat(lcl_comp2, len(comp_score2_PII)), 'k--')
    # tmp = np.min(t2ewmaII)
    # for i in np.arange(time_stamp[idx_PII], max(time_stamp)+1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     ax2.text(j - idx_PII, ax2.get_ylim()[0]+0.1*(ax2.get_ylim()[1]-ax2.get_ylim()[0]),
    #              MonIdx2Date(time_stamp[j], date_index_flag), rotation=45, fontsize=label_size*0.5)
    #
    # # lines2, labels2 = ax2.get_legend_handles_labels()
    # # # ax2.legend(lines1+lines2, labels1+labels2, loc=0)
    # # ax2.legend(lines2, labels2, loc='upper left',
    # #            bbox_to_anchor=(0,0.95), prop={'size':label_size}, ncol=2)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, fig_name), bbox_inches='tight')
    # plt.show()
    plt.close()


# def PlotHotellingT2Single(
#         t2ewma,
#         alarm_level,
#         time_stamp,
#         folder_path,
#         fig_name,
#         time_step,
#         date_index_flag=False):
#     """Based on a batch of ewma hotelling t2 score and time_stamp to plot
#        control chart. The control charts are in two separate plots one over
#        the other. The above is Phase-I and below is Phase-II.
#     """
#     ucl5 = np.percentile(t2ewma, alarm_level)
#
#     label_size = 16
#     # line_width = 1.5
#     # marker_size = 5
#     plt.figure(num=None, figsize=(10, 5), dpi=DPI, facecolor='w', edgecolor='k')
#     plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=0.5)
#     ax1 = plt.subplot(111)
#     ax1.set_xlabel('Observation Index', size=label_size)
#     ax1.set_ylabel(
#         'MEWMA Multivariate\nScore Vector',
#         color='k',
#         size=label_size)
#     ax1.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
#     ax1.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
#     ax1.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
#
#     ax1.plot(np.arange(len(t2ewma)), t2ewma)
#     ax1.plot(np.arange(len(t2ewma)), np.repeat(ucl5, len(t2ewma)))
#     ax1.plot(np.arange(len(t2ewma)), np.repeat(0, len(t2ewma)), 'k', lw=0.5)
#     tmp = np.min(t2ewma)
#     for i in np.arange(time_stamp[0], time_stamp[-1] + 1, time_step):
#         j = np.argmax(time_stamp >= i)
#         ax1.text(j,
#                  ax1.get_ylim()[0] + 0.1 * (ax1.get_ylim()
#                                             [1] - ax1.get_ylim()[0]),
#                  MonIdx2Date(time_stamp[j],
#                              date_index_flag),
#                  rotation=45)
#
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     plt.savefig(folder_path + fig_name)
#     # plt.show()
#     plt.close()


def HotellingCC_Multi_Lines(
        score_PI,
        score_PII,
        col_ls,
        label_ls,
        time_stamp,
        N_PIIs,
        gamma,
        alarm_level,
        nugget,
        fig_name,
        FLAGS,
        mu_train, Sinv_train,
        eff_wind_len_factor,
        start_score_PI,
        time_step=1,
        xlabel_name='Observation Index',
        thre_flag=True,
        date_index_flag=False, 
        to_decode=False,
        fplot=True,
        log_flag=False,
        plot_row=6,
        phase='PIPII'):
    """ Calculate and plot hotelling control chart for each coordinates or
        mix different coordinates together."""

    offset = OFFSET

    # EWMA-T2
    # Phase-I
    ext_len, start_score_PII, ewma_score_PI, t2ewmaI = EwmaT2PI(score_PI, mu_train, Sinv_train,
        gamma, eff_wind_len_factor, nugget, start_score_PI)
    ewma_score_PI = ewma_score_PI[offset+ext_len:]
    t2ewmaI = t2ewmaI[offset+ext_len:]

    # Phase-II
    ewma_score_PII, t2ewmaII = EwmaT2PII(score_PII, mu_train, Sinv_train, gamma, start_score_PII)

    # score_dim = score_PI.shape[1]
    # # Phase I data
    # N_PI = score_PI.shape[0]
    # Sinv = InversedCov(score_PI, nugget)  # nnet needs 0.15
    # mu = np.mean(score_PI, axis=0)
    #
    # # gamma = 0.001
    # score_ewma =  Scores_ewma(
    #     score_PI, gamma, np.mean(score_PI, 0))  # starting point
    #
    # Sinv2 = Sinv * (2 - gamma) / gamma
    # ewmaI = np.zeros((N_PI, score_dim))
    # t2ewmaI = np.zeros((N_PI, ))
    # for t in range(N_PI):
    #     # Statistical quality control-(11.32)
    #     # * (Sinv2 / (1 - (1 - gamma)**(2 * (1 + t))))**0.5
    #     ewmaI[t] = (score_ewma[ t, :] - mu)
    #     t2ewmaI[t] = HotellingT2(
    #         score_ewma[ t, :], mu, Sinv2 / (1 - (1 - gamma)**(2 * (1 + t))))
    #
    # # Phase II data:
    # N_PII = score_PII.shape[0]
    # # scoreII = score(model, X_test, y_test, reg_val)
    # score_ewma = Scores_ewma(score_PII, gamma, score_ewma[ -1, :])
    # ewmaII = np.zeros((N_PII, score_dim))
    # t2ewmaII = np.zeros((N_PII,))
    # for t in range(N_PII):
    #     ewmaII[t] = (score_ewma[t, :] - mu)  # * Sinv2**0.5
    #     t2ewmaII[t] = HotellingT2(score_ewma[t, :], mu, Sinv2)

    # Control Chart
    # alarm_level = 99.99

    # Linear y in Phase-II
    # Multi-lines a subplot
    offset = OFFSET
    # PlotHotellingCC_Multi_Lines(t2ewmaI,
    #                             t2ewmaII,
    #                             ewma_score_PI,
    #                             ewma_score_PII,
    #                             col_ls,
    #                             label_ls,
    #                             alarm_level,
    #                             time_stamp[offset:],
    #                             N_PIIs,
    #                             FLAGS.training_res_folder,
    #                             fig_name,
    #                             time_step=time_step,
    #                             thre_flag=thre_flag,
    #                             sign_factor=1,
    #                             log_flag=log_flag)

    # PlotHotellingCC_Multi_Lines(t2ewmaI,
    #                             t2ewmaII,
    #                             ewma_score_PI,
    #                             ewma_score_PII,
    #                             col_ls,
    #                             label_ls,
    #                             alarm_level,
    #                             time_stamp[offset:],
    #                             N_PIIs,
    #                             FLAGS.training_res_folder,
    #                             fig_name,
    #                             time_step=time_step,
    #                             thre_flag=thre_flag,
    #                             sign_factor=-1,
    #                             log_flag=log_flag)

    # PlotHotellingCC_PII_Multi_Lines(t2ewmaI,
    #                                 t2ewmaII,
    #                                 ewma_score_PI,
    #                                 ewma_score_PII,
    #                                 col_ls,
    #                                 label_ls,
    #                                 alarm_level,
    #                                 time_stamp[offset:],
    #                                 N_PIIs,
    #                                 FLAGS.training_res_folder,
    #                                 fig_name,
    #                                 time_step=time_step,
    #                                 thre_flag=thre_flag,
    #                                 sign_factor=1,
    #                                 log_flag=log_flag)

    # PlotHotellingCC_PII_Multi_Lines(t2ewmaI,
    #                                 t2ewmaII,
    #                                 ewma_score_PI,
    #                                 ewma_score_PII,
    #                                 col_ls,
    #                                 label_ls,
    #                                 alarm_level,
    #                                 time_stamp[offset:],
    #                                 N_PIIs,
    #                                 FLAGS.training_res_folder,
    #                                 fig_name,
    #                                 time_step=time_step,
    #                                 thre_flag=thre_flag,
    #                                 sign_factor=-1,
    #                                 log_flag=log_flag)

    # Single-line a subplot
    if phase=='PIPII':
        PlotHotellingCC_Single_Line(t2ewmaI,
                                    t2ewmaII,
                                    ewma_score_PI-mu_train,
                                    ewma_score_PII-mu_train,
                                    col_ls,
                                    label_ls,
                                    alarm_level,
                                    time_stamp[offset:],
                                    N_PIIs,
                                    FLAGS.training_res_folder,
                                    'single_' + fig_name,
                                    time_step=time_step,
                                    xlabel_name=xlabel_name,
                                    date_index_flag=date_index_flag, 
                                    to_decode=to_decode,
                                    thre_flag=thre_flag,
                                    sign_factor=1,
                                    log_flag=log_flag,
                                    plot_row=plot_row)

        PlotHotellingCC_Single_Line(t2ewmaI,
                                    t2ewmaII,
                                    ewma_score_PI-mu_train,
                                    ewma_score_PII-mu_train,
                                    col_ls,
                                    label_ls,
                                    alarm_level,
                                    time_stamp[offset:],
                                    N_PIIs,
                                    FLAGS.training_res_folder,
                                    'single_' + fig_name,
                                    time_step=time_step,
                                    xlabel_name=xlabel_name,
                                    date_index_flag=date_index_flag,
                                    to_decode=to_decode,
                                    thre_flag=thre_flag,
                                    sign_factor=-1,
                                    log_flag=log_flag,
                                    plot_row=plot_row)
    
    elif phase=='PII':
        PlotHotellingCC_PII_Single_Line(t2ewmaI,
                                        t2ewmaII,
                                        ewma_score_PI-mu_train,
                                        ewma_score_PII-mu_train,
                                        col_ls,
                                        label_ls,
                                        alarm_level,
                                        time_stamp[offset:],
                                        N_PIIs,
                                        FLAGS.training_res_folder,
                                        'single_' + fig_name,
                                        time_step=time_step,
                                        xlabel_name=xlabel_name,
                                        date_index_flag=date_index_flag,
                                        to_decode=to_decode,
                                        thre_flag=thre_flag,
                                        sign_factor=1,
                                        log_flag=log_flag,
                                        plot_row=plot_row)

        PlotHotellingCC_PII_Single_Line(t2ewmaI,
                                        t2ewmaII,
                                        ewma_score_PI-mu_train,
                                        ewma_score_PII-mu_train,
                                        col_ls,
                                        label_ls,
                                        alarm_level,
                                        time_stamp[offset:],
                                        N_PIIs,
                                        FLAGS.training_res_folder,
                                        'single_' + fig_name,
                                        time_step=time_step,
                                        xlabel_name=xlabel_name,
                                        date_index_flag=date_index_flag,
                                        to_decode=to_decode,
                                        thre_flag=thre_flag,
                                        sign_factor=-1,
                                        log_flag=log_flag,
                                        plot_row=plot_row)
                                        
    else:
        PlotHotellingCC_PI_Single_Line(t2ewmaI,
                                       ewma_score_PI-mu_train,
                                       col_ls,
                                       label_ls,
                                       alarm_level,
                                       time_stamp[offset:offset+t2ewmaI.shape[0]],
                                       FLAGS.training_res_folder,
                                       'single_' + fig_name,
                                       time_step = time_step,
                                       xlabel_name=xlabel_name,
                                       date_index_flag=date_index_flag,
                                       to_decode=to_decode,
                                       thre_flag=thre_flag,
                                       sign_factor=1,
                                       log_flag=log_flag,
                                       plot_row=plot_row)

        PlotHotellingCC_PI_Single_Line(t2ewmaI,
                                       ewma_score_PI-mu_train,
                                       col_ls,
                                       label_ls,
                                       alarm_level,
                                       time_stamp[offset:offset+t2ewmaI.shape[0]],
                                       FLAGS.training_res_folder,
                                       'single_' + fig_name,
                                       time_step = time_step,
                                       xlabel_name=xlabel_name,
                                       date_index_flag=date_index_flag,
                                       to_decode=to_decode,
                                       thre_flag=thre_flag,
                                       sign_factor=-1,
                                       log_flag=log_flag,
                                       plot_row=plot_row)

    # ## Log y in Phase-II
    # # Multi-lines a subplot
    # offset = OFFSET
    # PlotHotellingCC_Multi_Lines(t2ewmaI[offset:], t2ewmaII, ewmaI[offset:],
    #                 ewmaII, col_ls, label_ls, alarm_level, time_stamp[offset:], idx_PII-offset,
    #                 os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'log_'+fig_name, time_step, 1, True)
    # PlotHotellingCC_Multi_Lines(t2ewmaI[offset:], t2ewmaII, ewmaI[offset:],
    #                 ewmaII, col_ls, label_ls, alarm_level, time_stamp[offset:], idx_PII-offset,
    #                 os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'log_'+fig_name, time_step, -1, True)
    #
    # PlotHotellingCC_PII_Multi_Lines(t2ewmaI[offset:], t2ewmaII, ewmaI[offset:],
    #                 ewmaII, col_ls, label_ls, alarm_level, time_stamp[offset:], idx_PII-offset,
    #                 os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'log_'+fig_name, time_step, 1, True)
    # PlotHotellingCC_PII_Multi_Lines(t2ewmaI[offset:], t2ewmaII, ewmaI[offset:],
    #                 ewmaII, col_ls, label_ls, alarm_level, time_stamp[offset:], idx_PII-offset,
    #                 os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'log_'+fig_name, time_step, -1, True)
    #
    # # Single-line a subplot
    # PlotHotellingCC_Single_Line(t2ewmaI[offset:], t2ewmaII, ewmaI[offset:],
    #                 ewmaII, col_ls, label_ls, alarm_level, time_stamp[offset:], idx_PII-offset,
    #                 os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'log_single_'+fig_name, time_step, 1, True)
    # PlotHotellingCC_Single_Line(t2ewmaI[offset:], t2ewmaII, ewmaI[offset:],
    #                 ewmaII, col_ls, label_ls, alarm_level, time_stamp[offset:], idx_PII-offset,
    #                 os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'log_single_'+fig_name, time_step, -1, True)
    #
    # PlotHotellingCC_PII_Single_Line(t2ewmaI[offset:], t2ewmaII, ewmaI[offset:],
    #                 ewmaII, col_ls, label_ls, alarm_level, time_stamp[offset:], idx_PII-offset,
    #                 os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'log_single_'+fig_name, time_step, 1, True)
    # PlotHotellingCC_PII_Single_Line(t2ewmaI[offset:], t2ewmaII, ewmaI[offset:],
    #                 ewmaII, col_ls, label_ls, alarm_level, time_stamp[offset:], idx_PII-offset,
    # os.path.join(FLAGS.res_root_dir, FLAGS.model_file_folder), 'log_single_'+fig_name, time_step, -1, True)


def PlotHotellingCC_Multi_Lines(
        t2ewmaI,
        t2ewmaII,
        ewmaI,
        ewmaII,
        col_ls,
        label_ls,
        alarm_level,
        time_stamp,
        N_PIIs,
        folder_path,
        fig_name,
        time_step,
        thre_flag=True,
        sign_factor=1,
        log_flag=False):
    """Based on Phase-I&-II ewma hotelling time_stamp to plot control chart."""
    ucl = np.percentile(ewmaI, (100 + alarm_level) / 2, axis=0)
    lcl = np.percentile(ewmaI, (100 - alarm_level) / 2, axis=0)
    ucl5 = np.percentile(t2ewmaI, alarm_level)

    label_size = LAB_SIZE
    line_width = 1.5
    marker_size = 3
    plt.figure(
        num=None, # Just the index of the plot.
        figsize=(15, 3*ONE_FIG_HEI), # Plot size
        dpi=DPI, # Control the resolution of the plot, if it is high the figure storage size can be really large
        facecolor='w', # The background color
        edgecolor='k') # The edge color
    grid = plt.GridSpec(3, 1, wspace=0.4, hspace=HSPACE) # Seperate a figure into a matrix of 3x1, with 0.4 times width-space and 0.5 times height-space
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE) # This can addjust some specification of the plots above.
    ax1 = plt.subplot(grid[0:2, :]) # This is like python matrix slicing, so that you don't need to constrain yourself using only one cell for a plot
    # ax1 = plt.subplot(211)
    ax1.set_xlabel('Phase-I Observation Index', size=label_size) # Set xlabel. I think font can be changed here. Here is link https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html
    ax1.set_ylabel(
        'EWMA Univariate\nScore Component',
        color='b',
        size=label_size) # Set ylabel
    ax1.get_yaxis().set_label_coords(YLAB_XPOS, 0.5) # Set the limits for the axis.
    ax1.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE) # Set the tick font parameters
    ax1.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

    for i, (_, lstyle) in enumerate(line_style.items()):
        if i < len(col_ls):
            ax1.plot(np.arange(ewmaI.shape[0]),
                     ewmaI[:,col_ls[i]] * sign_factor,
                     color=line_colors[i],
                     linestyle=lstyle,
                     lw=line_width,
                     ms=marker_size,
                     label=label_ls[i])
            if thre_flag:
                ax1.plot(np.arange(ewmaI.shape[0]),
                        np.repeat(ucl[col_ls[i]] * sign_factor,
                                ewmaI.shape[0]),
                        color=line_colors[i],
                        linestyle=lstyle,
                        lw=line_width,
                        ms=marker_size)
                ax1.plot(np.arange(ewmaI.shape[0]),
                        np.repeat(lcl[col_ls[i]] * sign_factor,
                                ewmaI.shape[0]),
                        color=line_colors[i],
                        linestyle=lstyle,
                        lw=line_width,
                        ms=marker_size)
    ax1.plot(
        np.arange(ewmaI.shape[0]),
        np.repeat(0,ewmaI.shape[0]),
        'b--',
        lw=line_width *
        0.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel(
        'MEWMA Multivariate\nScore Vector',
        color='k',
        size=label_size)
    ax2.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
    ax2.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    ax2.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

    ax2.plot(
        np.arange(len(t2ewmaI)),
        t2ewmaI,
        'k-',
        lw=line_width *0.7,
        label='$T^2$')
    if thre_flag:
        ax2.plot(
            np.arange(len(t2ewmaI)),
            np.repeat(ucl5,len(t2ewmaI)),
            'k-',
            lw=line_width *0.7)
    # ax2.plot(np.arange(len(t2ewmaI)), np.repeat(0, len(t2ewmaI)), 'k-')

    # tmp = np.min(ewmaI[:,col_ls])
    # for i in np.arange(time_stamp[0], time_stamp[idx_PII-1]+1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     plt.text(j, tmp+0.2, time_stamp[j],rotation=45)

    ax1 = plt.subplot(grid[2, :])
    ax1.set_xlabel('Phase-II Observation Index', size=label_size)
    ax1.set_ylabel(
        'EWMA Univariate\nScore Component',
        color='b',
        size=label_size)
    ax1.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
    ax1.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    if log_flag:
        ax1.set_yscale('log')
    ax1.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

    for i, (_, lstyle) in enumerate(line_style.items()):
        if i < len(col_ls):
            ax1.plot(ewmaI.shape[0] + np.arange(ewmaII.shape[0]),
                     ewmaII[:,col_ls[i]] * sign_factor,
                     color=line_colors[i],
                     linestyle=lstyle,
                     lw=line_width,
                     ms=marker_size,
                     label=label_ls[i])
            if thre_flag:
                ax1.plot(ewmaI.shape[0] + np.arange(ewmaII.shape[0]),
                        np.repeat(ucl[col_ls[i]],
                                ewmaII.shape[0]) * sign_factor,
                        color=line_colors[i],
                        linestyle=lstyle,
                        lw=line_width,
                        ms=marker_size)
                ax1.plot(ewmaI.shape[0] + np.arange(ewmaII.shape[0]),
                        np.repeat(lcl[col_ls[i]],
                                ewmaII.shape[0]) * sign_factor,
                        color=line_colors[i],
                        linestyle=lstyle,
                        lw=line_width,
                        ms=marker_size)
    ax1.plot(
        ewmaI.shape[0] + np.arange(ewmaII.shape[0]),
        np.repeat(0, ewmaII.shape[0]),
        'b--',
        lw=line_width * 0.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel(
        'MEWMA Multivariate\nScore Vector',
        color='k',
        size=label_size)
    ax2.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
    ax2.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    if log_flag:
        ax2.set_yscale('log')
    ax2.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

    ax2.plot(
        ewmaI.shape[0] + np.arange(len(t2ewmaII)),
        t2ewmaII,
        'k-',
        lw=line_width * 0.7,
        label='$T^2$')
    if thre_flag:
        ax2.plot(
            ewmaI.shape[0] + np.arange(len(t2ewmaII)),
            np.repeat(ucl5, len(t2ewmaII)),
            'k-',
            lw=line_width * 0.7)
    ax2.plot(
        ewmaI.shape[0] + np.arange(len(t2ewmaII)),
        np.repeat(0, len(t2ewmaII)),
        'k-',
        lw=line_width * 0.7)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines1+lines2, labels1+labels2, loc=0)
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper left',
        bbox_to_anchor=(0, 0.95),
        prop={'size': label_size},
        ncol=2)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # tmp = np.min(ewmaII)
    # for i in np.arange(time_stamp[idx_PII], max(time_stamp)+1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     plt.text(j - idx_PII, tmp+0.8, time_stamp[j],rotation=45)
    if sign_factor == 1:
        plt.savefig(os.path.join(folder_path, 'pos_' + fig_name), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(folder_path, 'neg_' + fig_name), bbox_inches='tight')
    # plt.show()
    plt.close()


def PlotHotellingCC_Single_Line(
        t2ewmaI,
        t2ewmaII,
        ewmaI,
        ewmaII,
        col_ls,
        label_ls,
        alarm_level,
        time_stamp,
        N_PIIs,
        folder_path,
        fig_name,
        time_step,
        xlabel_name='Phase-I&-II Observation Index',
        date_index_flag=False,
        to_decode=False,
        thre_flag=True,
        sign_factor=1,
        log_flag=False,
        plot_row=6):
    """ Based on Phase-I&-II ewma hotelling time_stamp to plot control
        chart. Plot the Hotelling T2 with a single coordinate together.
    """

    plot_col = math.ceil(len(col_ls)/plot_row)
    # Fill in those empty lines to make sure that all subplots are filled.
    num_line_diff = plot_col*plot_row - len(col_ls)
    ewmaI = np.concatenate((ewmaI, np.zeros((ewmaI.shape[0], num_line_diff))), axis=1)
    ewmaII = np.concatenate((ewmaII, np.zeros((ewmaII.shape[0], num_line_diff))), axis=1)
    # print(col_ls)
    col_ls = col_ls + list(range(col_ls[-1]+1, col_ls[-1]+1+num_line_diff))
    label_ls = label_ls + ['empty']*num_line_diff

    # Calculate upper control limits
    ucl = np.percentile(ewmaI, (100 + alarm_level) / 2, axis=0)
    lcl = np.percentile(ewmaI, (100 - alarm_level) / 2, axis=0)
    ucl5 = np.percentile(t2ewmaI, alarm_level)

    label_size = int(0.75*LAB_SIZE)
    line_width = 1.5
    marker_size = 3
    plt.figure(
        num=None,
        figsize=(15*plot_col, 0.7 * ONE_FIG_HEI * (1+plot_row)),
        dpi=DPI,
        facecolor='w',
        edgecolor='k')
    grid = plt.GridSpec(plot_row + 1, plot_col, wspace=0.2, hspace=HSPACE)
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE)

    for plot_col_idx in range(plot_col):
        tmp_col_ls = col_ls[plot_col_idx*plot_row:min((plot_col_idx+1)*plot_row, len(col_ls))]
        tmp_label_ls = label_ls[plot_col_idx*plot_row:min((plot_col_idx+1)*plot_row, len(col_ls))]
        # print(plot_col_idx, num_line_diff, len(tmp_col_ls), len(tmp_label_ls))
        
        # Plot Phase-I of the Hotelling T2 for the score function
        ax0 = plt.subplot(grid[0, plot_col_idx])
        # ax1 = plt.subplot(211)
        # ax0.set_xlabel('Phase-I&-II Observation Index', size=label_size)
        ax0.set_ylabel(
            'MEWMA Multivariate\nScore Vector',
            color='k',
            size=label_size)
        ax0.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
        t2ewmaI_II = np.hstack((t2ewmaI, t2ewmaII))
        n_sample = len(t2ewmaI_II)
        ax0.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
        if log_flag:
            ax0.set_yscale('log')
        ax0.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

        ax0.plot(
            np.arange(n_sample),
            t2ewmaI_II,
            'k--',
            lw=line_width * 0.7,
            label='Score Function\n(Hotelling $T^2$ of EWMA)')
        if thre_flag:
            ax0.plot(
                np.arange(n_sample),
                np.repeat(ucl5, n_sample),
                'k--',
                lw=line_width * 0.7)
        ax0.plot(
            np.arange(n_sample),
            np.repeat(0, n_sample),
            'k--',
            lw=line_width * 0.7)
        ax0.axvline(x=len(t2ewmaI)) # Separation b/w Phase-I&-II
        for idx in len(t2ewmaI) + np.cumsum(N_PIIs[:-1]):
            ax0.axvline(x=idx, color='g') # Separation b/w different stages of concept drift

        if time_stamp.shape[0]>0 and date_index_flag:
            set_xticks_xticklabels(ax0, time_stamp, time_step, label_size=label_size, rotation=ROTATION, to_decode=to_decode)

        lines0, labels0 = ax0.get_legend_handles_labels()

        ax0.legend(
            lines0,
            labels0,
            loc='upper left',
            bbox_to_anchor=(0, 0.95),
            prop={'size': label_size},
            ncol=2)

        for idx, (col_idx, col_label) in enumerate(zip(tmp_col_ls, tmp_label_ls)):
            ax1 = plt.subplot(grid[idx + 1, plot_col_idx])
            if idx == plot_row-1:
                ax1.get_xaxis().set_label_coords(0.5, XLAB_YPOS)
                ax1.set_xlabel(xlabel_name, size=1.5*label_size)
            ax1.set_ylabel('EWMA Univariate\nScore Component', color='b', size=label_size)
            ax1.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
            ewma_col_I_II = np.hstack((ewmaI[:, col_idx], ewmaII[:, col_idx]))
            ax1.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
            if log_flag:
                ax1.set_yscale('log')
            ax1.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
            # ax1.set_xlim(ax0.get_xlim())

            ax1.plot(np.arange(n_sample),
                    ewma_col_I_II * sign_factor,
                    color=line_colors[idx],
                    linestyle=list(line_style.items())[idx][1],
                    lw=line_width,
                    ms=marker_size,
                    label=col_label)
            if thre_flag:
                ax1.plot(
                    np.arange(n_sample),
                    np.repeat(ucl[col_idx] * sign_factor, n_sample),
                    color=line_colors[idx],
                    linestyle=list(line_style.items())[idx][1],
                    lw=line_width,
                    ms=marker_size)
                ax1.plot(
                    np.arange(n_sample),
                    np.repeat(lcl[col_idx] * sign_factor, n_sample),
                    color=line_colors[idx],
                    linestyle=list(line_style.items())[idx][1],
                    lw=line_width,
                    ms=marker_size)
                if lcl[col_idx]< 0 < ucl[col_idx]:
                    ax1.plot(
                        np.arange(n_sample),
                        np.repeat(0, n_sample),
                        'b--',
                        lw=line_width * 0.5)
            ax1.axvline(x=ewmaI.shape[0]) # Separation b/w Phase-I&-II
            for idx in ewmaI.shape[0] + np.cumsum(N_PIIs[:-1]):
                ax1.axvline(x=idx, color='g') # Separation b/w different stages of concept drift

            if time_stamp.shape[0]>0 and date_index_flag:
                set_xticks_xticklabels(ax1, time_stamp, time_step, label_size=label_size, rotation=ROTATION, to_decode=to_decode)

            lines1, labels1 = ax1.get_legend_handles_labels()
            # ax2.legend(lines1+lines2, labels1+labels2, loc=0)
            ax1.legend(
                lines1,
                labels1,
                loc='upper left',
                bbox_to_anchor=(0, 0.95),
                prop={'size': label_size},
                ncol=2)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # tmp = np.min(ewmaII)
    # for i in np.arange(time_stamp[idx_PII], max(time_stamp)+1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     plt.text(j - idx_PII, tmp+0.8, time_stamp[j],rotation=45)
    plt.tight_layout()
    if sign_factor == 1:
        plt.savefig(os.path.join(folder_path, 'pos_' + fig_name), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(folder_path, 'neg_' + fig_name), bbox_inches='tight')
    # plt.show()
    plt.close()


def PlotHotellingCC_PII_Multi_Lines(
        t2ewmaI,
        t2ewmaII,
        ewmaI,
        ewmaII,
        col_ls,
        label_ls,
        alarm_level,
        time_stamp,
        N_PIIs,
        folder_path,
        fig_name,
        time_step,
        thre_flag=True,
        sign_factor=1,
        log_flag=False):
    """Based on Phase-I&-II ewma hotelling time_stamp to plot control chart."""
    ucl = np.percentile(ewmaI, (100 + alarm_level) / 2, axis=0)
    lcl = np.percentile(ewmaI, (100 - alarm_level) / 2, axis=0)
    ucl5 = np.percentile(t2ewmaI, alarm_level)

    label_size = int(0.75*LAB_SIZE)
    line_width = 1.5
    marker_size = 3
    plt.figure(
        num=None,
        figsize=(10, 0.8 * ONE_FIG_HEI * (1 + len(col_ls))),
        dpi=DPI,
        facecolor='w',
        edgecolor='k')
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE)

    ax1 = plt.subplot(111)
    ax1.set_xlabel('Phase-II Observation Index', size=label_size)
    ax1.set_ylabel(
        'EWMA Univariate\nScore Component',
        color='b',
        size=label_size)
    ax1.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
    ax1.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    if log_flag:
        ax1.set_yscale('log')
    ax1.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

    for i, (_, lstyle) in enumerate(line_style.items()):
        if i < len(col_ls):
            ax1.plot(np.arange(ewmaII.shape[0]),
                     ewmaII[:, col_ls[i]] * sign_factor,
                     color=line_colors[i],
                     linestyle=lstyle,
                     lw=line_width,
                     ms=marker_size,
                     label=label_ls[i])
            if thre_flag:
                ax1.plot(np.arange(ewmaII.shape[0]),
                        np.repeat(ucl[col_ls[i]] * sign_factor, ewmaII.shape[0]),
                        color=line_colors[i],
                        linestyle=lstyle,
                        lw=line_width,
                        ms=marker_size)
                ax1.plot(np.arange(ewmaII.shape[0]),
                        np.repeat(lcl[col_ls[i]] * sign_factor, ewmaII.shape[0]),
                        color=line_colors[i],
                        linestyle=lstyle,
                        lw=line_width,
                        ms=marker_size)
    ax1.plot(
        np.arange(ewmaII.shape[0]),
        np.repeat(0, ewmaII.shape[0]),
        'k--',
        lw=line_width * 0.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel(
        'MEWMA Multivariate\nScore Vector',
        color='k',
        size=label_size)
    ax2.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
    ax2.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
    if log_flag:
        ax2.set_yscale('log')
    ax2.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

    ax2.plot(
        np.arange(len(t2ewmaII)),
        t2ewmaII,
        'k-',
        lw=line_width * 0.7,
        label='Score Function\n(Hotelling $T^2$ of EWMA)')
    if thre_flag:
        ax2.plot(
            np.arange(len(t2ewmaII)),
            np.repeat(ucl5, len(t2ewmaII)),
            'k-',
            lw=line_width * 0.7)
    ax2.plot(
        np.arange(len(t2ewmaII)),
        np.repeat(0, len(t2ewmaII)),
        'k-',
        lw=line_width * 0.7)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines1+lines2, labels1+labels2, loc=0)
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper left',
        bbox_to_anchor=(0, 0.95),
        prop={'size': label_size},
        ncol=2)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # tmp = np.min(ewmaII)
    # for i in np.arange(time_stamp[idx_PII], max(time_stamp)+1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     plt.text(j - idx_PII, tmp+0.8, time_stamp[j],rotation=45)
    plt.tight_layout()
    if sign_factor == 1:
        plt.savefig(os.path.join(folder_path, 'PII_pos_' + fig_name), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(folder_path, 'PII_neg_' + fig_name), bbox_inches='tight')
    # plt.show()
    plt.close()


def PlotHotellingCC_PII_Single_Line(
        t2ewmaI,
        t2ewmaII,
        ewmaI,
        ewmaII,
        col_ls,
        label_ls,
        alarm_level,
        time_stamp,
        N_PIIs,
        folder_path,
        fig_name,
        time_step,
        xlabel_name='Phase-II Observation Index',
        date_index_flag=False,
        to_decode=False,
        thre_flag=True,
        sign_factor=1,
        log_flag=False,
        plot_row=6):
    """Based on Phase-I&-II ewma hotelling time_stamp to plot control chart."""
    plot_col = math.ceil(len(col_ls)/plot_row)
    # Fill in those empty lines to make sure that all subplots are filled.
    num_line_diff = plot_col*plot_row - len(col_ls)
    ewmaI = np.concatenate((ewmaI, np.zeros((ewmaI.shape[0], num_line_diff))),axis=1)
    ewmaII = np.concatenate((ewmaII, np.zeros((ewmaII.shape[0], num_line_diff))),axis=1)
    col_ls = col_ls + list(range(col_ls[-1]+1, col_ls[-1]+1+num_line_diff))
    label_ls = label_ls + ['empty']*num_line_diff

    # Calculate upper control limits
    ucl = np.percentile(ewmaI, (100 + alarm_level) / 2, axis=0)
    lcl = np.percentile(ewmaI, (100 - alarm_level) / 2, axis=0)
    ucl5 = np.percentile(t2ewmaI, alarm_level)

    label_size = int(0.75*LAB_SIZE)
    line_width = 1.5
    marker_size = 3
    plt.figure(
        num=None,
        figsize=(12*plot_col, 0.7 * ONE_FIG_HEI * (1 + plot_row)),
        dpi=DPI,
        facecolor='w',
        edgecolor='k')
    grid = plt.GridSpec(plot_row + 1, plot_col, wspace=0.2, hspace=HSPACE)
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=HSPACE)

    for plot_col_idx in range(plot_col):
        tmp_col_ls = col_ls[plot_col_idx*plot_row:min((plot_col_idx+1)*plot_row, len(col_ls))]
        tmp_label_ls = label_ls[plot_col_idx*plot_row:min((plot_col_idx+1)*plot_row, len(col_ls))]

        # Plot Phase-II of the Hotelling T2 for the score function
        ax01 = plt.subplot(grid[0, plot_col_idx])
        # ax1 = plt.subplot(211)
        # ax01.set_xlabel('Phase-II Observation Index', size=label_size)
        ax01.set_ylabel(
            'MEWMA Multivariate\nScore Vector',
            color='k',
            size=label_size)
        ax01.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
        ax01.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
        if log_flag:
            ax01.set_yscale('log')
        ax01.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
        n_sample_PI = ewmaI.shape[0]
        n_sample_PII = ewmaII.shape[0]
        ax01.plot(
            # n_sample_PI + np.arange(n_sample_PII),
            n_sample_PI+np.arange(n_sample_PII),
            t2ewmaII,
            'k--',
            lw=line_width * 0.7,
            label='Score Function\n(Hotelling $T^2$ of EWMA)')
        if thre_flag:
            ax01.plot(
                # n_sample_PI + np.arange(n_sample_PII),
                n_sample_PI+np.arange(n_sample_PII),
                np.repeat(ucl5, n_sample_PII),
                'k--',
                lw=line_width * 0.7)
        ax01.plot(
            # n_sample_PI + np.arange(n_sample_PII),
            n_sample_PI+np.arange(n_sample_PII),
            np.repeat(0, n_sample_PII),
            'k--',
            lw=line_width * 0.7)
        for idx in len(t2ewmaI) + np.cumsum(N_PIIs[:-1]):
            ax01.axvline(x=idx, color='g') # Separation b/w different stages of concept drift
        
        if time_stamp.shape[0]>0 and date_index_flag:
            set_xticks_xticklabels(ax01, time_stamp, time_step, label_size=label_size, rotation=ROTATION, to_decode=to_decode)
                    
        lines01, labels01 = ax01.get_legend_handles_labels()

        ax01.legend(
            lines01,
            labels01,
            loc='upper left',
            bbox_to_anchor=(0, 0.95),
            prop={'size': label_size},
            ncol=2)

        for idx, (col_idx, col_label) in enumerate(zip(tmp_col_ls, tmp_label_ls)):
            ax1 = plt.subplot(grid[idx + 1, plot_col_idx])
            if idx == plot_row-1:
                ax1.get_xaxis().set_label_coords(0.5, XLAB_YPOS)
                ax1.set_xlabel(xlabel_name, size=1.5*label_size)
            ax1.set_ylabel(
                'EWMA Univariate\nScore Component',
                color='b',
                size=label_size)
            ax1.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
            ax1.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
            if log_flag:
                ax1.set_yscale('log')
            ax1.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

            ax1.plot(
                # n_sample_PI + np.arange(n_sample_PII),
                n_sample_PI+np.arange(n_sample_PII),
                ewmaII[:,col_idx] * sign_factor,
                color=line_colors[idx],
                linestyle=list(line_style.items())[idx][1],
                lw=line_width,
                ms=marker_size,
                label=col_label)
            if thre_flag:
                ax1.plot(
                    # n_sample_PI+np.arange(n_sample_PII),
                    n_sample_PI+np.arange(n_sample_PII),
                    np.repeat(ucl[col_idx] * sign_factor, n_sample_PII),
                    color=line_colors[idx],
                    linestyle=list(line_style.items())[idx][1],
                    lw=line_width,
                    ms=marker_size)
                ax1.plot(
                    # n_sample_PI+np.arange(n_sample_PII),
                    n_sample_PI+np.arange(n_sample_PII),
                    np.repeat(lcl[col_idx] * sign_factor, n_sample_PII),
                    color=line_colors[idx],
                    linestyle=list(line_style.items())[idx][1],
                    lw=line_width,
                    ms=marker_size)
                if lcl[col_idx] < 0 < ucl[col_idx]:
                    ax1.plot(
                        # n_sample_PI+np.arange(n_sample_PII),
                        n_sample_PI+np.arange(n_sample_PII),
                        np.repeat(0, n_sample_PII),
                        'b--',
                        lw=line_width * 0.5)
            for idx in ewmaI.shape[0] + np.cumsum(N_PIIs[:-1]):
                ax1.axvline(x=idx, color='g') # Separation b/w different stages of concept drift
            
            if time_stamp.shape[0]>0 and date_index_flag:
                set_xticks_xticklabels(ax1, time_stamp, time_step, label_size=label_size, rotation=ROTATION, to_decode=to_decode)

            lines1, labels1 = ax1.get_legend_handles_labels()

            # ax2.legend(lines1+lines2, labels1+labels2, loc=0)
            ax1.legend(
                lines1,
                labels1,
                loc='upper left',
                bbox_to_anchor=(0, 0.95),
                prop={'size': label_size},
                ncol=2)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # tmp = np.min(ewmaII)
    # for i in np.arange(time_stamp[idx_PII], max(time_stamp)+1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     plt.text(j - idx_PII, tmp+0.8, time_stamp[j],rotation=45)
    plt.tight_layout()
    if sign_factor == 1:
        plt.savefig(os.path.join(folder_path, 'PII_pos_' + fig_name), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(folder_path, 'PII_neg_' + fig_name), bbox_inches='tight')
    # plt.show()
    plt.close()


def PlotHotellingCC_PI_Single_Line(
        t2ewmaI,
        ewmaI,
        col_ls,
        label_ls,
        alarm_level,
        time_stamp,
        folder_path,
        fig_name,
        time_step,
        xlabel_name='Phase-I Observation Index',
        date_index_flag=False,
        to_decode=False,
        thre_flag=True,
        sign_factor=1,
        log_flag=False,
        plot_row=6):
    """Based on Phase-I ewma hotelling time_stamp to plot control chart."""
    plot_col = math.ceil(len(col_ls)/plot_row)
    # Fill in those empty lines to make sure that all subplots are filled.
    num_line_diff = plot_col*plot_row - len(col_ls)
    ewmaI = np.concatenate((ewmaI, np.zeros((ewmaI.shape[0], num_line_diff))),axis=1)
    col_ls = col_ls + list(range(col_ls[-1]+1, col_ls[-1]+1+num_line_diff))
    label_ls = label_ls + ['empty']*num_line_diff

    # Calculate upper control limits
    ucl = np.percentile(ewmaI, (100 + alarm_level) / 2, axis=0)
    lcl = np.percentile(ewmaI, (100 - alarm_level) / 2, axis=0)
    ucl5 = np.percentile(t2ewmaI, alarm_level)

    label_size = int(0.75*LAB_SIZE)
    line_width = 1.5
    marker_size = 3
    plt.figure(
        num=None,
        figsize=(10*plot_col, 0.7 * ONE_FIG_HEI * (1 + plot_row)),
        dpi=DPI,
        facecolor='w',
        edgecolor='k')
    grid = plt.GridSpec(plot_row + 1, plot_col, wspace=0.2, hspace=HSPACE)
    plt.subplots_adjust(top=AX_TOP, bottom=AX_BOT, left=AX_LEFT, right=AX_RIGHT, hspace=1.5*HSPACE)

    for plot_col_idx in range(plot_col):
        tmp_col_ls = col_ls[plot_col_idx*plot_row:min((plot_col_idx+1)*plot_row, len(col_ls))]
        tmp_label_ls = label_ls[plot_col_idx*plot_row:min((plot_col_idx+1)*plot_row, len(col_ls))]

        # Plot Phase-II of the Hotelling T2 for the score function
        ax01 = plt.subplot(grid[0, plot_col_idx])
        # ax1 = plt.subplot(211)
        # ax01.set_xlabel('Phase-II Observation Index', size=label_size)
        ax01.set_ylabel(
            'MEWMA Multivariate\nScore Vector',
            color='k',
            size=label_size)
        ax01.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
        ax01.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
        if log_flag:
            ax01.set_yscale('log')
        ax01.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
        n_sample_PI = ewmaI.shape[0]
        ax01.plot(
            np.arange(n_sample_PI),
            t2ewmaI,
            'k--',
            lw=line_width * 0.7,
            label='Score Function\n(Hotelling $T^2$ of EWMA)')
        if thre_flag:
            ax01.plot(
                np.arange(n_sample_PI),
                np.repeat(ucl5, n_sample_PI),
                'k--',
                lw=line_width * 0.7)
        ax01.plot(
            np.arange(n_sample_PI),
            np.repeat(0, n_sample_PI),
            'k--',
            lw=line_width * 0.7)
        # for idx in len(t2ewmaI) + np.cumsum(N_PIIs[:-1]):
        #     ax01.axvline(x=idx, color='g') # Separation b/w different stages of concept drift

        if time_stamp.shape[0]>0 and date_index_flag:
            set_xticks_xticklabels(ax01, time_stamp, time_step, label_size=label_size, rotation=ROTATION, to_decode=to_decode)

        lines01, labels01 = ax01.get_legend_handles_labels()

        ax01.legend(
            lines01,
            labels01,
            loc='upper left',
            bbox_to_anchor=(0, 0.95),
            prop={'size': label_size},
            ncol=2)

        for idx, (col_idx, col_label) in enumerate(zip(tmp_col_ls, tmp_label_ls)):
            ax1 = plt.subplot(grid[idx + 1, plot_col_idx])
            if idx == plot_row-1:
                ax1.get_xaxis().set_label_coords(0.5, XLAB_YPOS)
                ax1.set_xlabel(xlabel_name, size=1.5*label_size)
            ax1.set_ylabel(
                'EWMA Univariate\nScore Component',
                color='b',
                size=label_size)
            ax1.get_yaxis().set_label_coords(YLAB_XPOS, 0.5)
            ax1.xaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)
            if log_flag:
                ax1.set_yscale('log')
            ax1.yaxis.set_tick_params(labelsize=label_size * AX_LAB_SCALE)

            ax1.plot(np.arange(n_sample_PI),
                ewmaI[:,col_idx] * sign_factor,
                color=line_colors[idx],
                linestyle=list(line_style.items())[idx][1],
                lw=line_width,
                ms=marker_size,
                label=col_label)
            if thre_flag:
                ax1.plot(np.arange(n_sample_PI),
                    np.repeat(ucl[col_idx] * sign_factor, n_sample_PI),
                    color=line_colors[idx],
                    linestyle=list(line_style.items())[idx][1],
                    lw=line_width,
                    ms=marker_size)
                ax1.plot(np.arange(n_sample_PI),
                    np.repeat(lcl[col_idx] * sign_factor, n_sample_PI),
                    color=line_colors[idx],
                    linestyle=list(line_style.items())[idx][1],
                    lw=line_width,
                    ms=marker_size)
                if lcl[col_idx] < 0 < ucl[col_idx]:
                    ax1.plot(np.arange(n_sample_PI),
                        np.repeat(0, n_sample_PI),
                        'b--',
                        lw=line_width * 0.5)
            # for idx in ewmaI.shape[0] + np.cumsum(N_PIIs[:-1]):
            #     ax1.axvline(x=idx, color='g') # Separation b/w different stages of concept drift

            if time_stamp.shape[0]>0 and date_index_flag:
                set_xticks_xticklabels(ax1, time_stamp, time_step, label_size=label_size, rotation=ROTATION, to_decode=to_decode)

            lines1, labels1 = ax1.get_legend_handles_labels()

            # ax2.legend(lines1+lines2, labels1+labels2, loc=0)
            ax1.legend(
                lines1,
                labels1,
                loc='upper left',
                bbox_to_anchor=(0, 0.95),
                prop={'size': label_size},
                ncol=2)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # tmp = np.min(ewmaII)
    # for i in np.arange(time_stamp[idx_PII], max(time_stamp)+1, time_step):
    #     j = np.argmax(time_stamp >= i)
    #     plt.text(j - idx_PII, tmp+0.8, time_stamp[j],rotation=45)
    plt.tight_layout()
    if sign_factor == 1:
        plt.savefig(os.path.join(folder_path, 'PI_pos_' + fig_name), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(folder_path, 'PI_neg_' + fig_name), bbox_inches='tight')
    # plt.show()
    plt.close()
