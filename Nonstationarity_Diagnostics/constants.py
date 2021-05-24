import os
import matplotlib.pyplot as plt

DATA_ROOT_DIR = '/projects/p30309/neurips2021/' # quest
# DATA_ROOT_DIR = '/projects/p31200/scratch/' # questzst
# DATA_ROOT_DIR = '/projects/p30618/neurips2021/' # questzst
# DATA_ROOT_DIR = '/home/ghhgkz/scratch/' # crunch
RES_ROOT_DIR = DATA_ROOT_DIR
CODE_ROOT_DIR = DATA_ROOT_DIR

# # comet
# # COMET_SSD_SCRATCH_PATH = os.path.join('/scratch/ghhgkz/', os.environ['SLURM_JOBID'])
# COMET_SSD_SCRATCH_PATH = '/oasis/scratch/comet/ghhgkz/temp_project/'
# DATA_ROOT_DIR = COMET_SSD_SCRATCH_PATH # comet data root director; only accessible after job starts. Fastest
# RES_ROOT_DIR = '/oasis/scratch/comet/ghhgkz/temp_project/' # Store results. Fast.
# CODE_ROOT_DIR = '/home/ghhgkz/scratch/' # Store code. Slow.

OFFSET = 0 # 300 for bike sharing data set  # 200  # offset of ignoring plot in phase-I
# COMP_SCORE_OFFSET = 5000
RCOND_NUM = 1E-4
LAB_SIZE = 45 # 30 for all data sets except credit risk data sets, 25.  
AX_LAB_SCALE = 0.55 # 0.75 for all data sets except credit risk data sets, .55. 
HSPACE = 0.4
WSPACE = 0.3
ONE_FIG_HEI = 5
ROTATION = 45 # 0 for all data sets except credit risk data sets, 45.
AX_TOP = 0.95
AX_BOT = 0.05
AX_LEFT = 0.05
AX_RIGHT = 0.95
UTIL_TASK_NJOBS = 20
YLAB_XPOS = -0.07 # -0.10 for all data sets except credit risk data sets, -0.05. 
XLAB_YPOS = -0.26
YR_TICK_MARGIN = 0.22 # 0.15 for all data sets except credit risk data sets, 0.22. 
YR_MOD = 100

DPI = 100

YWEI = 7.5 # Upsample weights for y=1 in logistic regression. 0 for all data sets except credit risk data sets, 7.5.

# https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html
CMAP = plt.cm.plasma # plt.cm.Spectral # plt.cm.bwr
GRAY_CMAP = plt.cm.gray
PT_SIZE = 3.0
ALPHA = 0.35
TICK_NUM = 4
LAB_PAD = 32
CB_PAD = 0.1
CB_SHRINK = 0.9
N_JOBS = UTIL_TASK_NJOBS
ELEV = 0
AZIM = 105
ONLY_PLOT = 0
SLEEP_TIME = 10
