import os
import tensorflow as tf
from tensorflow.python.client import device_lib
# print to see if I can access GPU.
print(device_lib.list_local_devices())

IMG_SIZE = 256 # 256*1.2
RND_SEED = 129 # 123

ROOT_DIR = '/projects/p30309/neurips2021' # quest
CODE_ROOT_DIR = ROOT_DIR
# PLT_ROOT_DIR = '/projects/p30309/neurips2021/Experiments/UHCSA_seg_examples'
PLT_ROOT_DIR = '/projects/p30309/neurips2021/Experiments/Dendrites_seg_examples'
# PLT_ROOT_DIR = '/projects/p30309/neurips2021/Experiments/Brodatz_seg_examples'

DTYPE_FLOAT_STR = 'float32'
DTYPE_FLOAT = tf.float32
DTYPE_INT = tf.int32
BATCH_SIZE = 32
VAL_BATCH_RATIO = 1 # Don't change this one. Won't be faster if use larger than 1
VAL_RATIO = 8 # Save some time for validation.
N_THREADS = 30
PIPLINE_JOBS = 30
MID_FLOW_BLOCK_NUM = 8 # Default is 16
MAX_SCALE_RATIO = 0.0
COLRESET = '\033[0m'
FIG_TITLE_WID = 23

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_X_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
WEIGHTS_PATH_MOBILE_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"

ERROR_COL = 'xkcd:red'
ACCU_STEP_VAL_PLOT = 1 # The threhold for densely plotting validation plots, usually 1280 to prevent taking too much time in plotting in the early training
FACTOR_VAL_PLOT = 1 # The factor for densely plotting validation plots in normal epoch, usually 7 to prevent dense plotting in every epoch
