# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import math
import scipy.ndimage as ndimage
from seg_utils.generate_collages import *
from constants import *
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

#mean of imagenet dataset in RGB (notice it is RGB, not BGR)
IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=DTYPE_FLOAT)

# class UnetDataGenerator(object):
#     """Wrapper class around the new Tensorflows dataset pipeline for Brodatz dataset.

#     Requires Tensorflow >= version 1.12rc0
#     """

#     def __init__(self, img_folder, mode, sample_size, batch_size, segmentation_regions):
#         """Create a new ImageDataGenerator.

#         Recieves a path string to a text file, which consists of many lines,
#         where each line has first a path string to an image and seperated by
#         a space an integer, referring to the class number. Using this data,
#         this class will create TensrFlow datasets, that can be used to train
#         e.g. a convolutional neural network.

#         Args:
#             txt_file: Path to the text file.
#             mode: Either 'training' or 'validation'. Depending on this value,
#                 different parsing functions will be used.
#             batch_size: Number of images per batch.
#             num_classes: Number of classes in the dataset.

#         Raises:
#             ValueError: If an invalid mode is passed.

#         """
#         self.img_folder = os.path.expanduser(img_folder)
#         self.all_textures = generate_texture(self.img_folder)
#         self.num_classes = self.all_textures.shape[0]
#         self.img_size = self.all_textures.shape[1]
#         self.segmentation_regions = segmentation_regions 

#         # generate anchor points for each collages
#         self.n_points = np.random.randint(2, self.segmentation_regions+1, size=sample_size)

#         # convert lists to TF tensor
#         self.n_points = tf.convert_to_tensor(self.n_points, dtype=DTYPE_INT)

#         # create dataset
#         data = tf.data.Dataset.from_tensor_slices((self.n_points, ))

#         # print(list(data.as_numpy_iterator())[:10])

#         # distinguish between train/infer. when calling the parsing functions
#         if mode == 'training':
#             # Without parallel calls, the training would be very slow on colab.
#             data = data.map(self._gen_function, num_parallel_calls=8)
#             # For small memory, the buffer_size needs to be small.
#             self.data = data.repeat().shuffle(5000, reshuffle_each_iteration=True).batch(batch_size).prefetch(20)
#         elif mode == 'inference':
#             data = data.map(self._gen_function, num_parallel_calls=8)
#             self.data = data.batch(batch_size).prefetch(2)
#         else:
#             raise ValueError("Invalid mode '%s'." % (mode))

#     def _gen_function(self, n_points):
#         """ Input parser for samples of the training set.
        
#             The image image of Brodatz dataset has 256*256 in size.
#         """
#         # When we use SparseCategoricalCrossentropy, we don't need one-hot coding.
#         # # convert label number into one-hot-encoding
#         # one_hot = tf.one_hot(label, self.num_classes)

#         img, label = generate_one_collage(self.all_textures, self.num_classes, self.img_size, 
#                                           segmentation_regions=self.segmentation_regions, n_points=n_points)
#         img = tf.convert_to_tensor(img, dtype=DTYPE_FLOAT)
#         label = tf.convert_to_tensor(label, dtype=DTYPE_INT)

#         """
#         Data augmentation comes here.
#         """
#         img_centered = tf.subtract(img, IMAGENET_MEAN)

#         # RGB -> BGR
#         # I think the reason that we need to change RGB to BGR is because that the pre-trained model is coming
#         # from Caffe framework. And Caffe processes image in BGR channels. So the AlexNet trained by Caffe has
#         # this order of channels.
#         # Refer to: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#         # and: https://github.com/BVLC/caffe/wiki/Image-Format:-BGR-not-RGB
#         # or: https://caffe2.ai/docs/tutorial-image-pre-processing.html
#         # Notice that OpenCV library (cv2) processes images in BGR order. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
#         #
#         # This can be confirmed in running validate_alexnet_on_imagent.py.
#         # The images are read using cv2. If we past in RGB image to AlexNet, the sea lion image has only 0.3041 as probability for correct class, 
#         # while when past in BGR, the probability is 0.9834. The reason is that sometimes the net use colors to get correct classification.
#         img_bgr = img_centered[:, :, ::-1]

#         return img_bgr, label

def gen_sample_wei(x, wei_map):
    return wei_map[x]

def gen_sample_wei_arr(lab_arr):
    uni_labs, uni_cnts = np.unique(lab_arr, return_counts=True)
    labs_wei = (uni_cnts/np.sum(uni_cnts))**(-1)/uni_labs.shape[0]
    wei_map = {}
    for lab, wei in zip(uni_labs, labs_wei):
        wei_map[lab] = wei
    sample_wei_arr = np.vectorize(gen_sample_wei)(lab_arr, wei_map)
    return sample_wei_arr


class GenSampleWei(Model):
    def __init__(self, lab_wei_map, input_shapes=(256, 256)):
        super(GenSampleWei, self).__init__(name='GenSampleWei')

        self.lab_wei_map = lab_wei_map
        self.num_classes = len(lab_wei_map)
        self.input_shapes = input_shapes
        self.wei_map = Conv2D(filters=1, kernel_size=1, strides=1, activation=None, use_bias=False, name='wei_map')

        # Initialize the model
        inputs = tf.keras.Input(shape=self.input_shapes, dtype=DTYPE_INT)
        outputs = self.call(inputs)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Assign weights for sample weight graph
        lab_wei_arr = np.zeros(shape=(self.num_classes,)) # Label follow the conventional naming order and number 
        for lab in range(self.num_classes):
            lab_wei_arr[lab] = lab_wei_map[lab]
        print(self.wei_map.trainable_variables)
        lab_wei_arr = np.reshape(lab_wei_arr, self.wei_map.trainable_variables[0].shape)
        self.wei_map.trainable_variables[0].assign(lab_wei_arr)
        print(self.wei_map.trainable_variables)
        self.wei_map.trainable = False
        print(self.wei_map.trainable_variables)

    def call(self, input_tensor):
        input_one_hot = tf.one_hot(indices=input_tensor, depth=len(self.lab_wei_map))
        return self.wei_map(input_one_hot)

    def get_config(self):
        config = super(GenSampleWei, self).get_config()
        config.update({'lab_wei_map': self.lab_wei_map, 
                       'num_classes': self.num_classes})
        return config


class HomoTextureDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline for Kylberg or Brodatz dataset.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, img_norm_flag=False, trfm_flag=False):
        """Create a new ImageDataGenerator.

        Receives a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.txt_file = txt_file
        self.num_classes = num_classes
        self.img_norm_flag = img_norm_flag
        self.trfm_flag = trfm_flag

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.img_paths)

        # convert lists to TF tensor
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.lab_paths = tf.convert_to_tensor(self.lab_paths, dtype=tf.string)
        self.lab_bd_paths = tf.convert_to_tensor(self.lab_bd_paths, dtype=tf.string)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.lab_paths, self.lab_bd_paths))

        # print(list(data.as_numpy_iterator())[:10])

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            # Without parallel calls, the training would be very slow on colab.
            data = data.map(self._parse_training, num_parallel_calls=PIPLINE_JOBS)
            # For small memory, the buffer_size needs to be small.
            self.data = data.repeat().shuffle(2000, reshuffle_each_iteration=True).batch(batch_size).prefetch(buffer_size=int(1.5*PIPLINE_JOBS)) #tf.data.experimental.AUTOTUNE would blow memory #.prefetch(6) https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
        elif mode == 'inference':
            data = data.map(self._parse_training, num_parallel_calls=PIPLINE_JOBS)
            self.data = data.batch(batch_size).prefetch(int(1.5*PIPLINE_JOBS))
        else:
            raise ValueError("Invalid mode '%s'." % (mode))


    def _read_txt_file(self):
        """ Read the content of the text file and store it into lists.

            For Kylberg dataset, each line of txt file is path, label, and the index of subimage.
        """
        self.img_paths = []
        self.lab_paths = []
        self.lab_bd_paths = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(',')
                self.img_paths.append(items[0])
                self.lab_paths.append(items[1])
                self.lab_bd_paths.append(items[2])


    def _parse_training(self, x_path, y_path, y_bd_path):
        """ Input parser for samples of the training set.
        
            The image of Kylberg dataset has 576*576 in size. We get a quadra-image of it.
        """
        # When we use SparseCategoricalCrossentropy, we don't need one-hot coding.
        # # convert label number into one-hot-encoding
        # one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        x_img_string = tf.io.read_file(x_path)
        y_img_string = tf.io.read_file(y_path)
        y_bd_img_string = tf.io.read_file(y_bd_path)
        x_img_decoded = tf.image.decode_png(x_img_string, channels=3)
        y_img_decoded = tf.image.decode_png(y_img_string, channels=1)
        y_bd_img_decoded = tf.image.decode_png(y_bd_img_string, channels=1)

        # It is important to let the data as tensor.
        x_img = tf.subtract(tf.cast(x_img_decoded, DTYPE_FLOAT), IMAGENET_MEAN)
        y_img = tf.cast(y_img_decoded, DTYPE_INT)
        y_bd_img = tf.cast(y_bd_img_decoded, DTYPE_INT)

        if self.img_norm_flag:
            x_img = tf.image.per_image_standardization(x_img)

        if self.trfm_flag:
            # Rotate
            num_rot90 = np.random.randint(low=0, high=4)
            x_img = tf.image.rot90(x_img, k=num_rot90)
            y_img = tf.image.rot90(y_img, k=num_rot90) # Image must be 3-dimensional.
            y_bd_img = tf.image.rot90(y_bd_img, k=num_rot90) # Image must be 3-dimensional.
            # Flip
            if np.random.randint(low=0, high=2):
                x_img = tf.image.flip_left_right(x_img)
                y_img = tf.image.flip_left_right(y_img)
                y_bd_img = tf.image.flip_left_right(y_bd_img)

        y_img = tf.squeeze(y_img, axis=-1)
        y_bd_img = tf.squeeze(y_bd_img, axis=-1)

        # RGB -> BGR
        # I think the reason that we need to change RGB to BGR is because that the pre-trained model is coming
        # from Caffe framework. And Caffe processes image in BGR channels. So the AlexNet trained by Caffe has
        # this order of channels.
        # Refer to: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        # and: https://github.com/BVLC/caffe/wiki/Image-Format:-BGR-not-RGB
        # or: https://caffe2.ai/docs/tutorial-image-pre-processing.html
        # Notice that OpenCV library (cv2) processes images in BGR order. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
        #
        # This can be confirmed in running validate_alexnet_on_imagent.py.
        # The images are read using cv2. If we past in RGB image to AlexNet, the sea lion image has only 0.3041 as probability for correct class, 
        # while when past in BGR, the probability is 0.9834. The reason is that sometimes the net use colors to get correct classification.
        # x_img_bgr = x_img_centered[:, :, ::-1]

        return x_img, y_img, y_bd_img

    def _parse_training_training(self, x_path, y_path, y_bd_path):
        x_img, y_img, y_bd_img = self._parse_training(x_path, y_path, y_bd_path)
        return x_img, y_img, y_bd_img, None 


class LabTextureDataGenerator(object):
    """ Wrapper class around the new Tensorflow dataset pipeline for pixel-wised labeled image dataset.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, val_batch_ratio=1, img_norm_flag=False, trfm_flag=False, sample_wei_flag=False, lab_wei_map=None, lab_wei_alp=1, img_size=IMG_SIZE):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.txt_file = txt_file
        self.num_classes = num_classes
        self.val_batch_ratio = val_batch_ratio
        self.img_norm_flag = img_norm_flag
        self.trfm_flag = trfm_flag
        self.sample_wei_flag = sample_wei_flag
        self.lab_wei_map = lab_wei_map
        self.lab_wei_alp = lab_wei_alp
        self.img_size = img_size

        # For generating rotation of image patches, we need padding for mirror boundary condition.
        self.paddings = tf.constant([[self.img_size-1, self.img_size-1], [self.img_size-1, self.img_size-1], [0, 0]])

        # Generate network for sample weight
        if sample_wei_flag:
            # self.wei_map = GenSampleWei(self.lab_wei_map)
            self.lab_wei_arr = np.zeros(shape=(self.num_classes,)) # Label follow the conventional naming order and number 
            for lab in range(self.num_classes):
                self.lab_wei_arr[lab] = self.lab_wei_map[lab]
            self.lab_wei_arr = self.lab_wei_arr**self.lab_wei_alp ## The power of weight of materials phases.
            print("The label weights are: {}.".format(self.lab_wei_arr))
            self.lab_wei_tensor = tf.convert_to_tensor(self.lab_wei_arr.reshape((1,1,-1,1)), dtype=DTYPE_FLOAT)

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.img_paths)

        # convert lists to TF tensor
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.lab_paths = tf.convert_to_tensor(self.lab_paths, dtype=tf.string)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.lab_paths))

        # print(list(data.as_numpy_iterator())[:10])

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            # Without parallel calls, the training would be very slow on colab.
            data = data.map(self._parse_training, num_parallel_calls=int(1.2*PIPLINE_JOBS)) # In the map the random number inside function won't really work.
            # For small memory, the buffer_size needs to be small.
            self.data = data.repeat().shuffle(2000, reshuffle_each_iteration=True).batch(batch_size).prefetch(buffer_size=int(1.5*PIPLINE_JOBS)) #tf.data.experimental.AUTOTUNE would blow memory #.prefetch(6) https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
        elif mode == 'inference': # For inference, we don't need rotation.
            data = data.map(self._parse_inference, num_parallel_calls=int(1.2*PIPLINE_JOBS))
            self.data = data.batch(val_batch_ratio*batch_size).prefetch(buffer_size=int(1.5*val_batch_ratio*PIPLINE_JOBS))
        else:
            raise ValueError("Invalid mode '%s'." % (mode))


    def _read_txt_file(self):
        """ Read the content of the text file and store it into lists.

            For Kylberg dataset, each line of txt file is path, label, and the index of subimage.
        """
        self.img_paths = []
        self.lab_paths = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(',')
                self.img_paths.append(items[0])
                self.lab_paths.append(items[1])

    def _rot_mirror(self, x_img, y_img):
        """ 
            This function is not used yet.

            rotate image with mirror boundary condition.

            x_img: input image tensor with three dimension
            y_img: input pixel-wised label tensor with three dimension
        """
        angles = tf.random.uniform(shape=[], minval=0, maxval=math.radians(360), dtype=DTYPE_FLOAT)
        
        def rot(img, angles, interp):
            img_ext = tf.pad(img, self.paddings, mode='REFLECT')
            img_ext_rot = tfa.image.rotate(img_ext, angles=angles, interpolation=interp)
            return tf.image.crop_to_bounding_box(img_ext_rot, offset_height=self.img_size-1, offset_width=self.img_size-1, target_height=self.img_size, target_width=self.img_size)

        x_img, y_img = rot(x_img, angles, 'BILINEAR'), rot(y_img, angles, 'BILINEAR')
        
        # Debug
        # tf.print("The random angle is:")
        # tf.print(angles)
        # rand_idx = tf.random.uniform(shape=[], minval=0, maxval=100000, dtype=DTYPE_INT)
        # tf.io.write_file(os.path.join('/projects/p30309/test_dir/', '{}_x'.format(rand_idx).replace('.','_')+'.png'), tf.image.encode_png(tf.cast((x_img-tf.reduce_min(x_img))/(tf.reduce_max(x_img)-tf.reduce_min(x_img))*255, dtype=tf.uint8)))
        # tf.io.write_file(os.path.join('/projects/p30309/test_dir/', '{}_y'.format(rand_idx).replace('.','_')+'.png'), tf.image.encode_png(tf.cast((y_img-tf.reduce_min(y_img))/(tf.reduce_max(y_img)-tf.reduce_min(y_img))*255, dtype=tf.uint8)))
        return x_img, y_img


    def _rot_90(self, x_img, y_img):
        """
            Only rotate integer value of 90 degrees.
            This would not have problem of interpolation.
        """
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=DTYPE_INT)
        x_img, y_img = tf.image.rot90(x_img, k=k), tf.image.rot90(y_img, k=k)

        return x_img, y_img

    def _rot_mirror_scipy(self, x_img, y_img):
        # degree = np.random.uniform(0,360)
        degree = np.random.randint(0,360) # Use int of degree
        # print("The rotation degree is {}.".format(degree))
        # By default an array of the same dtype as input will be created
        x_img = ndimage.rotate(input=x_img, angle=degree, axes=(1,0), reshape=False, order=3, mode='mirror', prefilter=True) # Bicubic interpolation
        y_img = ndimage.rotate(input=y_img, angle=degree, axes=(1,0), reshape=False, order=1, mode='mirror', prefilter=True) # Bilinear interpolation to prevent generating labels outside of range.
        # # For debugging
        # y_uniq = np.unique(y_img)
        # if np.any(y_uniq>3) or np.any(y_uniq<0):
        #     print("!!!@@@: The bicubic interploation on label image create labels outside of range:{}({},{})!!!".format(y_uniq, x_img.dtype, y_img.dtype))
        # print("X and Y shapes are {}, {}.".format(x_img.shape, y_img.shape))
        return x_img, y_img

    def _rot_mirror_py_func(self, x_img, y_img):
        x_img_shape, y_img_shape = x_img.shape, y_img.shape
        # https://www.tensorflow.org/guide/function
        [x_img,y_img] = tf.py_function(func=self._rot_mirror_scipy, inp=[x_img,y_img], Tout=[DTYPE_FLOAT,DTYPE_INT])
        x_img.set_shape(x_img_shape)
        y_img.set_shape(y_img_shape)
        # tf.print(x_img_shape, y_img_shape)
        return x_img, y_img

    def _trfm(self, x_img, y_img):
        """
            x_imag: [batch, h, w, channel]
            y_imag: [batch, h, w, channel] 
        """
        # x_img, y_img = self._rot_mirror(x_img, y_img)
        # x_img, y_img = self._rot_90(x_img, y_img) # Try only integer times 90 degree of rotation.
        x_img, y_img = self._rot_mirror_py_func(x_img, y_img)
        
        # In eager mode, the predicate can be int32, but here it has to be strictly boolean.
        x_img, y_img = tf.cond(tf.random.uniform(shape=[], minval=0, maxval=2, dtype=DTYPE_INT)>0, 
                               lambda: (tf.image.flip_left_right(x_img), tf.image.flip_left_right(y_img)),
                               lambda: (x_img, y_img))
        return x_img, y_img
    

    def _parse_helper(self, x_path, y_path):
        """ Input parser for samples of the training set.
        
            The image image of Kylberg dataset has 576*576 in size. We get a quadra-image of it.

            x_imag: [batch, h, w, channel]
            y_imag: [batch, h, w, channel]
        """
        # When we use SparseCategoricalCrossentropy, we don't need one-hot coding.
        # # convert label number into one-hot-encoding
        # one_hot = tf.one_hot(label, self.num_classes)

        # tf.print("!!!!!!!!!!!")

        # load and preprocess the image
        x_img_string = tf.io.read_file(x_path)
        y_img_string = tf.io.read_file(y_path)

        x_img_decoded = tf.image.decode_png(x_img_string, channels=3)
        y_img_decoded = tf.image.decode_png(y_img_string, channels=1)

        # It is important to let the data as tensor.
        x_img = tf.subtract(tf.cast(x_img_decoded, DTYPE_FLOAT), IMAGENET_MEAN)
        y_img = tf.cast(y_img_decoded, DTYPE_INT)

        if self.img_norm_flag:
            x_img = tf.image.per_image_standardization(x_img)

        # Set the shape of image and label so that the Graph knows their shape
        x_img.set_shape((self.img_size, self.img_size, 3))
        y_img.set_shape((self.img_size, self.img_size, 1))

        # if self.trfm_flag:
        #     # # Rotate
        #     # num_rot90 = np.random.randint(low=0, high=4)
        #     # x_img = tf.image.rot90(x_img, k=num_rot90)
        #     # y_img = tf.image.rot90(y_img, k=num_rot90) # Image must be 3-dimensional.
        #     # # Flip
        #     # flip_flag = np.random.randint(low=0, high=2)
        #     # if flip_flag:
        #     #     x_img = tf.image.flip_left_right(x_img)
        #     #     y_img = tf.image.flip_left_right(y_img)
        #     # print("!!!!!!!!!!")
        #     # print(num_rot90, flip_flag)

        #     # rand_idx = np.random.randint(low=0, high=100000)
        #     # tf.io.write_file(os.path.join('/projects/p30309/test_dir/', '{}_{}_{}_x.png'.format(rand_idx, num_rot90, flip_flag)), tf.image.encode_png(tf.cast((x_img-tf.reduce_min(x_img))/(tf.reduce_max(x_img)-tf.reduce_min(x_img))*255, dtype=tf.uint8)))

        #     # tf.io.write_file(os.path.join('/projects/p30309/test_dir/', '{}_{}_{}_y.png'.format(rand_idx, num_rot90, flip_flag)), tf.image.encode_png(tf.cast((y_img-tf.reduce_min(y_img))/(tf.reduce_max(y_img)-tf.reduce_min(y_img))*255, dtype=tf.uint8)))

        #     # Correct way to do data augmentation.
        #     # https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/

        #     # xy_img = tf.concat([x_img, tf.cast(y_img, dtype=DTYPE_FLOAT)], axis=-1)
        #     # xy_img = tf.image.rot90(xy_img, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=DTYPE_INT))
        #     # xy_img = tf.image.random_flip_left_right(xy_img)

        #     x_img, y_img = self._rot_mirror(x_img, y_img)
        #     if tf.random.uniform(shape=[], minval=0, maxval=2, dtype=DTYPE_INT):
        #         x_img = tf.image.flip_left_right(x_img)
        #         y_img = tf.image.flip_left_right(y_img)
            
        return x_img, y_img
        # return xy_img[...,:3], tf.squeeze(tf.cast(xy_img[...,3:], dtype=DTYPE_INT), axis=-1)

    def _cla_wei(self, lab_tensor, filters):
        """ 
            Use tf function instead of node would parallelize the computation.

            The lab_tensor has shape [height, width] 
        """
        one_hot_coding = tf.expand_dims(tf.one_hot(tf.squeeze(lab_tensor, axis=-1), depth=self.num_classes), axis=0) # Make the tensor 4-dimension
        # tf.nn.conv2d doesn't have activation and batch normalization
        sample_wei = tf.squeeze(tf.nn.conv2d(one_hot_coding, filters, strides=1, padding="SAME"), axis=(0, -1))
        return sample_wei

    def _parse_inference(self, x_path, y_path):
        x_img, y_img = self._parse_helper(x_path, y_path)
        y = tf.squeeze(y_img, axis=-1) # y_img is 3-dimensional.
        return x_img, y 

    def _parse_training(self, x_path, y_path):
        x_img, y_img = self._parse_helper(x_path, y_path)

        if self.trfm_flag:
            x_img, y_img = self._trfm(x_img, y_img)

        if self.sample_wei_flag:
            sample_wei = self._cla_wei(y_img, self.lab_wei_tensor)
            # sample_wei = tf.squeeze(self.wei_map(tf.expand_dims(y_img, axis=0)), axis=(0, -1))
            # sample_wei = self.wei_map(y_img)
        else:
            sample_wei = 1 # Cannot use None

        y = tf.squeeze(y_img, axis=-1) # y_img is 3-dimensional.

        return x_img, y, sample_wei

    # # Python function used in Dataset.map() using tf.py_function
    # def _py_parse_helper(self, x_path, y_path):
    #     x_img = Image.open(x_path).convert(mode='RGB') # (h, w, channel)
    #     y_img = Image.open(y_path).convert(mode='L') # (h, w)
    #     x_img

