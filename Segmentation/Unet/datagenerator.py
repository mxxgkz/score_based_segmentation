# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import os
from generate_collages import *

#mean of imagenet dataset in RGB (notice it is RGB, not BGR)
IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

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
#         self.n_points = tf.convert_to_tensor(self.n_points, dtype=tf.int32)

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
#         img = tf.convert_to_tensor(img, dtype=tf.float32)
#         label = tf.convert_to_tensor(label, dtype=tf.int32)

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

class UnetDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline for Kylberg dataset.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes):
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
            data = data.map(self._parse_function, num_parallel_calls=8)
            # For small memory, the buffer_size needs to be small.
            self.data = data.repeat().shuffle(500, reshuffle_each_iteration=True).batch(batch_size).prefetch(20)
        elif mode == 'inference':
            data = data.map(self._parse_function, num_parallel_calls=8)
            self.data = data.batch(batch_size).prefetch(2)
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


    def _parse_function(self, x_path, y_path):
        """ Input parser for samples of the training set.
        
            The image image of Kylberg dataset has 576*576 in size. We get a quadra-image of it.
        """
        # When we use SparseCategoricalCrossentropy, we don't need one-hot coding.
        # # convert label number into one-hot-encoding
        # one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        x_img_string = tf.io.read_file(x_path)
        y_img_string = tf.io.read_file(y_path)
        x_img_decoded = tf.image.decode_png(x_img_string, channels=3)
        y_img_decoded = tf.image.decode_png(y_img_string, channels=1)

        x_img_centered = tf.subtract(tf.cast(x_img_decoded, tf.float32), IMAGENET_MEAN)
        y_img = tf.cast(tf.squeeze(y_img_decoded, axis=-1), tf.int32)

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
        x_img_bgr = x_img_centered[:, :, ::-1]

        return x_img_bgr, y_img