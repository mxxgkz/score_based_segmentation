"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np

#mean of imagenet dataset in RGB (notice it is RGB, not BGR)
IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000):
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
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.txt_file = txt_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_threads=8,
                      output_buffer_size=100*batch_size)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_threads=8,
                      output_buffer_size=100*batch_size)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(',')
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.io.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize(img_decoded, [227, 227])
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

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
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.io.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot



class KylbergDataGenerator(object):
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
        self.data_size = len(self.labels)

        # convert lists to TF tensor
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
        self.subimg_idx = tf.convert_to_tensor(self.subimg_idx, dtype=tf.int32)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels, self.subimg_idx))

        # print(list(data.as_numpy_iterator())[:10])

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            # Without parallel calls, the training would be very slow on colab.
            data = data.map(self._parse_function, num_parallel_calls=20)
            # For small memory, the buffer_size needs to be small.
            self.data = data.repeat().shuffle(2000, reshuffle_each_iteration=True).batch(batch_size).prefetch(buffer_size=30)
        elif mode == 'inference':
            data = data.map(self._parse_function, num_parallel_calls=20)
            self.data = data.batch(batch_size).prefetch(30)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))


    def _read_txt_file(self):
        """ Read the content of the text file and store it into lists.

            For Kylberg dataset, each line of txt file is path, label, and the index of subimage.
        """
        self.img_paths = []
        self.labels = []
        self.subimg_idx = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(',')
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))
                self.subimg_idx.append(int(items[2]))


    def _parse_function(self, filename, label, subimg_idx):
        """ Input parser for samples of the training set.
        
            The image image of Kylberg dataset has 576*576 in size. We get a quadra-image of it.
        """
        # When we use SparseCategoricalCrossentropy, we don't need one-hot coding.
        # # convert label number into one-hot-encoding
        # one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.io.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        # Looks like the following line cannot run in eager mode.
        # I cannot directly get the shape of img_decoded.
        # print(filename, img_decoded.shape, label, subimg_idx, one_hot)
        block_size = 576//2
        r, c = subimg_idx//2, subimg_idx%2
        margin = (block_size-227)//2
        img_resized = tf.image.crop_to_bounding_box(
            img_decoded, 
            offset_height=r*block_size+margin,
            offset_width=c*block_size+margin,
            target_height=227,
            target_width=227)
        """
        Data augmentation comes here.
        """
        img_centered = tf.subtract(tf.cast(img_resized, tf.float32), IMAGENET_MEAN)

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
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, label

    # def _parse_function_inference(self, filename, label, subimg_idx):
    #     """Input parser for samples of the validation/test set."""
    #     # # convert label number into one-hot-encoding
    #     # one_hot = tf.one_hot(label, self.num_classes)

    #     # load and preprocess the image
    #     img_string = tf.io.read_file(filename)
    #     img_decoded = tf.image.decode_png(img_string, channels=3)
    #     block_size = 576//2
    #     r, c = subimg_idx//2, subimg_idx%2
    #     margin = (block_size-227)//2
    #     img_resized = tf.image.crop_to_bounding_box(
    #         img_decoded, 
    #         offset_height=r*block_size+margin,
    #         offset_width=c*block_size+margin,
    #         target_height=227,
    #         target_width=227)

    #     img_centered = tf.subtract(tf.cast(img_resized, tf.float32), IMAGENET_MEAN)

    #     # RGB -> BGR
    #     img_bgr = img_centered[:, :, ::-1]

    #     return img_bgr, label
