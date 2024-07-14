"""This is an implementation of DeepLab and MobileNet in TensorFlow 2.0

The model comes from https://github.com/Golbstein/Keras-segmentation-deeplab-v3.1/blob/master/deeplabv3p.py

The file above is modified from https://github.com/bonlime/keras-deeplab-v3-plus.git
This is the original DeepLab

Another reference: https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/DeepLabV3_plus.py

Need to cross check with the original implementation https://github.com/tensorflow/models/blob/41185bc7c523706f378e1c7964883f1baa21c078/research/deeplab/model.py#L544
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import re
import copy
import os
import dill
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Layer, InputSpec, Conv2D, DepthwiseConv2D, UpSampling2D, ZeroPadding2D, Lambda, AveragePooling2D, Input, Activation, Concatenate, Add, Reshape, BatchNormalization, Dropout
from tensorflow.keras import backend as K
from constants import *

import imp

name = os.path.join(CODE_ROOT_DIR, 'Segmentation/DeepLab3/keras-deeplab-v3-plus-1.1/model')
pathname = name+'.py'
with open(pathname, 'rb') as fp:
    orig_deeplab3 = imp.load_module(name, fp, pathname, ('.py', 'rb', imp.PY_SOURCE))

name_multi_backbone = os.path.join(CODE_ROOT_DIR, 'Segmentation/DeepLab3/keras-deeplab-v3-plus/deeplab_origin') 
pathname_multi_backbone = name_multi_backbone+'.py'
with open(pathname_multi_backbone, 'rb') as fp:
    orig_multi_backbone_deeplab3 = imp.load_module(name_multi_backbone, fp, pathname_multi_backbone, ('.py', 'rb', imp.PY_SOURCE))

name_multi_backbone_modified = os.path.join(CODE_ROOT_DIR, 'Segmentation/DeepLab3/keras-deeplab-v3-plus/deeplab_modified') 
pathname_multi_backbone_modified = name_multi_backbone_modified+'.py'
with open(pathname_multi_backbone_modified, 'rb') as fp:
    modified_multi_backbone_deeplab3 = imp.load_module(name_multi_backbone_modified, fp, pathname_multi_backbone_modified, ('.py', 'rb', imp.PY_SOURCE))

class SepConvBNBlock(Layer):
    def __init__(self, filters, kernel_size=3, strides=1, rate=1, depth_acti=False, epsilon=1e-3, name='', plot_model=False):
        """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
            Implements right "same" padding for even kernel sizes
            Args:
                x: input tensor
                filters: num of filters in pointwise convolution
                prefix: prefix before name
                stride: stride at depthwise conv
                kernel_size: kernel size for depthwise convolution
                rate: atrous rate for depthwise convolution
                depth_activation: flag to use activation between depthwise & pointwise convs
                epsilon: epsilon to use in BN layer

            Reference: https://github.com/tensorflow/models/blob/9737810ff14dfe91e824931c72e167c6f2e5d327/research/deeplab/core/xception.py#L104
        """
        super(SepConvBNBlock, self).__init__(name=name)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.rate = rate
        self.depth_acti = depth_acti
        self.epsilon = epsilon

        if self.strides == 1:
            depth_padding = 'SAME'
        else:
            kernel_size_eff = kernel_size + (kernel_size-1)*(rate-1) # Refer to the paper [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf)
            pad_tot = kernel_size_eff - 1 # When we choose odd-number kernel_size, this will be even-number
            pad_beg = pad_tot // 2
            pad_end = pad_tot - pad_beg
            depth_padding = 'VALID' # No padding
            self.zeropad = ZeroPadding2D(padding=((pad_beg, pad_end), (pad_beg, pad_end)), name='zero_padding')

        if not self.depth_acti:
            self.acti = Activation(activation='relu', name='acti')

        self.dpconv = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, 
                                      dilation_rate=rate, padding=depth_padding, 
                                      use_bias=False, name='depth_conv')
        self.dpbn = BatchNormalization(epsilon=epsilon, name='depthwise_bn')
        
        if self.depth_acti:
            self.dpacti = Activation(activation='relu', name='depthwise_acti')
        
        self.ptconv = Conv2D(filters=filters, kernel_size=1, padding='SAME',
                             use_bias=False, name='pointwise_conv')
        self.ptbn = BatchNormalization(epsilon=epsilon, name='pointwise_bn')

        if self.depth_acti:
            self.ptacti = Activation(activation='relu', name='pointwise_acti')

        # Functional API for plotting the architecture
        if plot_model:
            inputs = tf.keras.Input(shape=(IMG_SIZE//2, IMG_SIZE//2, 48)) # 128
            outputs = self.call(inputs)
            self.graph = tf.keras.Model(inputs=inputs, outputs=outputs)
            

    def _sequence(self, inputs, training=False):
        x = inputs
        if self.strides != 1:
            x = self.zeropad(x)
        
        if not self.depth_acti:
            x = self.acti(x)

        x = self.dpconv(x)
        x = self.dpbn(x, training=training)

        if self.depth_acti:
            x = self.dpacti(x)
        
        x = self.ptconv(x)
        x = self.ptbn(x, training=training)

        if self.depth_acti:
            x = self.ptacti(x)
        return x


    def call(self, inputs, training=False):
        # dum_input_shape = [-1]
        # dum_input_shape.extend([ele for ele in self.input_shape])
        # inputs = tf.reshape(input_tensor, dum_input_shape)
        outputs = self._sequence(inputs, training=training)
        # self.input_shape, self.output_shape = inputs.shape, outputs.shape
        return outputs


    def get_config(self):
        config = super(SepConvBNBlock, self).get_config()
        config.update({'filters': self.filters,
                       'kernel_size': self.kernel_size,
                       'strides': self.strides,
                       'rate': self.rate,
                       'depth_acti': self.depth_acti,
                       'epsilon': self.epsilon})
        return config
        

class Conv2DSame(Layer):
    """ Implements right 'SAME' padding for even kernel sizes.
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution

        Reference: https://github.com/tensorflow/models/blob/9737810ff14dfe91e824931c72e167c6f2e5d327/research/slim/nets/resnet_utils.py#L78
    """
    def __init__(self, filters, kernel_size=3, strides=1, rate=1, name='', plot_model=False):
        super(Conv2DSame, self).__init__(name=name)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.rate = rate
        
        if strides == 1:
            self.conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding='SAME', use_bias=False, dilation_rate=rate)
        else:
            kernel_size_eff = kernel_size + (kernel_size-1)*(rate-1) # Refer to the paper [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf)
            pad_tot = kernel_size_eff - 1 # When we choose odd-number kernel_size, this will be even-number
            pad_beg = pad_tot // 2
            pad_end = pad_tot - pad_beg
            self.zeropad = ZeroPadding2D(padding=((pad_beg, pad_end), (pad_beg, pad_end)))
            self.conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding='VALID', use_bias=False, dilation_rate=rate)

        # Functional API for plotting the architecture
        if plot_model:
            inputs = tf.keras.Input(shape=(IMG_SIZE//2, IMG_SIZE//2, 48)) # 64
            outputs = self.call(inputs)
            self.graph = tf.keras.Model(inputs=inputs, outputs=outputs)

    def call(self, inputs):
        # print(inputs.shape)
        if self.strides == 1:
            outputs = self.conv(inputs)
        else:
            x = self.zeropad(inputs)
            outputs = self.conv(x)
        # self.input_shape, self.output_shape = inputs.shape, outputs.shape
        return outputs

    def get_config(self):
        config = super(Conv2DSame, self).get_config()
        config.update({'filters': self.filters,
                       'kernel_size': self.kernel_size,
                       'strides': self.strides,
                       'rate': self.rate})
        return config


class XceptionBlock(Layer):
    def __init__(self, ls_filters, skip_con_type, strides, rate=1, depth_acti=False, return_skip=False, name='', plot_model=False):
        """ Basic building block of modified Xception network
            Args:
                inputs: input tensor
                depth_list: number of filters in each SepConv layer. len(depth_list) == 3
                prefix: prefix before name
                skip_connection_type: one of {'conv','sum','none'}
                stride: stride at last depthwise conv
                rate: atrous rate for depthwise convolution
                depth_activation: flag to use activation between depthwise & pointwise convs
                return_skip: flag to return additional tensor after 2 SepConvs for decoder

            Reference: https://github.com/tensorflow/models/blob/9737810ff14dfe91e824931c72e167c6f2e5d327/research/deeplab/core/xception.py#L205
        """
        super(XceptionBlock, self).__init__(name=name)

        self.ls_filters=ls_filters
        self.skip_con_type=skip_con_type
        self.strides=strides
        self.rate=rate
        self.depth_acti=depth_acti
        self.return_skip=return_skip

        self.ls_sepconvbn = []
        for i in range(3):
            self.ls_sepconvbn.append(
                SepConvBNBlock(filters=ls_filters[i], 
                               strides=strides if i == 2 else 1, 
                               rate=rate, 
                               depth_acti=depth_acti, 
                               name='sepconvbn_{}'.format(i+1),
                               plot_model=False)) # plot_model=plot_model if i==1 or i==2 else False

        if self.skip_con_type == 'conv':
            self.conv_same = Conv2DSame(filters=ls_filters[-1], kernel_size=1, strides=strides, name='shortcut_conv', plot_model=plot_model)
            self.bn = BatchNormalization(name='shortcut_bn')
            self.add = Add(name='shortcut_conv_add')

        elif skip_con_type == 'sum':
            self.add = Add(name='shortcut_add')

        # Functional API for plotting the architecture
        if plot_model:
            inputs = Input(shape=(IMG_SIZE//2, IMG_SIZE//2, 32)) # 128
            outputs = self.call(inputs)
            self.graph = Model(inputs=inputs, outputs=outputs)

        
    def call(self, inputs, training=False):
        resi = inputs
        # When strides=2, the following 4 lines would shrink the input by 2.
        # However, would be of same size, when strides=1.
        for i in range(3):
            # print(resi.shape)
            resi = self.ls_sepconvbn[i](resi, training=training)
            if i ==1:
                skip = resi

        if self.skip_con_type == 'conv':
            shortcut = self.conv_same(inputs)
            shortcut = self.bn(shortcut, training=training)
            outputs = self.add([resi, shortcut])
        elif self.skip_con_type == 'sum':
            outputs = self.add([resi, inputs])
        elif self.skip_con_type == 'none':
            outputs = resi
        
        if self.return_skip:
            res = outputs, skip
        else:
            res = outputs
        self.input_shapes, self.output_shapes = inputs.shape, outputs.shape
        return res

    def get_config(self):
        config = super(XceptionBlock, self).get_config()
        config.update({'ls_filters': self.ls_filters,
                       'skip_con_type': self.skip_con_type,
                       'strides': self.strides,
                       'rate': self.rate,
                       'depth_acti': self.depth_acti})
        return config


class DeepLab3PlusModel(Model):
    """ Instantiates the DeepLab3Plus+ architecture
        Optionally loads weights pre-trained
        on PASCAL VOC. This model is available for TensorFlow only,
        and can only be used with inputs following the TensorFlow
        data format `(width, height, channels)`.
        # Arguments
            weights: one of 'pascal_voc' (pre-trained on pascal voc)
                or None (random initialization)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: shape of input image. format HxWxC
                PASCAL VOC model was trained on (512,512,3) images
            classes: number of desired classes. If classes != 21,
                last layer is initialized randomly
            backbone: backbone to use. one of {'xception','mobilenetv2'}
            OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
                Used only for xception backbone.
            alpha: controls the width of the MobileNetV2 network. This is known as the
                width multiplier in the MobileNetV2 paper.
                    - If `alpha` < 1.0, proportionally decreases the number
                        of filters in each layer.
                    - If `alpha` > 1.0, proportionally increases the number
                        of filters in each layer.
                    - If `alpha` = 1, default number of filters from the paper
                        are used at each layer.
                Used only for mobilenetv2 backbone
        # Returns
            A Keras model instance.
        # Raises
            RuntimeError: If attempting to run this model with a
                backend that does not support separable convolutions.
            ValueError: in case of invalid argument for `weights` or `backbone`

        Reference: 
            Multiscale logits: https://github.com/tensorflow/models/blob/41185bc7c523706f378e1c7964883f1baa21c078/research/deeplab/model.py#L220
            Heavylifting function for network: https://github.com/tensorflow/models/blob/9737810ff14dfe91e824931c72e167c6f2e5d327/research/deeplab/model.py#L544
    """
    def __init__(self, keep_prob=0.9, num_classes=21, output_stride=16, input_shapes=(256, 256, 3), plot_model=False):
        super(DeepLab3PlusModel, self).__init__(name='DeepLab3PlusModel')

        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.drop_prob = 1 - keep_prob
        self.plot_model = plot_model
        self.input_shapes = input_shapes

        if self.output_stride == 8:
            self.entry_block3_stride = 1
            self.middle_block_rate = 2 # ! Not mentioned in paper, but required
            self.exit_block_rates = (2, 4)
            self.atrous_rates = (12, 24, 36)
        else:
            self.entry_block3_stride = 2
            self.middle_block_rate = 1 # ! Not mentioned in paper, but required
            self.exit_block_rates = (1, 2)
            self.atrous_rates = (6, 12, 18)


    def build(self, input_shape):
        """
            Reference: https://towardsdatascience.com/the-evolution-of-deeplab-for-semantic-segmentation-95082b025571

            Useful description to understand the architecture of the DeepLab.
            "Decoder: The encoder is based on an output stride of 16, i.e. the input image is down-sampled by a factor 
             of 16. So, instead of using bilinear up-sampling with a factor of 16, the encoded features are first up-sampled 
             by a factor of 4 and concatenated with corresponding low level features from the encoder module having the same 
             spatial dimensions. Before concatenating, 1 x 1 convolutions are applied on the low level features to reduce the 
             number of channels. After concatenation, a few 3 x 3 convolutions are applied and the features are up-sampled by 
             a factor of 4. This gives the output of the same size as that of the input image."
        
        """
        # self.scale_center = Lambda(lambda x: x/127.5-1, name='scale_center')

        # =======Feature extractor=======
        # Shrink 16 times
        # This feature extractor comes from xception: https://github.com/tensorflow/models/blob/9737810ff14dfe91e824931c72e167c6f2e5d327/research/deeplab/core/xception.py#L417
        # The following is xception71: https://github.com/tensorflow/models/blob/9737810ff14dfe91e824931c72e167c6f2e5d327/research/deeplab/core/xception.py#L747

        # Entry block
        self.entry_conv = Conv2D(filters=32, kernel_size=3, strides=2, padding='SAME',
                                 use_bias=False, name='entry_conv')
        self.entry_bn_1 = BatchNormalization(name='entry_bn_1')
        self.entry_acti_1 = Activation(activation='relu', name='entry_acti_1')
        self.entry_conv_same = Conv2DSame(filters=64, name='entry_conv_same')
        self.entry_bn_2 = BatchNormalization(name='entry_bn_2')
        self.entry_acti_2 = Activation(activation='relu', name='entry_acti_2')
        
        self.entry_xception_1 = XceptionBlock(ls_filters=[128]*3, skip_con_type='conv', strides=2, depth_acti=False, name='entry_xception_1')
        self.entry_xception_2 = XceptionBlock(ls_filters=[256]*3, skip_con_type='conv', strides=2, depth_acti=False, return_skip=True, name='entry_xception_2')
        self.entry_xception_3 = XceptionBlock(ls_filters=[728]*3, skip_con_type='conv', strides=self.entry_block3_stride, depth_acti=False, name='entry_xception_3')

        # Middle block
        self.middle_ls_xception = []
        for i in range(16):
            self.middle_ls_xception.append(
                XceptionBlock(ls_filters=[728]*3, skip_con_type='sum', strides=1, 
                              rate=self.middle_block_rate, depth_acti=False, name='middle_xception_{}'.format(i+1)))
        # Exit block
        self.exit_xception_1 = XceptionBlock(ls_filters=[728, 1024, 1024], skip_con_type='conv', strides=1, 
                                             rate=self.exit_block_rates[0], depth_acti=False, name='exit_xception_1')
        self.exit_xception_2 = XceptionBlock(ls_filters=[1536, 1536, 2048], skip_con_type='none', strides=1, 
                                             rate=self.exit_block_rates[1], depth_acti=True, name='exit_xception_2')

        # =======Branching for Atrous Spatial Pyramid Pooling=======

        # ASPP layers
        self.aspp0_conv = Conv2D(filters=256, kernel_size=1, padding='SAME', use_bias=False, name='aspp0_conv')
        self.aspp0_bn = BatchNormalization(epsilon=1e-5, name='aspp0_bn')
        self.aspp0_acti = Activation(activation='relu', name='aspp0_acti')

        self.aspp1 = SepConvBNBlock(filters=256, rate=self.atrous_rates[0], depth_acti=True, epsilon=1e-5, name='aspp1')
        self.aspp2 = SepConvBNBlock(filters=256, rate=self.atrous_rates[1], depth_acti=True, epsilon=1e-5, name='aspp2')
        self.aspp3 = SepConvBNBlock(filters=256, rate=self.atrous_rates[2], depth_acti=True, epsilon=1e-5, name='aspp3')

        # Image feature branch
        out_size = int(np.ceil(input_shape[1] / self.output_stride))
        # print("This is an input shape {} and out_size {}.".format(input_shape, out_size))
        self.bran4_ave_pool = AveragePooling2D(pool_size=out_size, name='bran4_ave_pool')
        self.bran4_conv = Conv2D(filters=256, kernel_size=1, padding='SAME', use_bias=False, name='bran4_conv')
        self.bran4_bn = BatchNormalization(name='bran4_bn')
        self.bran4_acti = Activation(activation='relu', name='bran4_acti')
        # See the parameter as an example: https://github.com/tensorflow/models/blob/41185bc7c523706f378e1c7964883f1baa21c078/research/deeplab/core/utils.py#L29
        # Also see a blog related to bugs: https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
        # A good reference: https://stackoverflow.com/questions/44186042/keras-methods-to-enlarge-spartial-dimension-of-the-layer-output-blob
        self.bran4_up = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(
                               x, size=(out_size, out_size), align_corners=True), name='bran4_up')

        # Concatenate ASPP branches & project
        self.aspp_concat = Concatenate(name='aspp_concat')
        
        self.proj_conv = Conv2D(filters=256, kernel_size=1, padding='SAME', use_bias=False, name='proj_conv')
        self.proj_bn = BatchNormalization(epsilon=1e-5, name='proj_bn')
        self.proj_acti = Activation(activation='relu', name='proj_acti')
        self.proj_dropout = Dropout(rate=self.drop_prob, name='proj_dropout')

        # =======Decoder=======
        # See decoder reference: https://github.com/tensorflow/models/blob/41185bc7c523706f378e1c7964883f1baa21c078/research/deeplab/model.py#L618

        deco_out_size = int(np.ceil(input_shape[1] / 4))
        self.deco_up = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(
                              x, size=(deco_out_size, deco_out_size), align_corners=True), name='deco_up')
        self.deco_skip_conv = Conv2D(filters=48, kernel_size=1, padding='SAME', use_bias=False,
                                     name='feature_proj_conv')
        self.deco_skip_bn = BatchNormalization(epsilon=1e-5, name='feature_proj_bn')
        self.deco_skip_acti = Activation(activation='relu', name='feature_proj_acti')
        # https://github.com/tensorflow/models/blob/9737810ff14dfe91e824931c72e167c6f2e5d327/research/deeplab/model.py#L804
        self.deco_concat = Concatenate(name='feature_proj_concat')
        self.deco_sepconv_1 = SepConvBNBlock(filters=256, depth_acti=True, epsilon=1e-5, name='deco_sepconv_0')
        self.deco_sepconv_2 = SepConvBNBlock(filters=256, depth_acti=True, epsilon=1e-5, name='deco_sepconv_1')

        # =======Get logit=======
        self.last_conv = Conv2D(filters=self.num_classes, kernel_size=1, padding='SAME', name='last_conv')
        self.last_up = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(
                              x, size=(input_shape[1], input_shape[2]), align_corners=True), name='last_up')
        self.last_acti = Activation(activation='softmax', name='last_acti')

        # Functional API for plotting network
        if self.plot_model:
            inputs = Input(shape=self.input_shapes)
            outputs = self.call(inputs)
            self.graph = Model(inputs=inputs, outputs=outputs)

    # @tf.function
    def call(self, inputs, training=False):
        # scale_center = self.scale_center(inputs)

        # =======Feature extractor=======
        # Shrink 16 times

        # Entry block
        entry_conv = self.entry_conv(inputs)
        entry_bn_1 = self.entry_bn_1(entry_conv, training=training)
        entry_acti_1 = self.entry_acti_1(entry_bn_1)
        entry_conv_same = self.entry_conv_same(entry_acti_1)
        entry_bn_2 = self.entry_bn_2(entry_conv_same, training=training)
        entry_acti_2 = self.entry_acti_2(entry_bn_2)
        entry_xception_1 = self.entry_xception_1(entry_acti_2, training=training)
        entry_xception_2, entry_xception_skip_2 = self.entry_xception_2(entry_xception_1, training=training)
        entry_xception_3 = self.entry_xception_3(entry_xception_2, training=training)

        # Middle block
        middle_ls_xception = [] 
        tmp_res = entry_xception_3
        for i in range(16):
            tmp_res = self.middle_ls_xception[i](tmp_res, training=training)
            middle_ls_xception.append(tmp_res)

        # Exit block
        exit_xception_1 = self.exit_xception_1(middle_ls_xception[-1], training=training)
        exit_xception_2 = self.exit_xception_2(exit_xception_1, training=training)

        # =======Branching for Atrous Spatial Pyramid Pooling=======

        # Image feature branch
        bran4_ave_pool = self.bran4_ave_pool(exit_xception_2)
        bran4_conv = self.bran4_conv(bran4_ave_pool)
        bran4_bn = self.bran4_bn(bran4_conv, training=training)
        bran4_acti = self.bran4_acti(bran4_bn)
        # print(exit_xception_2.shape)
        # print(bran4_ave_pool.shape)
        # print(bran4_conv.shape)
        # print(bran4_bn.shape)
        # print("The bran4_acti shape is {}.".format(bran4_acti.shape))
        bran4_up = self.bran4_up(bran4_acti)

        # ASPP layers
        aspp0_conv = self.aspp0_conv(exit_xception_2)
        aspp0_bn = self.aspp0_bn(aspp0_conv, training=training)
        aspp0_acti = self.aspp0_acti(aspp0_bn)

        aspp1 = self.aspp1(exit_xception_2, training=training)
        aspp2 = self.aspp2(exit_xception_2, training=training)
        aspp3 = self.aspp3(exit_xception_2, training=training)

        # Concatenate ASPP branches & project
        aspp_concat = self.aspp_concat([bran4_up, aspp0_acti, aspp1, aspp2, aspp3])

        proj_conv = self.proj_conv(aspp_concat)
        proj_bn = self.proj_bn(proj_conv, training=training)
        proj_acti = self.proj_acti(proj_bn)
        proj_dropout = self.proj_dropout(proj_acti, training=training)

        # =======Decoder=======
        deco_up = self.deco_up(proj_dropout)
        deco_skip_conv = self.deco_skip_conv(entry_xception_skip_2)
        deco_skip_bn = self.deco_skip_bn(deco_skip_conv, training=training)
        deco_skip_acti = self.deco_skip_acti(deco_skip_bn)
        deco_concat = self.deco_concat([deco_up, deco_skip_acti])
        deco_sepconv_1 = self.deco_sepconv_1(deco_concat, training=training)
        deco_sepconv_2 = self.deco_sepconv_2(deco_sepconv_1, training=training)

        # =======Get logit=======
        last_conv = self.last_conv(deco_sepconv_2)
        last_up = self.last_up(last_conv)
        if not training:
            return self.last_acti(last_up)
        else:
            return last_up
    
    def get_config(self):
        config = super(DeepLab3PlusModel, self).get_config()
        config.update({'keep_prob': self.keep_prob,
                       'num_classes': self.num_classes,
                       'output_stride': self.output_stride})
        return config


class DeepLab3PlusModifiedModel(Model):
    """ Instantiates the DeepLab3Plus+ architecture
        Smaller network with only 1.5 million parameters (instead of 41 millions original model).

        Optionally loads weights pre-trained
        on PASCAL VOC. This model is available for TensorFlow only,
        and can only be used with inputs following the TensorFlow
        data format `(width, height, channels)`.
        # Arguments
            weights: one of 'pascal_voc' (pre-trained on pascal voc)
                or None (random initialization)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: shape of input image. format HxWxC
                PASCAL VOC model was trained on (512,512,3) images
            classes: number of desired classes. If classes != 21,
                last layer is initialized randomly
            backbone: backbone to use. one of {'xception','mobilenetv2'}
            OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
                Used only for xception backbone.
            alpha: controls the width of the MobileNetV2 network. This is known as the
                width multiplier in the MobileNetV2 paper.
                    - If `alpha` < 1.0, proportionally decreases the number
                        of filters in each layer.
                    - If `alpha` > 1.0, proportionally increases the number
                        of filters in each layer.
                    - If `alpha` = 1, default number of filters from the paper
                        are used at each layer.
                Used only for mobilenetv2 backbone
        # Returns
            A Keras model instance.
        # Raises
            RuntimeError: If attempting to run this model with a
                backend that does not support separable convolutions.
            ValueError: in case of invalid argument for `weights` or `backbone`

        Reference: 
            Multiscale logits: https://github.com/tensorflow/models/blob/41185bc7c523706f378e1c7964883f1baa21c078/research/deeplab/model.py#L220
            Heavylifting function for network: https://github.com/tensorflow/models/blob/9737810ff14dfe91e824931c72e167c6f2e5d327/research/deeplab/model.py#L544
    """
    def __init__(self, keep_prob=0.9, num_classes=21, output_stride=16, input_shapes=(256, 256, 3), plot_model=False):
        super(DeepLab3PlusModifiedModel, self).__init__(name='DeepLab3PlusModifiedModel')

        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.drop_prob = 1 - keep_prob
        self.plot_model = plot_model
        self.input_shapes = input_shapes # self.input_shape is reserved as a built-in member variable.

        if self.output_stride == 8:
            self.entry_block3_stride = 1
            self.middle_block_rate = 2 # ! Not mentioned in paper, but required
            self.exit_block_rates = (2, 4)
            self.atrous_rates = (12, 24, 36)
        else:
            self.entry_block3_stride = 2
            self.middle_block_rate = 1 # ! Not mentioned in paper, but required
            self.exit_block_rates = (1, 2)
            self.atrous_rates = (6, 12, 18)


    def build(self, input_shape):
        """
            Reference: https://towardsdatascience.com/the-evolution-of-deeplab-for-semantic-segmentation-95082b025571

            Useful description to understand the architecture of the DeepLab.
            "Decoder: The encoder is based on an output stride of 16, i.e. the input image is down-sampled by a factor 
             of 16. So, instead of using bilinear up-sampling with a factor of 16, the encoded features are first up-sampled 
             by a factor of 4 and concatenated with corresponding low level features from the encoder module having the same 
             spatial dimensions. Before concatenating, 1 x 1 convolutions are applied on the low level features to reduce the 
             number of channels. After concatenation, a few 3 x 3 convolutions are applied and the features are up-sampled by 
             a factor of 4. This gives the output of the same size as that of the input image."
        
        """
        # self.scale_center = Lambda(lambda xx: xx/127.5-1, name='scale_center')

        # =======Feature extractor=======
        # Shrink 16 times
        # This feature extractor comes from xception: https://github.com/tensorflow/models/blob/9737810ff14dfe91e824931c72e167c6f2e5d327/research/deeplab/core/xception.py#L417
        # The following is xception71: https://github.com/tensorflow/models/blob/9737810ff14dfe91e824931c72e167c6f2e5d327/research/deeplab/core/xception.py#L747

        # Entry block
        self.entry_conv = Conv2D(filters=16, kernel_size=3, strides=2, padding='SAME',
                                 use_bias=False, name='entry_conv')
        self.entry_bn_1 = BatchNormalization(name='entry_bn_1')
        self.entry_acti_1 = Activation(activation='relu', name='entry_acti_1')
        self.entry_conv_same = Conv2DSame(filters=32, name='entry_conv_same')
        self.entry_bn_2 = BatchNormalization(name='entry_bn_2')
        self.entry_acti_2 = Activation(activation='relu', name='entry_acti_2')
        
        self.entry_xception_1 = XceptionBlock(ls_filters=[48]*3, skip_con_type='conv', strides=2, depth_acti=False, name='entry_xception_1')
        self.entry_xception_2 = XceptionBlock(ls_filters=[64]*3, skip_con_type='conv', strides=2, depth_acti=False, return_skip=True, name='entry_xception_2')
        self.entry_xception_3 = XceptionBlock(ls_filters=[128]*3, skip_con_type='conv', strides=self.entry_block3_stride, depth_acti=False, name='entry_xception_3')

        # Middle block
        self.middle_ls_xception = []
        for i in range(MID_FLOW_BLOCK_NUM):
            self.middle_ls_xception.append(
                XceptionBlock(ls_filters=[128]*3, skip_con_type='sum', strides=1, 
                              rate=self.middle_block_rate, depth_acti=False, name='middle_xception_{}'.format(i+1)))
        # Exit block
        self.exit_xception_1 = XceptionBlock(ls_filters=[128, 256, 256], skip_con_type='conv', strides=1, 
                                             rate=self.exit_block_rates[0], depth_acti=False, name='exit_xception_1')
        self.exit_xception_2 = XceptionBlock(ls_filters=[384, 384, 512], skip_con_type='none', strides=1, 
                                             rate=self.exit_block_rates[1], depth_acti=True, name='exit_xception_2')

        # =======Branching for Atrous Spatial Pyramid Pooling=======

        # ASPP layers
        self.aspp0_conv = Conv2D(filters=64, kernel_size=1, padding='SAME', use_bias=False, name='aspp0_conv')
        self.aspp0_bn = BatchNormalization(epsilon=1e-5, name='aspp0_bn')
        self.aspp0_acti = Activation(activation='relu', name='aspp0_acti')

        self.aspp1 = SepConvBNBlock(filters=64, rate=self.atrous_rates[0], depth_acti=True, epsilon=1e-5, name='aspp1')
        self.aspp2 = SepConvBNBlock(filters=64, rate=self.atrous_rates[1], depth_acti=True, epsilon=1e-5, name='aspp2')
        self.aspp3 = SepConvBNBlock(filters=64, rate=self.atrous_rates[2], depth_acti=True, epsilon=1e-5, name='aspp3')

        # Image feature branch
        out_size = int(np.ceil(input_shape[1] / self.output_stride))
        # print("This is an input shape {} and out_size {}.".format(input_shape, out_size))
        self.bran4_ave_pool = AveragePooling2D(pool_size=out_size, name='bran4_ave_pool')
        self.bran4_conv = Conv2D(filters=64, kernel_size=1, padding='SAME', use_bias=False, name='bran4_conv')
        self.bran4_bn = BatchNormalization(epsilon=1e-5, name='bran4_bn')
        self.bran4_acti = Activation(activation='relu', name='bran4_acti')
        # See the parameter as an example: https://github.com/tensorflow/models/blob/41185bc7c523706f378e1c7964883f1baa21c078/research/deeplab/core/utils.py#L29
        # Also see a blog related to bugs: https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
        # A good reference: https://stackoverflow.com/questions/44186042/keras-methods-to-enlarge-spartial-dimension-of-the-layer-output-blob
        self.bran4_up = Lambda(lambda xx: tf.compat.v1.image.resize_bilinear(
                               xx, size=(out_size, out_size), align_corners=True), name='bran4_up')

        # Concatenate ASPP branches & project
        self.aspp_concat = Concatenate(name='aspp_concat')
        
        self.proj_conv = Conv2D(filters=64, kernel_size=1, padding='SAME', use_bias=False, name='proj_conv')
        self.proj_bn = BatchNormalization(epsilon=1e-5, name='proj_bn')
        self.proj_acti = Activation(activation='relu', name='proj_acti')
        self.proj_dropout = Dropout(rate=self.drop_prob, name='proj_dropout')

        # =======Decoder=======
        # See decoder reference: https://github.com/tensorflow/models/blob/41185bc7c523706f378e1c7964883f1baa21c078/research/deeplab/model.py#L618

        deco_out_size = int(np.ceil(input_shape[1] / 4))
        self.deco_up = Lambda(lambda xx: tf.compat.v1.image.resize_bilinear(
                              xx, size=(deco_out_size, deco_out_size), align_corners=True), name='deco_up')
        self.deco_skip_conv = Conv2D(filters=24, kernel_size=1, padding='SAME', use_bias=False,
                                     name='feature_proj_conv')
        self.deco_skip_bn = BatchNormalization(epsilon=1e-5, name='feature_proj_bn')
        self.deco_skip_acti = Activation(activation='relu', name='feature_proj_acti')
        # https://github.com/tensorflow/models/blob/9737810ff14dfe91e824931c72e167c6f2e5d327/research/deeplab/model.py#L804
        self.deco_concat = Concatenate(name='feature_proj_concat')
        self.deco_sepconv_1 = SepConvBNBlock(filters=64, depth_acti=True, epsilon=1e-5, name='deco_sepconv_0')
        self.deco_sepconv_2 = SepConvBNBlock(filters=64, depth_acti=True, epsilon=1e-5, name='deco_sepconv_1')

        # =======Get logit=======
        self.last_conv = Conv2D(filters=self.num_classes, kernel_size=1, padding='SAME', name='last_conv')
        self.last_up = Lambda(lambda xx: tf.compat.v1.image.resize_bilinear(
                              xx, size=(input_shape[1], input_shape[2]), align_corners=True), name='last_up')
        self.last_acti = Activation(activation='softmax', name='last_acti')

        # Functional API for plotting network
        if self.plot_model:
            inputs = Input(shape=self.input_shapes)
            outputs = self.call(inputs)
            self.graph = Model(inputs=inputs, outputs=outputs)

    # @tf.function
    def call(self, inputs, training=False):
        # scale_center = self.scale_center(inputs)

        # =======Feature extractor=======
        # Shrink 16 times

        # Entry block
        entry_conv = self.entry_conv(inputs)
        entry_bn_1 = self.entry_bn_1(entry_conv, training=training)
        entry_acti_1 = self.entry_acti_1(entry_bn_1)
        entry_conv_same = self.entry_conv_same(entry_acti_1)
        entry_bn_2 = self.entry_bn_2(entry_conv_same, training=training)
        entry_acti_2 = self.entry_acti_2(entry_bn_2)
        entry_xception_1 = self.entry_xception_1(entry_acti_2, training=training)
        entry_xception_2, entry_xception_skip_2 = self.entry_xception_2(entry_xception_1, training=training)
        entry_xception_3 = self.entry_xception_3(entry_xception_2, training=training)

        # Middle block
        middle_ls_xception = [] 
        tmp_res = entry_xception_3
        for i in range(MID_FLOW_BLOCK_NUM):
            tmp_res = self.middle_ls_xception[i](tmp_res, training=training)
            middle_ls_xception.append(tmp_res)

        # Exit block
        exit_xception_1 = self.exit_xception_1(middle_ls_xception[-1], training=training)
        exit_xception_2 = self.exit_xception_2(exit_xception_1, training=training)

        # =======Branching for Atrous Spatial Pyramid Pooling=======

        # Image feature branch
        bran4_ave_pool = self.bran4_ave_pool(exit_xception_2)
        bran4_conv = self.bran4_conv(bran4_ave_pool)
        bran4_bn = self.bran4_bn(bran4_conv, training=training)
        bran4_acti = self.bran4_acti(bran4_bn)
        # print(exit_xception_2.shape)
        # print(bran4_ave_pool.shape)
        # print(bran4_conv.shape)
        # print(bran4_bn.shape)
        # print("The bran4_acti shape is {}.".format(bran4_acti.shape))
        bran4_up = self.bran4_up(bran4_acti)

        # ASPP layers
        aspp0_conv = self.aspp0_conv(exit_xception_2)
        aspp0_bn = self.aspp0_bn(aspp0_conv, training=training)
        aspp0_acti = self.aspp0_acti(aspp0_bn)

        aspp1 = self.aspp1(exit_xception_2, training=training)
        aspp2 = self.aspp2(exit_xception_2, training=training)
        aspp3 = self.aspp3(exit_xception_2, training=training)

        # Concatenate ASPP branches & project
        aspp_concat = self.aspp_concat([bran4_up, aspp0_acti, aspp1, aspp2, aspp3])

        proj_conv = self.proj_conv(aspp_concat)
        proj_bn = self.proj_bn(proj_conv, training=training)
        proj_acti = self.proj_acti(proj_bn)
        proj_dropout = self.proj_dropout(proj_acti, training=training)

        # =======Decoder=======
        deco_up = self.deco_up(proj_dropout)
        deco_skip_conv = self.deco_skip_conv(entry_xception_skip_2)
        deco_skip_bn = self.deco_skip_bn(deco_skip_conv, training=training)
        deco_skip_acti = self.deco_skip_acti(deco_skip_bn)
        deco_concat = self.deco_concat([deco_up, deco_skip_acti])
        deco_sepconv_1 = self.deco_sepconv_1(deco_concat, training=training)
        deco_sepconv_2 = self.deco_sepconv_2(deco_sepconv_1, training=training)

        # =======Get logit=======
        last_conv = self.last_conv(deco_sepconv_2)
        last_up = self.last_up(last_conv)
        if not training:
            return self.last_acti(last_up)
        else:
            return last_up
    
    def get_config(self):
        config = super(DeepLab3PlusModifiedModel, self).get_config()
        config.update({'keep_prob': self.keep_prob,
                       'num_classes': self.num_classes,
                       'output_stride': self.output_stride})
        return config


class DeepLab3Plus(Model):
    def __init__(self, keep_prob, num_classes, output_stride=16, skip_layer=[], input_shapes=(256,256,3), 
                 pretrained_wei_path='Segmentation/DeepLab3/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5', 
                 deeplab3plusmodel=DeepLab3PlusModel, 
                 deeplab3plusmodel_kwargs={'keep_prob': 0.9,
                                           'num_classes': 21,
                                           'output_stride': 16,
                                           'input_shapes': (256,256,3)}, 
                 plot_model=False):
        super(DeepLab3Plus, self).__init__(name='DeepLab3Plus')

        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.skip_layer = skip_layer
        self.pretrained_wei = pretrained_wei_path
        self.input_shapes = input_shapes
        self.drop_prob = 1 - keep_prob

        # self.model = DeepLab3PlusModel(self.keep_prob, self.num_classes, self.output_stride, plot_model=plot_model)
        self.model = deeplab3plusmodel(plot_model=plot_model, **deeplab3plusmodel_kwargs)
        self.model(tf.expand_dims(tf.zeros(self.input_shapes, dtype=DTYPE_FLOAT), axis=0))

    @tf.function
    def call(self, input_tensor, training=False):
        if not training:
            # In case for testing the dataset object is unbatched.
            input_tensor = tf.reshape(input_tensor, [-1]+[dim for dim in self.input_shapes])

        return self.model(input_tensor, training=training)

    def get_config(self):
        config = super(DeepLab3Plus, self).get_config()
        config.update({'keep_prob': self.keep_prob, 
                       'num_classes': self.num_classes,
                       'output_stride': self.output_stride,
                       'skip_layer': self.skip_layer,
                       'pretrained_wei': self.pretrained_wei})
        return config
    
    def load_weights_from_file(self, by_name=True):
        # weights_path = tf.keras.utils.get_file(fname, WEIGHTS_PATH_X, cache_subdir='models')
        self.model.load_weights(self.pretrained_wei_path, by_name=by_name)

    def load_layer_weights_expand_last_layer(self, old_model):
        # old_model = dill.load(open(old_model_path, 'rb')).model
        for old_ly, ly in zip(old_model.layers, self.model.layers):
            for old_var, var in zip(old_ly.trainable_variables, ly.trainable_variables):
                if old_var.shape == var.shape:
                    var.assign(old_var.numpy())
                else:
                    new_wei_shape = tuple([dim if old_dim==dim else 1 for old_dim, dim in zip(old_var.shape, var.shape)])
                    if re.search(r'kernel', var.name):
                        # Use the same initialization method
                        init = tf.keras.initializers.GlorotUniform(RND_SEED)
                        var.assign(np.concatenate((old_var.numpy(), init(new_wei_shape).numpy()), axis=-1))
                        # Initialize as zero
                        # var.assign(np.concatenate((old_var.numpy(), np.zeros(new_wei_shape)), axis=-1))
                        # Initialize as minimum
                        # var.assign(np.concatenate((old_var.numpy(), np.min(old_var.numpy(), axis=-1, keepdims=True)), axis=-1))
                    else:
                        var.assign(np.concatenate((old_var.numpy(), np.zeros(new_wei_shape)), axis=-1))
                        # # Initialize as minimum
                        # var.assign(np.concatenate((old_var.numpy(), np.min(old_var.numpy(), axis=-1, keepdims=True)), axis=-1))
    
    def load_layer_weights_ignore_mismatch(self, old_model):
        # old_model = dill.load(open(old_model_path, 'rb')).model
        for old_ly, ly in zip(old_model.layers, self.model.layers):
            for old_var, var in zip(old_ly.trainable_variables, ly.trainable_variables):
                if old_var.shape == var.shape:
                    var.assign(old_var.numpy())

    



