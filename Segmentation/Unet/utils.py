import numpy as np
import tensorflow as tf
import six
from tensorflow.keras.losses import Loss
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops.losses import util as tf_losses_util


"""
Utility functions
"""

# Note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    # Convert labels to int 32 for tf cross-entropy function.
    if y.dtype != tf.int32:
        y = tf.cast(y, tf.int32)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=-1, output_type=tf.int32), 
                                tf.cast(y_true, tf.int32))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def class_tp(y_pred, y_true, num_classes):
    cla_acc = []
    if y_true.dtype != tf.int32:
        y_true = tf.cast(y_true, tf.int32) # This step won't change the original object.
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=-1, output_type=tf.int32), y_true)
    for lab in range(num_classes):
        cla_all = tf.equal(lab, y_true)
        cla_corr = tf.logical_and(cla_all, correct_prediction)
        num_cla = tf.reduce_sum(tf.cast(cla_all, tf.float32)).numpy()
        if num_cla>0:
            cla_acc.append(tf.reduce_sum(tf.cast(cla_corr, tf.float32)).numpy()/num_cla)
        else:
            cla_acc.append(-1)
    return cla_acc


def soft_dice_loss(y_true, y_pred, from_logits=False, epsilon=1e-6): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
    
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022

        Modified from https://www.jeremyjordan.me/semantic-segmentation/
    '''
    
    # skip the batch and class axis for calculating Dice score
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    if y_true.dtype != tf.float32:
        y_true = tf.cast(y_true, tf.float32)
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * tf.reduce_sum(y_pred * y_true, axes)
    # denominator = tf.reduce_sum(tf.square(y_pred) + tf.square(y_true), axes)
    denominator = tf.reduce_sum(y_pred + y_true, axes)
    
    return 1 - tf.reduce_mean(numerator / (denominator + epsilon)) # average over classes and batch


def sparse_soft_dice_loss(y_true, y_pred, from_logits=False, epsilon=1e-6):
    # tf.one_hot default dtype is tf.float32
    # https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/ops/array_ops.py#L3539
    y_true_oh = tf.one_hot(indices=y_true, depth=y_pred.shape[-1])
    return soft_dice_loss(y_true_oh, y_pred, from_logits=from_logits, epsilon=epsilon)


class LossFunctionWrapper(Loss):
    """Wraps a loss function in the `Loss` class.
    Args:
        fn: The loss function to wrap, with signature `fn(y_true, y_pred,
        **kwargs)`.
        reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
        Default value is `AUTO`. `AUTO` indicates that the reduction option will
        be determined by the usage context. For almost all cases this defaults to
        `SUM_OVER_BATCH_SIZE`.
        When used with `tf.distribute.Strategy`, outside of built-in training
        loops such as `tf.keras` `compile` and `fit`, using `AUTO` or
        `SUM_OVER_BATCH_SIZE` will raise an error. Please see
        https://www.tensorflow.org/tutorials/distribute/custom_training
        for more details on this.
        name: (Optional) name for the loss.
        **kwargs: The keyword arguments that are passed on to `fn`.

    Reference: https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/losses.py#L180
    """

    def __init__(self,
                fn,
                reduction=losses_utils.ReductionV2.AUTO,
                name=None,
                **kwargs):
        super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.
        Args:
            y_true: Ground truth values.
            y_pred: The predicted values.
        Returns:
            Loss values per sample.
        """
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
            y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(y_pred, y_true)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SparseSoftDiceLoss(LossFunctionWrapper):
    """
        Reference: https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/losses.py#L473
        Other kinds of loss: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
    """
    def __init__(self,
               from_logits=False,
               reduction=losses_utils.ReductionV2.AUTO,
               name='sparse_soft_dice_loss',
               epsilon=1e-6):
        super(SparseSoftDiceLoss, self).__init__(
            sparse_soft_dice_loss,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            epsilon=epsilon)

class SoftDiceLoss(LossFunctionWrapper):
    def __init__(self,
               from_logits=False,
               reduction=losses_utils.ReductionV2.AUTO,
               name='soft_dice_loss',
               epsilon=1e-6):
        super(SoftDiceLoss, self).__init__(
            soft_dice_loss,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            epsilon=epsilon)
