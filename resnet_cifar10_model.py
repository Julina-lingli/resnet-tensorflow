import tensorflow as tf
import resnet_cifar10_flag as model_input

_BATCH_NORM_DECAY = model_input._BATCH_NORM_DECAY

_BATCH_NORM_EPSILON = model_input._BATCH_NORM_EPSILON

NUM_CLASSES = model_input.NUM_CLASSES

HEIGHT = model_input.HEIGHT

WIDTH = model_input.WIDTH

NUM_CHANNELS = model_input.NUM_CHANNELS


################################################################################

# Convenience functions for building the ResNet model.

################################################################################
def batch_norm(inputs, training, data_format='channels_last'):
    """Performs a batch normalization using a standard set of parameters."""

    # We set fused=True for a significant performance boost. See

    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops

    return tf.layers.batch_normalization(

        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,

        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,

        scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
    """Pads the input along the spatial dimensions independently of input size.



  Args:

    inputs: A tensor of size [batch, channels, height_in, width_in] or

      [batch, height_in, width_in, channels] depending on data_format.

    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.

                 Should be a positive integer.

    data_format: The input format ('channels_last' or 'channels_first').



  Returns:

    A tensor with the same format as the input with the data either intact

    (if kernel_size == 1) or padded (if kernel_size > 1).

  """
    # n+2p-f+1=n ->p=（f-1）/2，[pad_beg, pad_end]取值一般都为[1, 1], [2, 2]，[3, 3]
    pad_total = kernel_size - 1

    pad_beg = pad_total // 2

    pad_end = pad_total - pad_beg



    if data_format == 'channels_first':

        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])

    else:

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])

    return padded_inputs



def conv2d_fixed_padding(inputs, filters, kernel_size, strides):

  """Strided 2-D convolution with explicit padding."""

  # The padding is consistent and is based only on `kernel_size`, not on the

  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  # padding是为了保证输入输出的尺寸一致，而stride是缩小尺寸，故当stride>1时，显示地padding即卷积之前就padding
  # 卷积的时候padding设置为VALID;当stride=1的时候，就直接在tf.layers.conv2d中一起padding
  # tf.variance_scaling_initializer()默认生成truncated normal   distribution（截断正态分布） 的随机数*+

  if strides > 1:

    inputs = fixed_padding(inputs, kernel_size)



  return tf.layers.conv2d(

      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,

      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,

      kernel_initializer=tf.variance_scaling_initializer())

################################################################################

# ResNet block definitions.

################################################################################
def _building_block_v2(inputs, filters, training, projection_shortcut, strides):

    """A single block for ResNet v2, without a bottleneck.



  Batch normalization then ReLu then convolution as described by:

    Identity Mappings in Deep Residual Networks

    https://arxiv.org/pdf/1603.05027.pdf

    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.



  Args:

    inputs: A tensor of size [batch, channels, height_in, width_in] or

      [batch, height_in, width_in, channels] depending on data_format.

    filters: The number of filters for the convolutions.

    training: A Boolean for whether the model is in training or inference

      mode. Needed for batch normalization.

    projection_shortcut: The function to use for projection shortcuts

      (typically a 1x1 convolution when downsampling the input).

    strides: The block's stride. If greater than 1, this block will ultimately

      downsample the input.

    data_format: The input format ('channels_last' or 'channels_first').



  Returns:

    The output tensor of the block; shape should match inputs.

  """
    # 对inputs进行预激活处理，bn，relu
    shortcut = inputs

    inputs = batch_norm(inputs, training)

    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU

    # since it performs a 1x1 convolution.

    if projection_shortcut is not None:

        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides)

    inputs = batch_norm(inputs, training)

    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1)

    return inputs + shortcut


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut, strides):
    """
    A single block for ResNet v2, with a bottleneck.



    Similar to _building_block_v2(), except using the "bottleneck" blocks

    described in:

    Convolution then batch normalization then ReLU as described by:

    Deep Residual Learning for Image Recognition

    https://arxiv.org/pdf/1512.03385.pdf

    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.



    Adapted to the ordering conventions of:

    Batch normalization then ReLu then convolution as described by:

    Identity Mappings in Deep Residual Networks

    https://arxiv.org/pdf/1603.05027.pdf

    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.



    Args:

    inputs: A tensor of size [batch, channels, height_in, width_in] or

    [batch, height_in, width_in, channels] depending on data_format.

    filters: The number of filters for the convolutions.

    training: A Boolean for whether the model is in training or inference

    mode. Needed for batch normalization.

    projection_shortcut: The function to use for projection shortcuts

    (typically a 1x1 convolution when downsampling the input).

    strides: The block's stride. If greater than 1, this block will ultimately

    downsample the input.

    data_format: The input format ('channels_last' or 'channels_first').



    Returns:

    The output tensor of the block; shape should match inputs.

    """
    shortcut = inputs

    inputs = batch_norm(inputs, training)

    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.

    if projection_shortcut is not None:

        shortcut = projection_shortcut(inputs)
    print("shortcut", shortcut)


    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1)

    inputs = batch_norm(inputs, training)

    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides)

    inputs = batch_norm(inputs, training)

    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)

    return inputs + shortcut

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides, training, name):

    """Creates one layer of blocks for the ResNet model.



  Args:

    inputs: A tensor of size [batch, channels, height_in, width_in] or

      [batch, height_in, width_in, channels] depending on data_format.

    filters: The number of filters for the first convolution of the layer.

    bottleneck: Is the block created a bottleneck block.

    block_fn: The block to use within the model, either `building_block` or

      `bottleneck_block`.

    blocks: The number of blocks contained in the layer.

    strides: The stride to use for the first convolution of the layer. If

      greater than 1, this layer will ultimately downsample the input.

    training: Either True or False, whether we are currently training the

      model. Needed for batch norm.

    name: A string name for the tensor output of the block layer.

    data_format: The input format ('channels_last' or 'channels_first').



  Returns:

    The output tensor of the block layer.

  """
    # Bottleneck blocks end with 4x the number of filters as they start with

    filters_out = filters * 4 if bottleneck else filters
    print("block_layer filters_out:", filters_out)
    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)

    # Only the first block per block_layer uses projection_shortcut and strides
    # 从第二个block开始，projection_shortcut=None，strides=1
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides)
    print("block_layer blocks:", blocks)
    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1)

    return tf.identity(inputs, name)

class Model(object):
    # Base class for building the Resnet Model.
    def __init__(self, resnet_size, is_bottleneck, num_classes, num_filters,

                kernel_size, conv_stride, first_pool_size, first_pool_stride,

                block_sizes, block_strides):

        """Creates a model for classifying an image.



           Args:

             resnet_size: A single integer for the size of the ResNet model.

             bottleneck: Use regular blocks or bottleneck blocks.

             num_classes: The number of classes used as labels.

             num_filters: The number of filters to use for the first block layer

               of the model. This number is then doubled for each subsequent block

               layer.

             kernel_size: The kernel size to use for convolution.

             conv_stride: stride size for the initial convolutional layer

             first_pool_size: Pool size to be used for the first pooling layer.

               If none, the first pooling layer is skipped.

             first_pool_stride: stride size for the first pooling layer. Not used

               if first_pool_size is None.

             block_sizes: A list containing n values, where n is the number of sets of

               block layers desired. Each value should be the number of blocks in the

               i-th set.

             block_strides: List of integers representing the desired stride size for

               each of the sets of block layers. Should be same length as block_sizes.

             resnet_version: Integer representing which version of the ResNet network

               to use. See README for details. Valid values: [1, 2]

             data_format: Input format ('channels_last', 'channels_first', or None).

               If set to None, the format is dependent on whether a GPU is available.

             dtype: The TensorFlow dtype to use for calculations. If not specified

               tf.float32 is used.



           Raises:

             ValueError: if invalid version is selected.

        """

        self.resnet_size = resnet_size

        self.is_bottleneck = is_bottleneck

        self.block_fn = _bottleneck_block_v2 if is_bottleneck else _building_block_v2

        self.num_classes = num_classes

        self.num_filters = num_filters

        self.kernel_size = kernel_size

        self.conv_stride = conv_stride

        self.first_pool_size = first_pool_size

        self.first_pool_stride = first_pool_stride

        self.block_sizes = block_sizes

        self.block_strides = block_strides

    def __call__(self, inputs, is_training):
        """Add operations to classify a batch of input images.



           Args:

             inputs: A Tensor representing a batch of input images.

             training: A boolean. Set to True to add operations required only when

               training the classifier.



           Returns:

             A logits Tensor with shape [<batch_size>, self.num_classes].

           """

        with tf.variable_scope('resnet_model'):
            inputs = conv2d_fixed_padding(
                        inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
                        strides=self.conv_stride)
            inputs = tf.identity(inputs, 'initial_conv')

            if self.first_pool_size:
                inputs = tf.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='SAME')
                inputs = tf.identity(inputs, 'initial_max_pool')
            # num_blocks：每个block_layer里面包含几个 残差块
            print("block_sizes():", self.block_sizes)
            for i, num_blocks in enumerate(self.block_sizes):
                # 当前block中filter的个数是上一个block中filter个数的2倍
                num_filters = self.num_filters * (2 ** i)
                print("num_filters: ", num_filters)
                inputs = block_layer(
                    inputs=inputs, filters=num_filters, bottleneck=self.is_bottleneck,
                    block_fn=self.block_fn, blocks=num_blocks,
                    strides=self.block_strides[i], training=is_training,
                    name='block_layer{}'.format(i + 1))

            # Only apply the BN and ReLU for model that does pre_activation in each

            # building/bottleneck block, eg resnet V2.
            inputs = batch_norm(inputs, training=is_training)

            inputs = tf.nn.relu(inputs)
            print("relu shape：", inputs.shape)
            # The current top layer has shape

            # `batch_size x pool_size x pool_size x final_size`.

            # ResNet does an Average Pooling layer over pool_size,

            # but that is the same as doing a reduce_mean. We do a reduce_mean

            # here because it performs better than AveragePooling2D.

            axes = [1, 2]
            inputs = tf.reduce_mean(inputs, axes, keepdims=True)
            print("reduce_mean shape：", inputs.shape)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            inputs = tf.squeeze(inputs, axes)
            print("squeeze shape：", inputs.shape)
            inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
            print("dense shape：", inputs.shape)
            inputs = tf.identity(inputs, 'final_dense')

            # We don't apply softmax here because

            # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits

            # and performs the softmax internally for efficiency.
            return inputs

###############################################################################

# Running the model

###############################################################################
class Cifar10Model(Model):
    """Model class with appropriate defaults for CIFAR-10 data."""

    def __init__(self, resnet_size, num_classes=NUM_CLASSES):
        """These are the parameters that work for CIFAR-10 data.
        Args:

          resnet_size: The number of convolutional layers needed in the model.

          data_format: Either 'channels_first' or 'channels_last', specifying which

            data format to use when setting up the model.

          num_classes: The number of output classes needed from the model. This

            enables users to extend the same model to their own datasets.

        Raises:

          ValueError: if invalid resnet_size is chosen

        """

        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)

        num_blocks = (resnet_size - 2) // 6
        print("num_blocks: ", num_blocks)
        super(Cifar10Model, self).__init__(

            resnet_size=resnet_size,

            is_bottleneck=True,

            num_classes=num_classes,

            num_filters=16,

            kernel_size=3,

            conv_stride=1,

            first_pool_size=2,

            first_pool_stride=2,

            block_sizes=[num_blocks] * 3,

            block_strides=[1, 2, 2])


def cifar10_model_fn(features, resnet_size, model_class, is_training=True):
    """Model function for CIFAR-10."""

    features = tf.reshape(features, [-1, HEIGHT, WIDTH, NUM_CHANNELS])

    # Learning rate schedule follows arXiv:1512.03385 for ResNet-56 and under.

    # Generate a summary node for the images

    # tf.summary.image('images', features, max_outputs=6)

    # 创建一个模型的实例
    model = model_class(resnet_size)

    #  调用模型创建
    logits = model(features, is_training)

    return logits