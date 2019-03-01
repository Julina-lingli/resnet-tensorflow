import tarfile
import pathlib
import os
import tensorflow as tf
import resnet_cifar10_flag as model_input


FILENAME = model_input.FILENAME
SRC_TAR_DIR = model_input.SRC_TAR_DIR
DST_PATH = model_input.DST_PATH
TRAIN_DIR = model_input.TRAIN_DIR

HEIGHT = model_input.HEIGHT

WIDTH = model_input.WIDTH

NUM_CHANNELS = model_input.NUM_CHANNELS

_DEFAULT_IMAGE_BYTES = model_input._DEFAULT_IMAGE_BYTES

# The record is the image plus a one-byte label

_RECORD_BYTES = model_input._RECORD_BYTES

NUM_CLASSES = model_input.NUM_CLASSES

_NUM_DATA_FILES = model_input._NUM_DATA_FILES

NUM_IMAGES = model_input.NUM_IMAGES



def untar_dir(src, dstPath):
    tarHandle = tarfile.open(src, "r:gz")
    for filename in tarHandle.getnames():
        print (filename)
    tarHandle.extractall(dstPath)
    tarHandle.close()

def cifar10_extract(src_tar, dst_path, file_name):
    cifar10_dir = os.path.join(dst_path, file_name)
    print(cifar10_dir)
    path = pathlib.Path(cifar10_dir)
    if (not (path.exists())):
        untar_dir(src_tar, dst_path)
    return cifar10_dir

###############################################################################

# Data processing

###############################################################################

def get_filenames(is_training):

    """Returns a list of filenames."""
    data_dir = cifar10_extract(SRC_TAR_DIR, DST_PATH, FILENAME)

    assert tf.gfile.Exists(data_dir), (

        'Run cifar10_extract.py first to download and extract the '
    
        'CIFAR-10 data.')



    if is_training:

        return [os.path.join(data_dir, 'data_batch_%d.bin' % i)

                for i in range(1, _NUM_DATA_FILES + 1)]

    else:

        return [os.path.join(data_dir, 'test_batch.bin')]


def preprocess_image(image, is_training):

    """Preprocess a single image of layout [height, width, depth]."""

    if is_training:
        print("is_training:", is_training)
        # Resize the image to add four extra pixels on each side.

        image = tf.image.resize_image_with_crop_or_pad(

                    image, HEIGHT + 8, WIDTH + 8)

        # Randomly crop a [HEIGHT, WIDTH] section of the image.

        image = tf.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

        # Randomly flip the image horizontally.

        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.

    image = tf.image.per_image_standardization(image)

    return image



def parse_record(raw_record, is_training):

    """Parse CIFAR-10 image and label from a raw record."""

    # Convert bytes to a vector of uint8 that is record_bytes long.

    record_vector = tf.decode_raw(raw_record, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32

    # and then to one-hot.
    label = tf.strided_slice(record_vector, [0], [1])
    # label = tf.cast(record_vector[0], tf.int32)

    label = tf.reshape(label, [1])
    print("lable:", label)
    # The remaining bytes after the label represent the image, which we reshape

    # from [depth * height * width] to [depth, height, width].
    """
    depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                       [NUM_CHANNELS, HEIGHT, WIDTH])
    """
    depth_major = tf.reshape(tf.strided_slice(record_vector, [1],[_RECORD_BYTES]),
                             [NUM_CHANNELS, HEIGHT, WIDTH])
    # Convert from [depth, height, width] to [height, width, depth], and cast as

    # float32.

    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    image = preprocess_image(image, is_training)

    image = tf.cast(image, tf.float32)

    return image, label

def input_fn(is_training, batch_size, num_epochs=1,

             num_parallel_batches=1, parse_record_fn=parse_record):

    """Input function which provides batches for train or eval.



    Args:

    is_training: A boolean denoting whether the input is for training.

    data_dir: The directory containing the input data.

    batch_size: The number of samples per batch.

    num_epochs: The number of epochs to repeat the dataset.

    dtype: Data type to use for images/features

    datasets_num_private_threads: Number of private threads for tf.data.

    num_parallel_batches: Number of parallel batches for tf.data.

    parse_record_fn: Function to use for parsing the records.



    Returns:

    A dataset that can be used for iteration.

    """

    filenames = get_filenames(is_training)

    print("filenames:", filenames)
    print("batch_size:", batch_size)
    print("num_epochs:", num_epochs)
    dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

    # Prefetches a batch at a time to smooth out the time taken to load input

    # files for shuffling and processing.
    if is_training:
        print("is_training:", is_training)
        dataset = dataset.prefetch(buffer_size=batch_size)

        # Shuffles records before repeating to respect epoch boundaries.

        dataset = dataset.shuffle(buffer_size=NUM_IMAGES['train'])

    print("DATASET", dataset)
    # 解析从二进制文件中读取的一个元素，转换为（label， image）
    dataset = dataset.map(lambda value: parse_record_fn(value, is_training))
    # dataset = dataset.map(lambda value:parse_record_fn(value, is_training),
    #                       num_parallel_calls=num_parallel_batches)
    print("DATASET_1", dataset)

    # dataset = dataset.shuffle(buffer_size=100)
    # print("DATASET_2", dataset)
    dataset = dataset.batch(batch_size)
    print("DATASET_3", dataset)

    # Repeats the dataset for the number of epochs to train.test阶段则设置为1默认值
    # 对repeat方法不设置重复次数,就不用算repeat的次数
    dataset = dataset.repeat(num_epochs)
    # dataset = dataset.repeat()
    print("DATASET_4", dataset)

    return dataset

def read_data(is_training, batch_size, num_epochs):

    dataset = input_fn(is_training, batch_size, num_epochs)

    iterator = dataset.make_one_shot_iterator()
    # 返回的one_element为batch_size个（_labels, _features）
    next_feature, next_label = iterator.get_next()
    print("next_label", next_label)
    print("next_feature", next_feature)

    return next_feature, next_label