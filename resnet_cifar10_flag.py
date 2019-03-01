"""Contains utility and supporting functions for ResNet.

"""
FILENAME = "cifar-10-batches-bin"
SRC_TAR_DIR = "D:\datasets\cifar-10-binary.tar.gz"
DST_PATH = "D:\datasets"
TRAIN_DIR = "D:\datasets\cifar10_train_resnet"

# RESNET_SIZE = 50
RESNET_SIZE = 8

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

HEIGHT = 32

WIDTH = 32

NUM_CHANNELS = 3

_DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS

# The record is the image plus a one-byte label

_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1

NUM_CLASSES = 10

_NUM_DATA_FILES = 5

NUM_IMAGES = {

    'train': 50000,

    'validation': 10000,

}

#一般取值为64，128，256，512，1024
BATCH_SIZE = 1024
# BATCH_SIZE = 1
# Global constants describing the CIFAR-10 data set.
NUM_EPOCHS = 2000
MAX_STEPS = NUM_EPOCHS * (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE)

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.

NUM_EPOCHS_PER_DECAY = 10.0  # Epochs after which learning rate decays.

LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.



_BATCH_NORM_DECAY = 0.997

_BATCH_NORM_EPSILON = 1e-5

# Weight decay of 2e-4 diverges from 1e-4 decay used in the ResNet paper

# and seems more stable in testing. The difference was nominal for ResNet-56.

WEIGHT_DECAY = 2e-4
