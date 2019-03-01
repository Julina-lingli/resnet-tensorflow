import tensorflow as tf
import numpy as np
import resnet_cifar10_model
import resnet_cifar10_input
import resnet_cifar10_flag as model_input
from datetime import datetime
import time

CHECKPOINT_DIR = model_input.TRAIN_DIR
FILENAME = model_input.FILENAME
SRC_TAR_DIR = model_input.SRC_TAR_DIR
DST_PATH = model_input.DST_PATH
TRAIN_DIR = model_input.TRAIN_DIR

NUM_IMAGES = model_input.NUM_IMAGES
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = NUM_IMAGES["validation"]
RUN_ONCE = False
# BATCH_SIZE = model_input.BATCH_SIZE
BATCH_SIZE = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
BATCH_NUM = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / BATCH_SIZE
EVAL_INTERVAL_SECS = 60 * 2

MOVING_AVERAGE_DECAY = model_input.MOVING_AVERAGE_DECAY     # The decay to use for the moving average.

def eval_once(saver, top_k_op):
# def eval_once(saver, predict_logits, _features, _y_test):
    """Run Eval once.
    Args:

      saver: Saver.

      summary_writer: Summary writer.

      top_k_op: Top K op.

      summary_op: Summary op.

    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:

            # Restores from checkpoint
            # 加载最新的模型
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("global_variables count:", len(tf.global_variables()))
            print("global_variables:", tf.global_variables())
            print("trainable_variables count:", len(tf.trainable_variables()))
            print("trainable_variables:", tf.trainable_variables())
            # Assuming model_checkpoint_path looks something like:

            #   /my-favorite-path/cifar10_train/model.ckpt-0,

            # extract global_step from it.

            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("global_step:", global_step)

        else:

            print('No checkpoint file found')

            return

        true_count = 0
        step = 0
        print("BATCH_NUM", BATCH_NUM)
        """
        while step < BATCH_NUM+1:
            print("step", stepl)
            # print("_features", len(sess.run(_features)))
            _y_test = tf.reshape(_y_test, [-1, ])
            _y_test = tf.cast(_y_test, tf.int64)

            # print("_y_test", sess.run(_y_test).shape)
            print("predict_logits", sess.run(predict_logits).shape)
            # 获得一个batch 的测试样本的预测正确的，是一个布尔类型的向量表
            pred = tf.nn.softmax(logits=predict_logits, name="pred")
            print("pred", len(sess.run(pred)))
            top_k_op = tf.nn.in_top_k(pred, _y_test, 1)
            predictions = sess.run([top_k_op])
            # 获得一个batch大小的样本中预测正确的样本个数
            true_count += np.sum(predictions)
            print("true_count:", true_count)

            step +=1
        """

        try:
            while True:
                """
                print("_features", len(sess.run(_features)))
                _y_test = tf.reshape(_y_test, [-1, ])
                _y_test = tf.cast(_y_test, tf.int64)

                print("_y_test", sess.run(_y_test).shape)
                print("predict_logits", sess.run(predict_logits).shape)
                #获得一个batch 的测试样本的预测正确的，是一个布尔类型的向量表
                pred = tf.nn.softmax(logits=predict_logits, name="pred")
                print("pred", len(sess.run(pred)))
                top_k_op = tf.nn.in_top_k(pred, _y_test, 1)
                """
                predictions = sess.run([top_k_op])
                #获得一个batch大小的样本中预测正确的样本个数
                true_count += np.sum(predictions)
                print("true_count:", true_count)
        except tf.errors.OutOfRangeError:
            print("End of dataset for test")

        # Compute precision @ 1.
        total_sample_count = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        precision = true_count / total_sample_count

        print('%s: precision @ step:%s = %.3f' % (datetime.now(), global_step, precision))

def predict_op(X_test, y_test, resnet_size, model_class, is_training):
    """
    with tf.Session(graph=graph_test_data) as sess:
        _y_test = sess.run(y_test)
        _y_test = _y_test.flatten()
    print("_y_test", _y_test.shape)
    """
    _y_test = tf.reshape(y_test, [-1, ])
    _y_test = tf.cast(_y_test, tf.int64)
    print("_y_test", _y_test)
    # Build a Graph that computes the logits predictions
    predict_logits = resnet_cifar10_model.cifar10_model_fn(X_test, resnet_size, model_class, is_training)

    # pred返回每个样本预测类型的概率
    pred = tf.nn.softmax(logits=predict_logits, name="pred")
    print("pred:", pred)
    # Calculate predictions.
    # tf.nn.in_top_k选择每个样本预测类型的最大概率，比较该最大概率的索引值是否与标签y_test中的值相匹配，返回布尔型
    top_k_op = tf.nn.in_top_k(pred, _y_test, 1)
    # top_k_op = predict_logits
    print("top_k_op:", top_k_op)
    return top_k_op

import cnn_cifar10_input
def evaluate_main():
    # tf.reset_default_graph()
    # with tf.Graph().as_default() as graph_test_data:
    #
    _features, _labels = resnet_cifar10_input.read_data(is_training=False,
                                                        batch_size=BATCH_SIZE, num_epochs=1)
    # _features, _labels = cnn_cifar10_input.read_test(SRC_TAR_DIR, DST_PATH, FILENAME, BATCH_SIZE)

    top_k_op = predict_op(_features, _labels, model_input.RESNET_SIZE, resnet_cifar10_model.Cifar10Model,
                          is_training=False)

    # Restore the moving average version of the learned variables for eval.

    variable_averages = tf.train.ExponentialMovingAverage(

        MOVING_AVERAGE_DECAY)

    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    while True:

        eval_once(saver, top_k_op)
        # eval_once(saver, top_k_op, _features, _labels)

        if RUN_ONCE:

            break

        time.sleep(EVAL_INTERVAL_SECS)

with tf.device('/cpu:0'):
    evaluate_main()