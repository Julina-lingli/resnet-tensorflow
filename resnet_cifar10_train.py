import tensorflow as tf
import resnet_cifar10_model
import resnet_cifar10_input
import resnet_cifar10_flag as model_input
import time
from datetime import datetime
import os
import numpy as np

TRAIN_DIR = model_input.TRAIN_DIR


NUM_CLASSES = model_input.NUM_CLASSES

HEIGHT = model_input.HEIGHT

WIDTH = model_input.WIDTH

NUM_CHANNELS = model_input.NUM_CHANNELS

NUM_IMAGES = model_input.NUM_IMAGES

# 一般取值为64，128，256，512，1024
BATCH_SIZE = model_input.BATCH_SIZE
# Global constants describing the CIFAR-10 data set.
NUM_EPOCHS = model_input.NUM_EPOCHS

MAX_STEPS = model_input.MAX_STEPS

MOVING_AVERAGE_DECAY = model_input.MOVING_AVERAGE_DECAY  # The decay to use for the moving average.

NUM_EPOCHS_PER_DECAY = model_input.NUM_EPOCHS_PER_DECAY  # Epochs after which learning rate decays.

LEARNING_RATE_DECAY_FACTOR = model_input.LEARNING_RATE_DECAY_FACTOR  # Learning rate decay factor.

INITIAL_LEARNING_RATE = model_input.INITIAL_LEARNING_RATE  # Initial learning rate.

WEIGHT_DECAY = model_input.WEIGHT_DECAY


def loss(logits, labels, loss_filter_fn, weight_decay):
    # Calculate loss, which includes softmax cross entropy and L2 regularization.


    labels = tf.reshape(labels, [-1,])
    labels = tf.cast(labels, tf.int64)
    print("loss labels:", labels)
    # cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)与下面两句是一样的
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('loss', cross_entropy_mean)
    # Create a tensor named cross_entropy for logging purposes.

    tf.identity(cross_entropy_mean, name='cross_entropy')

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    # If no loss_filter_fn is passed, assume we want the default behavior,

    # which is that batch_normalization variables are excluded from loss.
    # 'batch_normalization'不在name列表中，则返回True
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    # 如果没有传递loss_filter_fn即loss_filter_fn=None，那么loss_filter_fn = exclude_batch_norm
    #  否则，loss_filter_fn就等于传进来的值
    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    # loss is computed using fp32 for numerical stability.
    # 将L2Loss添加到所有可训练变量除了BN中的beta，gamma
    l2_loss = weight_decay * tf.add_n(

        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()

         if loss_filter_fn(v.name)])
    tf.add_to_collection('loss', l2_loss)
    tf.summary.scalar('l2_loss', l2_loss)
    print("所有的权重衰减项（L2损失）:", tf.get_collection('loss'))
    total_loss = cross_entropy_mean + l2_loss

    tf.identity(total_loss, name='total_loss')

    return total_loss

def _add_loss_averages(total_loss):

    """Add summaries for losses in CIFAR-10 model.



    Generates moving average for all losses and associated summaries for

    visualizing the performance of the network.



     Args:

        total_loss: Total loss from loss().

    Returns:

        loss_averages_op: op for generating moving averages of losses.

    """

    # Compute the moving average of all individual losses and the total loss.
    # 计算所有单个损失和总损失的移动平均
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')

    losses = tf.get_collection('loss')
    print("权重损失和交叉损失：", losses)
    # loss_averages_op = loss_averages.apply(losses + [total_loss])
    loss_averages_op = loss_averages.apply([total_loss])
    return loss_averages_op

def model_train_fn(total_loss, global_steps):

    """Train CIFAR-10 model.



    Create an optimizer and apply to all trainable variables. Add moving

    average for all trainable variables.



    Args:

        total_loss: Total loss from loss().

        global_step: Integer Variable counting the number of training steps

        processed.

    Returns:

        train_op: op for training.

    """

    # Variables that affect learning rate.

    num_batches_per_epoch = NUM_IMAGES["train"] / BATCH_SIZE
    print("num_batches_per_epoch:", num_batches_per_epoch)
    # 设置每隔多少个epoch后就衰减学习率
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    print("decay_steps:", decay_steps)
    # Decay the learning rate exponentially based on the number of steps.
    # 根据步骤数以指数方式衰减学习率
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
        global_steps,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)

    #tf.summary.scalar('learning_rate', lr)



    # Generate moving averages of all losses and associated summaries.

    loss_averages_op = _add_loss_averages(total_loss)
    # loss_averages_op = total_loss
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # 神经网络训练开始前很难估计所需的迭代次数global_step，系统在训练时会自动更新global_step,
    # 学习速率第一次训练开始变化，global_steps每次自动加1
    # with tf.control_dependencies([loss_averages_op, update_ops]):
    with tf.control_dependencies([loss_averages_op]):
        method = tf.train.GradientDescentOptimizer(lr)
        optimizer = method.minimize(total_loss, global_step=global_steps)

    """
    # Add histograms for trainable variables.

    for var in tf.trainable_variables():

        tf.summary.histogram(var.op.name, var)



    # Add histograms for gradients.

    for grad, var in grads:

        if grad is not None:

            tf.summary.histogram(var.op.name + '/gradients', grad)


    """
    # Track the moving averages of all trainable variables.

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_steps)
    #更新列表中的变量tf.trainable_variables()所有参加训练的变量参数
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([optimizer]):

        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    return variables_averages_op

def predict(predict_logits, _y):
    _y = tf.reshape(_y, [-1, ])
    _y = tf.cast(_y, tf.int64)
    print("_y_test", _y)
    # pred返回每个样本预测类型的概率
    pred = tf.nn.softmax(logits=predict_logits, name="pred")
    print("pred:", pred)
    # Calculate predictions.
    # tf.nn.in_top_k选择每个样本预测类型的最大概率，比较该最大概率的索引值是否与标签y_test中的值相匹配，返回布尔型
    top_k_op = tf.nn.in_top_k(pred, _y, 1)
    # top_k_op = predict_logits
    print("top_k_op:", top_k_op)
    return top_k_op

def resnet_model_fn(model_class, resnet_size, train_dir):
    # 因为数据读取的计算图在该函数外，而两者需要在同一个计算图中，故不能重新设置计算图
    with tf.Graph().as_default():
        _features, _labels = resnet_cifar10_input.read_data(is_training=True,
                                                    batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)
        print("do_train graph:", tf.get_default_graph())

        # print("_features:", _features.shape)
        # print("_labels:", _labels.shape)
        # global_step = tf.train.get_or_create_global_step()
        step = 0
        global_step = tf.Variable(0, trainable=False)
        # Build a Graph that computes the logits predictions from the

        # inference model.
        logits = resnet_cifar10_model.cifar10_model_fn(_features, resnet_size, model_class)

        # Empirical testing showed that including batch_normalization variables

        # in the calculation of regularized loss helped validation accuracy

        # for the CIFAR-10 dataset, perhaps because the regularization prevents

        # overfitting on the small data set. We therefore include all vars when

        # regularizing and computing loss during training.

        def loss_filter_fn(_):

            return True
        # Calculate loss.
        total_loss = loss(logits, _labels, loss_filter_fn, WEIGHT_DECAY)

        # Build a Graph that trains the model with one batch of examples and

        # updates the model parameters.

        train_op = model_train_fn(total_loss, global_step)

        with tf.control_dependencies([train_op]):
            predict_op = predict(logits, _labels)

        print("global_variables count:", len(tf.global_variables()))
        print("global_variables:", tf.global_variables())
        print("trainable_variables count:", len(tf.trainable_variables()))
        print("trainable_variables:", tf.trainable_variables())
        # 用tf.train.Saver()创建一个Saver来管理模型中的所有变量
        saver = tf.train.Saver(tf.global_variables())

        num_examples = 0
        true_count = 0
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(tf.global_variables_initializer())
            # 若batch size为64，则5000个数据集可划分为5000/64个batch，
            # dataset在执行iterator.get_next()时返回一个batch的数据集，
            # 那么多个时期比如100个epoch迭代的数据集总共的batch 个数就为100*（5000/64）
            print("MAX_STEPS: ", MAX_STEPS)
            try:
                while step < MAX_STEPS:
                    print("_features", sess.run(_features).shape)
                    print("predict_logits", sess.run(logits).shape)
                    # 记录运行计算图一次的时间
                    start_time = time.time()
                    _, _total_loss = sess.run([train_op, total_loss])
                    duration_time = time.time() - start_time
                    step += 1
                    print("total_loss", total_loss)
                    print("_total_loss", _total_loss)
                    print("step:", step)
                    if step % 10 == 0:
                        num_examples_per_step = sess.run(_features).shape[0]
                        examples_per_sec = num_examples_per_step / float(duration_time)
                        sec_per_batch = float(duration_time)

                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now(), step, _total_loss,
                                             examples_per_sec, sec_per_batch))
                        predictions = sess.run([predict_op])
                        true_count = np.sum(predictions)
                        print("num_examples", num_examples_per_step)
                        print('%s: precision @  = %.3f' % (datetime.now(), true_count / num_examples_per_step))
                    """
                    if step % 100 == 0:
                        # 添加summary日志
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                    """
                    # num_examples += sess.run(_features).shape[0]
                    # print("num_examples", num_examples)
                    # 获得一个batch大小的样本中预测正确的样本个数
                    # true_count += np.sum(predictions)
                    # print("true_count:", true_count)
                    # 定期保存模型检查点
                    if step % 100 == 0 or (step + 1) == MAX_STEPS:
                        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

                        # Compute precision @ 1.
                        # precision = true_count / num_examples
                        # true_count = 0
                        # num_examples = 0
                        # print('%s: precision @ step:%s = %.3f' % (datetime.now(), step, precision))

            except tf.errors.OutOfRangeError:
                print("End of dataset per epoch")

def train_main():
    train_dir = TRAIN_DIR
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)

    tf.gfile.MakeDirs(train_dir)
    # 重置tensorflow的graph，确保神经网络可多次运行
    tf.reset_default_graph()
    tf.set_random_seed(1908)
    resnet_model_fn(resnet_cifar10_model.Cifar10Model, model_input.RESNET_SIZE, train_dir)

train_main()
