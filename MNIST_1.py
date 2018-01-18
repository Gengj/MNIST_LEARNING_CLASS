# -*- coding:utf-8 -*-

"""
-------------------------------------------------
    @Author:        GengJia
    @Contact:       35285770@qq.com
    @Site:          https://github.com/Gengj
-------------------------------------------------
    @Version:       1.0
    @License:       (C) Copyright 2013-2020 
    @File:          MNIST_1.py
    @Time:          2018/1/16 下午7:25
    @Desc:
-------------------------------------------------
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MINTS基本参数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络参数

LAYER1_NODE = 500  # 神经网路含有一个隐藏层，该层含有500个节点
BATCH_SIZE = 100  # 一个batch中含有的数据个数

LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAT = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAING_STEP = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 辅助函数
# 给定神经网路的输入和所有参数，计算神经网络的前向传播结果
# 定义一个使用ReLU激活函数的三层全连接神经网络
# 通过加入隐藏层实现了多层网络结构
# 通过ReLU激活函数实现去线性化
# 函数支持传入用于计算参数平均值的类，方便在测试时使用滑动平均模型
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        # 计算输出层的前向传播结果。因为在计算损失函数时会一并计算softmax函数
        # 所有在这里不需要加入激活函数，而且不加入softmax函数不会影响预测结果
        # 因为预测时，使用的时不同类别对应节点输出值的相对大小，有没有softmax层对分类结果的计算没有影响
        # 因此，在计算整个神经网络的前向传播时，可以不加入最后的softmax层
        return tf.matmul(layer1, weights2) + biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))
                            + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为None
    # 因此函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量
    # 这个变量不需要计算滑动平均值，所以这个变量一般为不可训练的变量
    # 使用tensorflow训练神经网络时，一般都会将代表训练论述的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均，其他辅助变量例如global_step就不需要训练了
    # variable_averages.apply返回：计算图上集合tf.GraphKeys.TRAINABLE_VARIABLES中的元素
    # 这个集合中的元素即所有没有指定trainable=False的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用滑动平均之后的前向传播结果
    # 滑动平均本身不会改变变量的取值，而是会维护一个影子变量来记录其滑动平均值
    # 所以当需要使用这个滑动平均值时，需要明确调用average函数
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    # 当分类问题只有一个正确答案时，可以使用tensorflow提供的nn.sparse_softmax_cross_entropy_with_logits
    # 函数来计算交叉熵
    # mnist问题的图片中只包含0～9中的一个数字，因此可以使用这个函数计算
    # nn.sparse_softmax_cross_entropy_with_logits函数
    # '''
    #     第一个参数：神经网络不包含softmax层的前向传播结果
    #     第二个参数：训练数据的正确答案
    #                 因为标准答案y_是一个长度为10的一维数组，而该函数需要提供的是正确答案的数字
    #                 因此使用tf.argmax函数得到y_数组中最大值的编号，即正确答案的数字
    # '''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数，定义正则率
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失，一般只需要计算神经网络边上权重的正则化损失，而不需要计算偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失是交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
        global_step,  # 迭代轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有训练数据需要的迭代次数
        LEARNING_RATE_DECAT  # 学习率衰减速度
    )

    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    # 注意：这里损失函数包含了交叉上损失和l2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        for i in range(TRAING_STEP):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s),validation using average model is %g" % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s),test accuracy using average model is %g" % (TRAING_STEP, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    print("-------------------START-------------------")
    tf.app.run()

# 
# -------------------START-------------------
# Extracting train-images-idx3-ubyte.gz
# Extracting train-labels-idx1-ubyte.gz
# Extracting t10k-images-idx3-ubyte.gz
# Extracting t10k-labels-idx1-ubyte.gz
# After 0 training step(s),validation using average model is 0.0934
# After 1000 training step(s),validation using average model is 0.976
# After 2000 training step(s),validation using average model is 0.9808
# After 3000 training step(s),validation using average model is 0.9816
# After 4000 training step(s),validation using average model is 0.9822
# After 5000 training step(s),validation using average model is 0.9824
# After 6000 training step(s),validation using average model is 0.9838
# After 7000 training step(s),validation using average model is 0.9834
# After 8000 training step(s),validation using average model is 0.9834
# After 9000 training step(s),validation using average model is 0.9838
# After 10000 training step(s),validation using average model is 0.9836
# After 11000 training step(s),validation using average model is 0.9844
# After 12000 training step(s),validation using average model is 0.9836
# After 13000 training step(s),validation using average model is 0.9844
# After 14000 training step(s),validation using average model is 0.9838
# After 15000 training step(s),validation using average model is 0.9854
# After 16000 training step(s),validation using average model is 0.9852
# After 17000 training step(s),validation using average model is 0.9852
# After 18000 training step(s),validation using average model is 0.9856
# After 19000 training step(s),validation using average model is 0.9848
# After 20000 training step(s),validation using average model is 0.985
# After 21000 training step(s),validation using average model is 0.985
# After 22000 training step(s),validation using average model is 0.9852
# After 23000 training step(s),validation using average model is 0.9856
# After 24000 training step(s),validation using average model is 0.985
# After 25000 training step(s),validation using average model is 0.9848
# After 26000 training step(s),validation using average model is 0.9858
# After 27000 training step(s),validation using average model is 0.9854
# After 28000 training step(s),validation using average model is 0.9854
# After 29000 training step(s),validation using average model is 0.985
# After 30000 training step(s),test accuracy using average model is 0.9837