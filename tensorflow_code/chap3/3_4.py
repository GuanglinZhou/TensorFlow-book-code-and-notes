#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-04 20:07:10
# @Author  : guanglinzhou (xdzgl812@163.com)

# 三层全连接神经网络解决二分类问题
import tensorflow as tf
from numpy.random import RandomState

# 选取的batch大小
batch_size = 8
# 定义网络参数矩阵
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播算法
y = tf.sigmoid(y)
cross_entropy = - tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0))
)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 随机数生成数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    # 初始化所有的变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    # 训练轮数
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: X, y_: Y}
            )
            print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
