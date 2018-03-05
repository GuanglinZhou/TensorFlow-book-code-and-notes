#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-04 20:07:10
# @Author  : guanglinzhou (xdzgl812@163.com)


import tensorflow as tf

# w1 = tf.Variable(tf.random_normal([2, 3], stddev=2))
# w2 = tf.Variable(tf.random_normal([3, 1], stddev=2))
# x = tf.constant([[0.7, 0.9]])
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)
# with tf.Session() as sess:
#     # sess.run(w1.initializer)
#     # sess.run(w2.initializer)
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     print(sess.run(y))


from numpy.random import RandomState
batch_size=8

# w1 = tf.Variable(tf.random_normal([2, 3], stddev=2))
# w2 = tf.Variable(tf.random_normal([3, 1], stddev=2))