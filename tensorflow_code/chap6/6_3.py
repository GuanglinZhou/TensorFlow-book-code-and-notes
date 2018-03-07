#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-07 10:43:59
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou/TensorFlow-book-code-and-notes


import tensorflow as tf

# tensorflow中提供的卷积神经网络前向传播算法的函数为tf.nn.conv2d

input = tf.get_variable('input', [5, 5], initializer=tf.truncated_normal_initializer(stddev=0.1))

filter_weight = tf.get_variable('weights', shape=[5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))

biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.1))

conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')
bias = tf.nn.bias_add(conv, biases)
actived_conv = tf.nn.relu(bias)

# 最大池化层
pool = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
