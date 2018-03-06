#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-05 13:52:10
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou/TensorFlow-book-code-and-notes

# 通过集合计算一个5层神经网络带L2正则化的损失函数的计算方法

import tensorflow as tf


def get_weight(shape, lambda_arg):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda_arg)(var))
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
# 每层网络中节点个数
layer_dimension = [2, 10, 10, 10, 1]
# 网络层数
n_layers = len(layer_dimension)

cur_layer = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    # 偏置项bias的维度和输出层维度一致
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

tf.add_to_collection('losses', mse_loss)
loss = tf.add_n(tf.get_collection('losses'))
print(loss)
