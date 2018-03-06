#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-05 14:24:10
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou/TensorFlow-book-code-and-notes

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print('Training data size: ', mnist.train.num_examples)
print('Validating data size: ', mnist.validation.num_examples)
print('Testing data size: ', mnist.test.num_examples)
print('Example training data: ', mnist.train.images[0])
print('Example training data label: ', mnist.train.labels[0])

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print('X shape:', xs.shape)
print('Y shape:', ys.shape)
