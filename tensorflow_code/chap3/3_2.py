#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-04 20:07:10
# @Author  : guanglinzhou (xdzgl812@163.com)


import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([3.0, 4.0], name="b")
result = tf.add(a, b, name="add")
print(result)
print(result.get_shape())

with tf.Session() as sess:
    print(sess.run(result))
with tf.Session() as sess1:
    print(sess1.run(result))
