#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-02 15:57:10
# @Author  : guanglinzhou (xdzgl812@163.com)


import tensorflow as tf

a = tf.constant([1, 2], dtype=tf.int8, name='a')
b = tf.constant([3, 4], dtype=tf.int8, name='b')
result = a + b
print(result)
print(a.graph)
print(tf.get_default_graph())

print('\n\n\n\n')

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable(
        "v", initializer=tf.zeros_initializer()(shape=[1])
    )

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable(
        "v", initializer=tf.ones_initializer()(shape=[1])
    )

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))
