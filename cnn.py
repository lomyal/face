#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright (c) 2017, All Rights Reserved
#
################################################################################
"""
Pure CNN
Authors: Wang Shijun
Date:    2017/03/07 23:00:00
"""

import datetime
import tensorflow as tf

import utils.io


face_train = utils.io.IO()
# face_test = utils.io.IO()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 灰度图像，大小 96x96
x = tf.placeholder(tf.float32, shape=[None, 96, 96, 1])

# 双眼中心和鼻尖的二维坐标
y_ = tf.placeholder(tf.float32, shape=[None, 4])

# == First Convolutional Layer == #

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = x

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# == Second Convolutional Layer == #

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# == Densely Connected Layer == #

W_fc1 = weight_variable([24 * 24 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 24 * 24 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# == Readout Layer == #

W_fc2 = weight_variable([1024, 4])
b_fc2 = bias_variable([4])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

mse_loss = tf.losses.mean_squared_error(
        labels=y_,
        predictions=y_conv)
train_step = tf.train.AdamOptimizer(1e-4).minimize(mse_loss)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# sess = tf.Session()
sess = tf.InteractiveSession()
print('ssesion created')
sess.run(tf.global_variables_initializer())
print('variables initialized')
for i in range(5000000):
    batch = face_train.next_batch(50)
    if i % 100 == 0:
        train_loss = mse_loss.eval(feed_dict={
            x: batch[0],
            y_: batch[1],
            keep_prob: 1.0})
        date = datetime.datetime.now()
        print("[%s] step %d, training loss %g" % (date, i, train_loss))
    train_step.run(feed_dict={
        x: batch[0],
        y_: batch[1],
        keep_prob: 0.5})

# print("test accuracy %g" % accuracy.eval(feed_dict={
#     x: rdhckrs_test.images,
#     y_: rdhckrs_test.labels,
#     keep_prob: 1.0}))
