#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright (c) 2017. All Rights Reserved
#
################################################################################
"""
搭建图结构

Authors: Wang Shijun
Create:  2017/07/04 17:32:00
Modify:  2017/07/04 17:32:00
"""

import tensorflow as tf


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
y_ = tf.placeholder(tf.float32, shape=[None, 6])

# == First Convolutional Layer == #

W_conv1 = weight_variable([10, 10, 1, 20])
b_conv1 = bias_variable([20])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# == Second Convolutional Layer == #

W_conv2 = weight_variable([8, 8, 20, 40])
b_conv2 = bias_variable([40])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# == Third Convolutional Layer == #

W_conv3 = weight_variable([6, 6, 40, 60])
b_conv3 = bias_variable([60])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# == Fourth Convolutional Layer == #

W_conv4 = weight_variable([4, 4, 60, 80])
b_conv4 = bias_variable([80])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

# == Densely Connected Layer == #

W_fc1 = weight_variable([6 * 6 * 80, 120])
b_fc1 = bias_variable([120])
h_pool4_flat = tf.reshape(h_pool4, [-1, 6 * 6 * 80])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# == Readout Layer == #

W_fc2 = weight_variable([120, 6])
b_fc2 = bias_variable([6])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2