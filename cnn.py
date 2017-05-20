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


rdhckrs_train = utils.io.IO('613.h5')
# rdhckrs_test = utils.io.IO('554.h5')


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


# 三色（RGB）图像，大小 320x320
x = tf.placeholder(tf.float32, shape=[None, 320, 320, 3])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 0.125 秒后的曲率
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# == First Convolutional Layer == #

# The convolution will compute 32 features for each 5x5 patch.
# The first two dimensions are the patch size, the next is the number
# of input channels, and the last is the number of output channels.
# 三色
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

# To apply the layer, we first reshape x to a 4d tensor, with the
# second and third dimensions corresponding to image width and height,
# and the final dimension corresponding to the number of color channels.

# x_image = tf.reshape(x, [-1, 320, 320, 3])
x_image = x

# We then convolve x_image with the weight tensor, add the bias, apply
# the ReLU function, and finally max pool. The max_pool_2x2 method will
# reduce the image size to 14x14.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# == Second Convolutional Layer == #

# In order to build a deep network, we stack several layers of this type.
# The second layer will have 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# == Densely Connected Layer == #

# Now that the image size has been reduced to 7x7, we add a
# fully-connected layer with 1024 neurons to allow processing on the
# entire image. We reshape the tensor from the pooling layer into a batch
# of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([80 * 80 * 64, 512])
b_fc1 = bias_variable([512])
h_pool2_flat = tf.reshape(h_pool2, [-1, 80 * 80 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# To reduce overfitting, we will apply DROPOUT before the readout layer.
# We create a placeholder for the probability that a neuron's output is
# kept during dropout. This allows us to turn dropout on during training,
# and turn it off during testing. TensorFlow's tf.nn.dropout op
# automatically handles scaling neuron outputs in addition to masking
# them, so dropout just works without any additional scaling.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# == Readout Layer == #

# Finally, we add a layer, just like for the one layer softmax regression
# above.
W_fc2 = weight_variable([512, 1])
b_fc2 = bias_variable([1])
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
for i in range(100000):
    batch = rdhckrs_train.next_batch(50)
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