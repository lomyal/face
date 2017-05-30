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

import os
import time
import datetime
import tensorflow as tf

import utils.io


#timestamp = str(int(time.time()))
timestamp = datetime.datetime.now().strftime('%Y%m%d%_H%M%S')

face_train = utils.io.IO()
test_data = face_train.get_test_data()


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

mse_loss = tf.losses.mean_squared_error(
        labels=y_,
        predictions=y_conv)
tf.summary.scalar('mse_loss', mse_loss)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(mse_loss)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# sess = tf.Session()
sess = tf.InteractiveSession()

# tf board
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('summary/train_' + timestamp, sess.graph)
test_writer = tf.summary.FileWriter('summary/test_' + timestamp)

sess.run(tf.global_variables_initializer())

save_path = ''

for i in range(5000000):
    batch = face_train.next_batch(200)

    if i % 100 == 0:
        train_loss = mse_loss.eval(feed_dict={
            x: batch[0],
            y_: batch[1],
            keep_prob: 1.0})
        date = datetime.datetime.now()
        print("[%s] step %d, training loss %g" % (date, i, train_loss))

        test_loss = mse_loss.eval(feed_dict={
            x: test_data[0],
            y_: test_data[1],
            keep_prob: 1.0})
        date = datetime.datetime.now()
        print("[%s] step %d, test loss %g" % (date, i, test_loss))
        # summary, acc = sess.run([merged, mse_loss], feed_dict={
        #     x: test_data[0],
        #     y_: test_data[1],
        #     keep_prob: 1.0})
        # test_writer.add_summary(summary, i)
        
        if test_loss < 1.5 and train_loss < 1.5:
        # if test_loss < 10000 and train_loss < 10000:
            os.system('mkdir -p model')
            save_path = saver.save(sess, 'model/model_' + timestamp + '.ckpt')
            print('model saved at ' + save_path)
            break
        
    summary, _ = sess.run([merged, optimizer], feed_dict={
        x: batch[0],
        y_: batch[1],
        keep_prob: 0.7})
    train_writer.add_summary(summary, i)

sess.close()

# Final test loss
with tf.Session() as sess:
    saver.restore(sess, save_path)
    test_loss = mse_loss.eval(feed_dict={
        x: test_data[0],
        y_: test_data[1],
        keep_prob: 1.0})
    date = datetime.datetime.now()
    print("[%s] step %d, test loss %g" % (date, i, test_loss))

