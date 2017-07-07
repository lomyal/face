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
from graph import *

#timestamp = str(int(time.time()))
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

face_train = utils.io.IO()
test_data = face_train.get_test_data()

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

for i in range(10000000):
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
        
        os.system('mkdir -p model')

        if test_loss < 1.0 and train_loss < 1.0:
            save_path = saver.save(sess, 'model/model_' + timestamp + '_loss1d0' + '.ckpt')

        if test_loss < 0.5 and train_loss < 0.5:
            save_path = saver.save(sess, 'model/model_' + timestamp + '_loss0d5' + '.ckpt')

        if test_loss < 0.3 and train_loss < 0.3:
            save_path = saver.save(sess, 'model/model_' + timestamp + '_loss0d3' + '.ckpt')

        if test_loss < 0.2 and train_loss < 0.2:
            save_path = saver.save(sess, 'model/model_' + timestamp + '_loss0d2' + '.ckpt')
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
    print("[%s] test loss %g" % (date, test_loss))

