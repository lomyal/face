#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright (c) 2017. All Rights Reserved
#
################################################################################
"""
读取模型，计算测试集误差

Authors: Wang Shijun
Create:  2017/07/03 23:58:00
Modify:  2017/07/03 23:58:00
"""

import datetime

import tensorflow as tf

import utils.io
from graph import *

face_train = utils.io.IO()
test_data = face_train.get_test_data()

# 读取模型，计算测试误差
saver = tf.train.Saver()
save_path = 'model/model_20170601_002056_loss0d2.ckpt'

with tf.Session() as sess:
    saver.restore(sess, save_path)
    mse_loss = tf.losses.mean_squared_error(
            labels=y_,
            predictions=y_conv)

    # sess.run()
    test_loss = mse_loss.eval(feed_dict={
        x: test_data[0],
        y_: test_data[1],
        keep_prob: 1.0})
    date = datetime.datetime.now()
    print("[%s] test loss %g" % (date, test_loss))
