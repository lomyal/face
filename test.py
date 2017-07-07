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


def calc_test_error():
    """
    计算测试集误差
    """
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


def test_single_image(image_sn):
    """
    对测试集中的单张图片，计算关键点的坐标
    """
    face_train = utils.io.IO()
    test_data = face_train.get_one_test_sample(image_sn)

    # 读取模型，计算测试误差
    saver = tf.train.Saver()
    save_path = 'model/model_20170601_002056_loss0d2.ckpt'

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        image = test_data[0]
        label = test_data[1]
        y = sess.run(y_conv, {
                x: image,
                y_: label,
                keep_prob: 1.0
            })
        print(y)
        print(label)

if __name__ == '__main__':
    # calc_test_error()
    test_single_image(100)
