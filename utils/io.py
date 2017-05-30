#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright (c) 2017 All Rights Reserved
#
################################################################################
"""
IO functions

Authors: Wang Shijun
Date:  2017/03/01 22:39:00
"""

import math

import numpy as np


class IO(object):
    """
    数据格式：
    left_eye_center_x,
    left_eye_center_y,
    right_eye_center_x,
    right_eye_center_y,
    left_eye_inner_corner_x,
    left_eye_inner_corner_y,
    left_eye_outer_corner_x,
    left_eye_outer_corner_y,
    right_eye_inner_corner_x,
    right_eye_inner_corner_y,
    right_eye_outer_corner_x,
    right_eye_outer_corner_y,
    left_eyebrow_inner_end_x,
    left_eyebrow_inner_end_y,
    left_eyebrow_outer_end_x,
    left_eyebrow_outer_end_y,
    right_eyebrow_inner_end_x,
    right_eyebrow_inner_end_y,
    right_eyebrow_outer_end_x,
    right_eyebrow_outer_end_y,
    nose_tip_x,
    nose_tip_y,
    mouth_left_corner_x,
    mouth_left_corner_y,
    mouth_right_corner_x,
    mouth_right_corner_y,
    mouth_center_top_lip_x,
    mouth_center_top_lip_y,
    mouth_center_bottom_lip_x,
    mouth_center_bottom_lip_y,
    Image
    """

    def __init__(self, file_name='training.csv'):
        self.training_attribute_count = 0
        self.dirty_data_count = 0
        self.dirty_data_velo_count = 0
        self.dirty_data_curv_count = 0
        self.training_file_name = './data/training.csv'
        self.training_images = []
        self.training_attribute = []
        self.test_images = []
        self.test_attribute = []
        self._read_data()
        self.data_size = len(self.training_attribute)
        
        print('Training data: %d' % len(self.training_images))
        print('Test data: %d' % len(self.test_images))

        # self._pre_process_data()

    def _read_data(self):
        """

        """
        with open(self.training_file_name) as file:
            i = 0
            for line in file:
                if i == 0:
                    i = 1
                    continue
                line_data_list = line.split(',')
                attribute = line_data_list[:-1]
                # position_list = [0, 1, 2, 3, 20, 21, 22, 23, 24, 25]  # 双眼中心和鼻尖的坐标在数据文件中的位置
                position_list = [0, 1, 2, 3, 20, 21]  # 双眼中心和鼻尖的坐标在数据文件中的位置
                is_dirty = False
                for pos in position_list:
                    if (self._is_dirty_data(attribute[pos])):
                        is_dirty = True
                        break
                if is_dirty:
                    i += 1
                    continue
                three_point_attribute = np.array([
                    float(attribute[0]),  # left_eye_center_x
                    float(attribute[1]),  # left_eye_center_y
                    float(attribute[2]),  # right_eye_center_x
                    float(attribute[3]),  # right_eye_center_y
                    float(attribute[20]),  # nose_tip_x
                    float(attribute[21]),  # nose_tip_y
                    # float(attribute[22]),  # mouth_left_corner_x
                    # float(attribute[23]),  # mouth_left_corner_y
                    # float(attribute[24]),  # mouth_right_corner_x
                    # float(attribute[25]),  # mouth_right_corner_y
                ]).reshape(1, 6)
                image_raw_data = line_data_list[-1].split(' ')
                image_float = [float(x) for x in image_raw_data]
                image = np.array(image_float).reshape(96, 96, 1)
                if i <= 5000:
                    self.training_images.append(image)
                    self.training_attribute.append(three_point_attribute)
                else:
                    self.test_images.append(image)
                    self.test_attribute.append(three_point_attribute)
                i += 1
            print('i = %d' % i)

    def get_test_data(self):
        """
        获取测试集数据
        """
        batch_size = len(self.test_images)
        images = np.ndarray(shape=(batch_size, 96, 96, 1,))
        labels = np.ndarray(shape=(batch_size, 6,))
        data_count = 0
        while data_count < batch_size:
            images[data_count] = self.training_images[self.training_attribute_count]
            labels[data_count] = self.training_attribute[self.training_attribute_count]
            data_count += 1
        return [images, labels]

    def next_batch(self, batch_size):
        """
        返回下一块数据
        :param batch_size:
        :return:
        """
        images = np.ndarray(shape=(batch_size, 96, 96, 1,))
        labels = np.ndarray(shape=(batch_size, 6,))
        data_count = 0
        while data_count < batch_size:
            images[data_count] = self.training_images[self.training_attribute_count]
            labels[data_count] = self.training_attribute[self.training_attribute_count]
            data_count += 1
            self._training_attribute_count_acc()
        return [images, labels]

    def _training_attribute_count_acc(self):
        """
        计数自增，无限循环使用
        :return:
        """
        if self.training_attribute_count + 1 >= self.data_size:
            # print('=-> Data used up. Start over again.')
            # print('=-> Total data: %d' % self.training_attribute_count)
            # print('=-> Dirty data: %d (%.2f%%)' % (
            #    self.dirty_data_count,
            #    self.dirty_data_count / self.training_attribute_count * 100))
            # print('=-> Dirty data (velocity): %d' % self.dirty_data_velo_count)
            # print('=-> Dirty data (curvature): %d' % self.dirty_data_curv_count)
            self.training_attribute_count = 0
            self.dirty_data_count = 0
            self.dirty_data_velo_count = 0
            self.dirty_data_curv_count = 0
        else:
            self.training_attribute_count += 1

    def _is_dirty_data(self, data):
        """
        检测是否是脏数据
        :param data: 字符串，应为 0～96 之间的实数
        :return:
        """
        try:
            value = float(data)
        except ValueError:
            self.dirty_data_count += 1
            return True
        if value < 0 or value > 96:
            self.dirty_data_count += 1
            return True
        return False

        # velocity = math.sqrt(data[1]**2 + data[2]**2)
        # curvature = abs(data[4])

        # # 车速<=5km/h，或转弯曲率>=0.5为脏数据
        # if velocity <= 5.0:
        #     self.dirty_data_count += 1
        #     self.dirty_data_velo_count += 1
        #     self._training_attribute_count_acc()
        #     return True
        # elif curvature >= 0.5:
        #     self.dirty_data_count += 1
        #     self.dirty_data_curv_count += 1
        #     self._training_attribute_count_acc()
        #     return True
        # else:
        #     return False

    # def _pre_process_data(self):
    #     """
    #     将原始数据组装成
    #     :return:
    #     """
    #     pass
