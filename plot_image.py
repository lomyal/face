#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright (c) 2017. All Rights Reserved
#
################################################################################
"""
读取 96x96 图像数据，并绘图

Authors: Wang Shijun
Create:  2017/04/22 23:58:00
Modify:  2017/04/22 23:58:00
"""

import os

import numpy as np
import matplotlib.pyplot as plt


def read_file_to_gray_images(file_name='data/training.csv'):
    """
    读取数据文件
    :param file_name:
    :return:
    """
    with open(file_name) as file:
        i = 0
        for line in file:
            if i == 0:
                i = 1
                continue
            image_raw_data = line.split(',')[-1].split(' ')
            image = np.matrix([float(x) for x in image_raw_data]).reshape(96, 96)
            os.system('mkdir -p data/images')
            file_name = 'data/images/' + str(i) + '.png'
            plt.imsave(arr=image, cmap=plt.gray(), fname=file_name)
            i += 1


def read_file_to_rgb_images(file_name='data/training.csv'):
    """
    读取数据文件
    :param file_name:
    :return:
    """
    with open(file_name) as file:
        image_id = 0
        for line in file:
            if image_id == 0:
                image_id = 1
                continue
            image_raw_data = line.split(',')[-1].split(' ')
            image_data = np.zeros(96 * 96 * 3)
            for i in range(96 * 96):
                for j in range(3):
                    image_data[i * 3 + j] = 256.0 - float(image_raw_data[i])
            image = np.array(image_data).reshape(96, 96, 3)
            os.system('mkdir -p data/images')
            file_name = 'data/images/' + str(image_id) + '.png'
            plt.imsave(arr=image, fname=file_name)
            image_id += 1


def read_file_to_rgb_images_labeled(file_name='data/training.csv'):
    """
    读取数据文件
    :param file_name:
    :return:
    """
    with open(file_name) as file:
        image_id = 0
        for line in file:
            if image_id == 0:
                image_id = 1
                continue
            data = line.split(',')
            key_points = data[:-1]
            key_points_dict = {}
            for i in range(int(len(key_points) / 2)):
                if key_points[i * 2] != '':
                    key_point_x = round(float(key_points[i * 2]))
                    key_point_y = round(float(key_points[i * 2 + 1]))
                    if key_point_x not in key_points_dict:
                        key_points_dict[key_point_x] = {}
                    key_points_dict[key_point_x][key_point_y] = 0
            image_raw_data = data[-1].split(' ')
            image_data = np.zeros(96 * 96 * 3)
            for i in range(96 * 96):
                x = i % 96
                y = int(i / 96)
                if x in key_points_dict and y in key_points_dict[x]:  # 标注
                    image_data[i * 3] = 256.0
                    image_data[i * 3 + 1] = 1.0
                    image_data[i * 3 + 2] = 256.0
                else:
                    for j in range(3):
                        image_data[i * 3 + j] = 256.0 - float(image_raw_data[i])
                        # image_data[i * 3] = 256.0 - float(image_raw_data[i])
                        # image_data[i * 3 + 1] = 0
                        # image_data[i * 3 + 2] = 0
            image = image_data.reshape(96, 96, 3)
            os.system('mkdir -p data/images_labeled')
            file_name = 'data/images_labeled/' + str(image_id) + '.png'
            plt.imsave(arr=image, fname=file_name)
            image_id += 1


if __name__ == '__main__':
    read_file_to_gray_images()
    #read_file_to_rgb_images_labeled()
