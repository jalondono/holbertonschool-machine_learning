#!/usr/bin/env python3
""" Load Images """
import glob
import cv2
import numpy as np
import csv


def load_images(images_path, as_array=True):
    """
    loads images from a directory or file:
    :param images_path: is the path to a directory from which to load images
    :param as_array:is a boolean indicating whether the images should be
     loaded as one numpy.ndarray
    :return: Returns: images, filenames
    """
    imgs = []
    names = []
    paths = glob.glob(images_path + '/*')
    paths = sorted([i for i in paths])
    for path in paths:
        img = cv2.imread(path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        names.append(path[len(images_path) + 1:])
    if not as_array:
        return imgs, names
    else:
        imgs = np.array(imgs)
        return imgs, names


def load_csv(csv_path, params={}):
    """
    Load CSV
    :param csv_path: is the path to the csv to load
    :param params: are the parameters to load the csv with
    :return:  a list of lists representing the contents found in csv_path
    """
    # with open(csv_path, 'r') as f:
    #     lines = [line.strip() for line in f]
    #
    # csv_list = [line.split(',') for line in lines]
    #
    # return csv_list
    csv_list = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', **params)
        for row in csv_reader:
            csv_list.append(row)
    return csv_list
