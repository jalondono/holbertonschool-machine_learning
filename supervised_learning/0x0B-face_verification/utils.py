#!/usr/bin/env python3
""" Load Images """
import glob
import cv2
import numpy as np
import csv
import os


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
        imgs = np.stack(imgs, axis=0)
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


def save_images(path, images, filenames):
    """
    Save Files
    :param path: is the path to the directory in which the
    images should be saved
    :param images: is a list/numpy.ndarray of images to save
    :param filenames:  is a list of filenames of the images to save
    :return: True on success and False on failure
    """
    try:
        # change current directory
        os.chdir(path)
        # Write the images
        for name, img in zip(filenames, images):
            cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # change to the preview directory
        os.chdir('../')
        return True
    except Exception:
        return False


def generate_triplets(images, filenames, triplet_names):
    """
    Generate Triplets
    :param images: is a numpy.ndarray of shape (i, n, n, 3)
     containing the aligned images in the dataset
    :param filenames: is a list of length i containing the
     corresponding filenames for images
    :param triplet_names: is a list of length m of lists where
    each sublist contains the filenames of an anchor, positive,
     and negative image, respectively
    :return: Returns: a list [A, P, N]
    """
    A, P, N = [], [], []
    try:
        for trip in triplet_names:
            A_idx = filenames.index(trip[0] + '.jpg')
            P_idx = filenames.index(trip[1] + '.jpg')
            N_idx = filenames.index(trip[2] + '.jpg')

            A.append(images[A_idx])
            P.append(images[P_idx])
            N.append(images[N_idx])
    except Exception:
        pass
    return [np.array(A), np.array(P), np.array(N)]
