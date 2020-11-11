#!/usr/bin/env python3
""" contains the pca_color function"""
import tensorflow as tf
import numpy as np


def pca_color(img, alpha):
    """
    performs PCA color augmentation as described in the AlexNet paper
    :param img: 3D tf.Tensor containing the image to change
    :param alpha: tuple of length 3 containing the amount that each channel
    :return: the augmented image
    """
    img = tf.keras.preprocessing.image.img_to_array(img)
    orig_img = img.astype(float).copy()

    img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    m2 = np.zeros((3, 1))

    m2[:, 0] = alpha * eig_vals[:]

    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):   # RGB
        orig_img[..., idx] += add_vect[idx]

    orig_img = np.clip(orig_img, 0.0, 255.0)

    orig_img = orig_img.astype(np.uint8)

    return orig_img
