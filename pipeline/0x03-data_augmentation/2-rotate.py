#!/usr/bin/env python3
""" rotate_image function"""
import tensorflow as tf


def rotate_image(image):
    """
    rotates an image by 90 degrees counter-clockwise
    :param image: 3D tf.Tensor containing the image to rotate
    :return:
    """
    img_90 = tf.image.rot90(image, k=1)
    return img_90
