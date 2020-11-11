#!/usr/bin/env python3
""" function flip_image"""
import tensorflow as tf


def flip_image(image):
    """
    flips an image
    :param image: 3D tf.Tensor containing the image to flip
    :return: flipped image
    """
    flip = tf.image.flip_left_right(image)
    return flip
