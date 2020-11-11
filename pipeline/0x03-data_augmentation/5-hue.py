#!/usr/bin/env python3
""" change_hue function"""
import tensorflow as tf


def change_hue(image, delta):
    """
    changes the hue of an image
    :param image: 3D tf.Tensor containing the image to change
    :param delta: the amount the hue should change
    :return:
    """
    img = tf.image.adjust_hue(image, delta)
    return img
