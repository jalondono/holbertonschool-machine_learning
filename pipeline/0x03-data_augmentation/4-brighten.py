#!/usr/bin/env python3
""" change_brightness function"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    hanges the brightness of an image
    :param image: 3D tf.Tensor containing the image to change
    :param max_delta: maximum amount the image should be brightened
    :return:
    """
    img = tf.image.adjust_brightness(image, max_delta)
    return img
