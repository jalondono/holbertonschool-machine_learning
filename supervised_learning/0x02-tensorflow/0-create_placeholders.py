#!/usr/bin/env python3
"""Placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Placeholders
    :param nx: the number of feature columns in our data
    :param classes: the number of classes in our classifier
    :return:  placeholders named x and y, respectively
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
