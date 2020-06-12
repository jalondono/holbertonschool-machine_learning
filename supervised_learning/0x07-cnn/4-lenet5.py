#!/usr/bin/env python3
"""LeNet-5 (Tensorflow)"""
import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5 architecture
     using tensorflow:
    :param x:is a tf.placeholder of shape (m, 28, 28, 1)
     containing the input images for the network
    :param y: is a tf.placeholder of shape (m, 10)
     containing the one-hot labels for the network
    :return:
    """

    kernel_init = tf.contrib.layers.variance_scaling_initializer()
    Z1 = tf.layers.Conv2D(filters=6,
                          kernel_size=(5, 5),
                          padding='same',
                          activation=tf.nn.relu,
                          kernel_initializer=kernel_init)(x)

    P1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(Z1)

    Z2 = tf.layers.Conv2D(filters=16,
                          kernel_size=(5, 5),
                          padding='valid',
                          activation=tf.nn.relu,
                          kernel_initializer=kernel_init)(P1)

    P2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(Z2)

    F = tf.layers.Flatten()(P2)

    Z3 = tf.layers.Dense(units=120,
                         activation=tf.nn.relu,
                         kernel_initializer=kernel_init)(F)
    Z4 = tf.layers.Dense(units=84,
                         activation=tf.nn.relu,
                         kernel_initializer=kernel_init)(Z3)
    Z5 = tf.layers.Dense(units=10,
                         kernel_initializer=kernel_init)(Z4)

    loss = tf.losses.softmax_cross_entropy(y, Z5)
    y_pred = tf.nn.softmax(Z5)
    train_op = tf.train.AdamOptimizer(name='Adam').minimize(loss)

    predict_ones = tf.argmax(y_pred, 1)
    label_ones = tf.argmax(y, 1)

    equals = tf.math.equal(predict_ones, label_ones)
    accuracy = tf.reduce_mean(tf.cast(equals, tf.float32), axis=0)

    return y_pred, train_op, loss, accuracy
