#!/usr/bin/env python3
""" Evaluate """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    evaluates the output of a neural network:
    :param X: is numpy.ndarray containing the input data to evaluate
    :param Y: is a numpy.ndarray containing the one-hot labels for X
    :param save_path: is the location to load the model from
    :return: the network’s prediction, accuracy, and loss, respectively
    """
    saver = saver = tf.train.import_meta_graph("{}.meta".format(save_path))
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        Y_predict = tf.get_collection("y_pred")[0]

        loss = tf.get_collection("loss")[0]

        Y_accuracy = tf.get_collection("accuracy")[0]

        train = tf.get_collection("train_op")[0]
        predict = sess.run(Y_predict, feed_dict={x: X, y: Y})
        train_loss, train_accuarcy_value = sess.run(
            [loss, Y_accuracy],
            feed_dict={x: X, y: Y})

        return predict, train_accuarcy_value, train_loss
