#!/usr/bin/env python3
"""Contains the train_mini_batch function"""

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network model using mini-batch gradient descent:
    :param X_train: is a numpy.ndarray of shape (m, 784)
     containing the training data
    :param Y_train: is a one-hot numpy.ndarray of shape
     (m, 10) containing the training labels
    :param X_valid: is a numpy.ndarray of shape (m, 784)
     containing the validation data
    :param Y_valid: is a one-hot numpy.ndarray of shape (m, 10)
     containing the validation labels
    :param batch_size: is the number of data points in a batch
    :param epochs: is the number of times the training should
    pass through the whole dataset
    :param load_path: is the path from which to load the model
    :param save_path: is the path to where the model
     should be saved after training
    :return: the path where the model was saved
    """
    saver = tf.train.import_meta_graph("{}.meta".format(load_path))
    with tf.Session() as sess:
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        number_mini_batches = X_train.shape[0] // batch_size
        if X_train.shape[0] % batch_size != 0:
            number_mini_batches += 1

        for idx in range(epochs + 1):
            loss_train, accuracy_train = sess.run([loss, accuracy],
                                                  feed_dict={
                                                      x: X_train,
                                                      y: Y_train})

            loss_validate, accuracy_validate = sess.run([loss, accuracy],
                                                        feed_dict={
                                                            x: X_valid,
                                                            y: Y_valid})
            print("After {} epochs:".format(idx))
            print("\tTraining Cost: {}".format(loss_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(loss_validate))
            print("\tValidation Accuracy: {}".format(accuracy_validate))
            if idx < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for mini in range(number_mini_batches):
                    """ split the data in mini-batchs"""
                    start = batch_size * mini
                    end = (mini + 1) * batch_size
                    if end > X_train.shape[0]:
                        end = X_train.shape[0]

                    """Batch for training"""
                    batch_x_train = X_shuffled[start:end]
                    batch_y_train = Y_shuffled[start:end]

                    training = sess.run(train_op, feed_dict={x: batch_x_train,
                                                             y: batch_y_train})
                    if ((mini + 1) % 100 == 0 and mini != 0) or\
                            mini == number_mini_batches:
                        loss_train, accuracy_train = \
                            sess.run([loss, accuracy],
                                     feed_dict={x: batch_x_train,
                                                y: batch_y_train})
                        print("\tStep {}:".format(mini + 1))
                        print("\t\tCost: {}".format(loss_train))
                        print("\t\tAccuracy: {}".format(accuracy_train))
        return saver.save(sess, save_path)
