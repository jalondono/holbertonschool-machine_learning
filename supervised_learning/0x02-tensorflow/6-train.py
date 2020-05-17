#!/usr/bin/env python3
""" Train_Op """
import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    builds, trains, and saves a neural network classifier
    :param X_train: the training input data
    :param Y_train: containing the training labels
    :param X_valid: containing the validation input data
    :param Y_valid: containing the validation labels
    :param layer_sizes: Is a list containing the number of nodes in
    each layer of the network
    :param activations: is a list containing the activation functions
     for each layer of the network
    :param alpha: is the learning rate
    :param iterations: is the number of iterations to train over
    :param save_path: designates where to save the model
    :return:
    """
    # For training after 0 iterations
    X, Y = create_placeholders(X_train.shape[1], Y_valid.shape[1])
    tf.add_to_collection('x', X)
    tf.add_to_collection('y', Y)

    Y_predict = forward_prop(X, layer_sizes, activations)
    tf.add_to_collection('y_pred', Y_predict)

    loss = calculate_loss(Y, Y_predict)
    tf.add_to_collection('loss', loss)

    Y_accuracy = calculate_accuracy(Y, Y_predict)
    tf.add_to_collection('accuracy', Y_accuracy)

    train = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for idx in range(iterations + 1):
            """for training"""
            train_loss, train_accuarcy_value = sess.run(
                [loss, Y_accuracy],
                feed_dict={X: X_train, Y: Y_train})
            """ For validation """
            val_train_loss, val_accuarcy_value = sess.run(
                [loss, Y_accuracy],
                feed_dict={X: X_valid, Y: Y_valid})

            if idx == iterations or idx % 100 == 0:
                print("After {} iterations:".format(idx))
                print('\tTraining Cost: {}'.format(train_loss))
                print("\tTraining Accuracy: {}".format(train_accuarcy_value))
                print('\tValidation Cost: {}'.format(val_train_loss))
                print('\tValidation Accuracy: {}'.format(val_accuarcy_value))
            if idx < iterations:
                training = sess.run(train, feed_dict={X: X_train, Y: Y_train})
        return saver.save(sess, save_path)
