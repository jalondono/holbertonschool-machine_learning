#!/usr/bin/env python3
"""Main"""
import tensorflow as tf
import numpy as np
generator = __import__("0-generator").generator
discriminator = __import__("1-discriminator").discriminator
train_discriminator = __import__('2-train_discriminator').train_discriminator
train_generator = __import__('3-train_generator').train_generator


if __name__ == '__main__':
    # lib = np.load('../data/MNIST.npz')
    # X_train_3D = lib['X_train']
    # Y_train = lib['Y_train']
    # X_valid_3D = lib['X_valid']
    # Y_valid = lib['Y_valid']
    # X_test_3D = lib['X_test']
    # Y_test = lib['Y_test']
    # X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    # X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
    # X_test = X_test_3D.reshape((X_test_3D.shape[0], -1)).T
    # print(generator())
    # Declare inputs and parameters to the model

    # Input image, foe discrminator model.
    X = tf.placeholder(tf.float32, shape=[None, 784])

    # Input noise for generator.
    Z = tf.placeholder(tf.float32, shape=[None, 100])
    G_sample = generator(Z)
    D_real = discriminator(X)
    D_fake = discriminator(G_sample)

    D_loss, D_solver = train_discriminator(Z, X)
    G_loss, G_solver = train_generator(Z)

    mb_size = 128

    # Dimention of input noise
    Z_dim = 100
    input_data = tf.examples.tutorials.mnist.input_data
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())