#!/usr/bin/env python3
"""Train model"""
import tensorflow as tf
import numpy as np
train_generator = __import__('3-train_generator').train_generator
train_discriminator = __import__('2-train_discriminator').train_discriminator
sample_Z = __import__('4-sample_Z').sample_Z


def train_gan(X, epochs, batch_size, Z_dim, save_path='/tmp'):
    """
    trains a GAN:
    :param X: is a np.ndarray of shape (m, 784) containing the real data input
    :param epochs: is the number of epochs that the each network should be trained for
    :param batch_size: is the batch size that should be used during training
    :param Z_dim: is the number of dimensions for the randomly generated input
    :param save_path: is the path to save the trained generator
    :return:
    """
    mb_size = 128

    # Dimension of input noise
    Z_dim = 100

    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
