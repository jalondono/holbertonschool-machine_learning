#!/usr/bin/env python3
"""Main"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os

generator = __import__("0-generator").generator
discriminator = __import__("1-discriminator").discriminator
train_discriminator = __import__('2-train_discriminator').train_discriminator
train_generator = __import__('3-train_generator').train_generator
train_gan = __import__('5-train_GAN').train_gan
plot = __import__('plots').plot
sample_Z = __import__('4-sample_Z').sample_Z


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

    mb_size = 128
    Z_dim = 10
    epochs = 50

    # Input image, foe discrminator model.
    X = tf.placeholder(tf.float32, shape=[None, 784])

    # Input noise for generator.
    Z = tf.placeholder(tf.float32, shape=[None, 100])
    G_sample = generator(Z)
    D_real = discriminator(X)
    D_fake = discriminator(G_sample)

    D_loss, D_solver = train_discriminator(Z, X)
    G_loss, G_solver = train_generator(Z)

    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out2/'):
        os.makedirs('out2/')

    i = 0

    for it in range(1000000):

        # Save generated images every 1000 iterations.
        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

            fig = plot(samples)
            plt.savefig('out2/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        # Get next batch of images. Each batch has mb_size samples.
        X_mb, _ = mnist.train.next_batch(mb_size)

        # Run disciminator solver
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})

        # Run generator solver
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

        # Print loss
        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
