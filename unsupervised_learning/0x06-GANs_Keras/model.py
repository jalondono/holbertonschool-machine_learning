#!/usr/bin/env python3
"""GAN model"""
import tensorflow.keras as K
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class GanModel:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.z_dim = 100

    # Generator model
    def build_generator(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.z_dim))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(28 * 28 * 1, activation='tanh'))
        model.add(Reshape(self.img_shape))
        return model

    # discriminator model
    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(1, activation='sigmoid'))
        return model

    # Build the model
    def build_gan(self, generator, discriminator):
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=Adam(),
                              metrics=['accuracy'])
        discriminator.trainable = False
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer=Adam())
        return model

    def sample_images(self, generator, image_grid_rows=4, image_grid_columns=4):
        z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, self.z_dim))
        gen_imgs = generator.predict(z)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(image_grid_rows,
                                image_grid_columns,
                                figsize=(4, 4),
                                sharey=True,
                                sharex=True)
        cnt = 0
        for i in range(image_grid_rows):
            for j in range(image_grid_columns):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()

    def train(self, iterations, batch_size, sample_interval):
        # instances
        generator = self.build_generator()
        discriminator = self.build_discriminator()
        gan = self.build_gan(generator, discriminator)

        # initialization
        losses = []
        accuracies = []
        iteration_checkpoints = []

        (X_train, _), (_, _) = mnist.load_data()
        X_train = X_train / 127.5 - 1.0
        X_train = np.expand_dims(X_train, axis=3)
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for iteration in range(iterations):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = generator.predict(z)
            d_loss_real = discriminator.train_on_batch(imgs, real)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)
            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = generator.predict(z)
            g_loss = gan.train_on_batch(z, real)
            if (iteration + 1) % sample_interval == 0:
                losses.append((d_loss, g_loss))
                accuracies.append(100.0 * accuracy)
                iteration_checkpoints.append(iteration + 1)
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                      (iteration + 1, d_loss, 100.0 * accuracy, g_loss))
                self.sample_images(generator)
        return generator, discriminator


if __name__ == '__main__':
    gan = GanModel()
    iter = 50000
    b_size = 128
    s_interval = 10000
    gen, dis = gan.train(iter, b_size, s_interval)
    gan.sample_images(gen)
