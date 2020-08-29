#!/usr/bin/env python3
""""Variational Autoencoder"""

import tensorflow.keras as keras


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates a variational autoencoder:
    :param input_dims: is an integer containing the dimensions
     of the model input
    :param hidden_layers:is a list containing the number of nodes
    for each hidden layer in the encoder, respectively
    :param latent_dims:is an integer containing the dimensions of the
     latent space representation
    :return: encoder, decoder, auto
    """
    # encoder
    enc_input = keras.Input(shape=(input_dims,))
    enc_hidden = keras.layers.Dense(units=hidden_layers[0],
                                activation='relu')(enc_input)
    for i in range(1, len(hidden_layers)):
        enc_hidden = keras.layers.Dense(hidden_layers[i],
                                    activation='relu')(enc_hidden)

    z_mean = keras.layers.Dense(latent_dims)(enc_hidden)
    z_var = keras.layers.Dense(latent_dims)(enc_hidden)
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_var])

    # decoder
    dec_input = keras.Input(shape=(latent_dims,))
    dec_hidden = keras.layers.Dense(hidden_layers[-1],
                                activation='relu')(dec_input)

    for i in range(len(hidden_layers) - 2, -1, -1):
        dec_hidden = keras.layers.Dense(hidden_layers[i],
                                    activation='relu')(dec_hidden)

    dec_hidden = keras.layers.Dense(input_dims,
                                activation='sigmoid')(dec_hidden)

    encoder = keras.models.Model(inputs=enc_input, outputs=[z, z_mean, z_var])
    decoder = keras.models.Model(inputs=dec_input, outputs=dec_hidden)

    encoderOut = encoder.layers[-1].output
    decoderOut = decoder(encoderOut)
    auto = keras.models.Model(inputs=enc_input, outputs=decoderOut)

    def loss(y_in, y_out):
        """ custom loss function """
        reconstruction_loss = keras.backend.binary_crossentropy(y_in, y_out)
        reconstruction_loss = keras.backend.sum(reconstruction_loss, axis=1)
        kl_loss = (1 + z_var - keras.backend.square(z_mean) - keras.backend.exp(z_var))
        kl_loss = -0.5 * keras.backend.sum(kl_loss, axis=1)
        return reconstruction_loss + kl_loss

    auto.compile(optimizer='Adam', loss=loss)

    return encoder, decoder, auto
