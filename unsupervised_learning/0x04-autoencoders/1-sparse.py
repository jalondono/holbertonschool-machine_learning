#!/usr/bin/env python3
""""Sparse Autoencoder"""

import tensorflow.keras as Keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    creates a sparse autoencoder::
    :param lambtha:  is the regularization parameter used for L1
    regularization on the encoded output
    :param input_dims: is an integer containing the dimensions of
     the model input
    :param hidden_layers: is a list containing the number of nodes
     for each hidden
     layer in the encoder, respectively
    :param latent_dims: is an integer containing the dimensions of
     the latent space
    representation
    :return: encoder, decoder, auto
    """
    # encoder
    regularizer = Keras.regularizers.l1(lambtha)
    inp = Keras.Input(shape=(input_dims,))
    encoder = Keras.layers.Dense(units=hidden_layers[0],
                                 activation='relu',
                                 activity_regularizer=regularizer)(inp)
    for layer in range(1, len(hidden_layers)):
        encoder = Keras.layers.Dense(units=hidden_layers[layer],
                                     activation='relu',
                                     activity_regularizer=regularizer)(encoder)
    last = Keras.layers.Dense(units=latent_dims,
                              activation='relu',
                              activity_regularizer=regularizer)(encoder)
    encoder = Keras.Model(inputs=inp, outputs=last)

    # decoder
    rev_hid_layers = hidden_layers[::-1]
    dec_input = Keras.Input(shape=(latent_dims,))
    decoder = Keras.layers.Dense(units=rev_hid_layers[0],
                                 activation='relu')(dec_input)
    for layer in reversed(rev_hid_layers[1:]):
        decoder = Keras.layers.Dense(units=layer,
                                     activation='relu')(decoder)

    decoder = Keras.layers.Dense(units=input_dims,
                                 activation='sigmoid')(decoder)
    decoder = Keras.Model(inputs=dec_input, outputs=decoder)

    # auto encoder
    autoencod_bottleneck = encoder.layers[-1].output
    autoencod_output = decoder(autoencod_bottleneck)

    auto = Keras.Model(inputs=inp, outputs=autoencod_output)

    auto.compile(optimizer=Keras.optimizers.Adam(),
                 loss='binary_crossentropy')
    return encoder, decoder, auto
