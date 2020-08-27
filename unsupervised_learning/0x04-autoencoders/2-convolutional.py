#!/usr/bin/env python3
""""Convolutional Autoencoder """

import tensorflow.keras as Keras


def autoencoder(input_dims, filters, latent_dims):
    """
    creates an autoencoder:
    :param input_dims: is an integer containing the dimensions of
     the model input
    :param hidden_layers: is a list containing the number of filters
     for each convolutional layer in the encoder, respectively
     layer in the encoder, respectively
    :param latent_dims: is an integer containing the dimensions of
     the latent space
    representation
    * Each convolution in the encoder should use a kernel size of (3, 3)
    with same padding and relu activation, followed by max pooling
    of size (2, 2)
    * Each convolution in the decoder, except for the last two,
     should use a filter size of (3, 3) with same padding and relu activation,
     followed by upsampling of size (2, 2)
    :return: encoder, decoder, auto
    """
    input = Keras.Input(shape=input_dims)
    x = Keras.layers.Conv2D(filters=filters[0],
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same')(input)
    x = Keras.layers.MaxPool2D((2, 2), padding='same')(x)
    for filter in filters[1:]:
        x = Keras.layers.Conv2D(filters=filter,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same')(x)
        x = Keras.layers.MaxPool2D((2, 2), padding='same')(x)
    encoder = Keras.Model(inputs=input, outputs=x)
    encoder.summary()

    # decoder
    rev_filters = list(filters)[::-1]
    dec_inp = Keras.Input(shape=latent_dims)

    y = Keras.layers.Conv2D(filters=rev_filters[0],
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same')(dec_inp)
    y = Keras.layers.UpSampling2D((2, 2))(y)

    for filter in rev_filters[1:-1]:
        y = Keras.layers.Conv2D(filters=filter,
                                kernel_size=(3, 3),
                                activation='relu',
                                padding='same')(y)
        y = Keras.layers.UpSampling2D((2, 2))(y)

    y = Keras.layers.Conv2D(filters=rev_filters[-1],
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='valid')(y)
    y = Keras.layers.UpSampling2D((2, 2))(y)

    y = Keras.layers.Conv2D(filters=input_dims[2],
                            kernel_size=(3, 3),
                            activation='sigmoid',
                            padding='same')(y)
    decoder = Keras.Model(inputs=dec_inp, outputs=y)
    decoder.summary()

    # AutoEncoder
    input_auto = Keras.Input(shape=input_dims)
    encoderOut = encoder(input_auto)
    decoderOut = decoder(encoderOut)
    auto = Keras.models.Model(inputs=input_auto, outputs=decoderOut)
    auto.summary()

    auto.compile(optimizer=Keras.optimizers.Adam(),
                 loss='binary_crossentropy')
    return encoder, decoder, auto
