#!/usr/bin/env python3
"""contains the Encoder class"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Class constructor
        :param N: number of blocks in the encoder
        :param dm: dimensionality of the model
        :param h: number of heads
        :param hidden: number of hidden units in the fully connected layer
        :param input_vocab: size of the input vocabulary
        :param max_seq_len: maximum sequence length possible
        :param drop_rate: dropout rate
        """
        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len,
                                                       self.dm)

        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        :param x: tensor of shape (batch, input_seq_len, dm)
            containing the input to the encoder
        :param training: boolean to determine if the model is training
        :param mask: mask to be applied for multi head attention
        :return: tensor of shape (batch, input_seq_len, dm)
            containing the encoder output
        """
        seq_len = tf.shape(x)[1]

        positional_encod = tf.cast(self.positional_encoding, dtype=tf.float32)

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += positional_encod[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)