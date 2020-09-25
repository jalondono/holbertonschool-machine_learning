#!/usr/bin/env python3
"""contains the Decoder class"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Class constructor
        :param N: number of blocks in the encoder
        :param dm: dimensionality of the model
        :param h:  number of heads
        :param hidden: number of hidden units in the fully connected layer
        :param target_vocab:  size of the target vocabulary
        :param max_seq_len:  maximum sequence length possible
        :param drop_rate: dropout rate
        """
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """
        :param x: tensor of shape (batch, target_seq_len, dm)
        containing the input to the decoder
        :param encoder_output:  tensor of shape (batch, input_seq_len, dm)
        containing the output of the encoder
        :param training: boolean to determine if the model is training
        :param look_ahead_mask: mask to be applied to the first
            multi head attention layer
        :param padding_mask: mask to be applied to the second
            multi head attention layer
        :return: tensor of shape (batch, target_seq_len, dm)
            containing the decoder output
        """
        seq_len = tf.shape(x)[1]
        positional_encod = tf.cast(self.positional_encoding, dtype=tf.float32)

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += positional_encod[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x
