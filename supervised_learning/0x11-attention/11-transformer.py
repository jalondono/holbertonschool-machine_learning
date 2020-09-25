#!/usr/bin/env python3
"""contains the Transformer class"""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor
        :param N: number of blocks in the encoder
        :param dm: dimensionality of the model
        :param h: number of heads
        :param hidden: number of hidden units in the fully connected layers
        :param input_vocab: size of the input vocabulary
        :param target_vocab: size of the target vocabulary
        :param max_seq_input: maximum sequence length possible for the input
        :param max_seq_target: maximum sequence length possible for the target
        :param drop_rate: dropout rate
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden,
                               input_vocab, max_seq_input, drop_rate)

        self.decoder = Decoder(N, dm, h, hidden,
                               target_vocab, max_seq_target, drop_rate)

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        :param inputs: tensor of shape (batch, input_seq_len, dm)
            containing the inputs
        :param target: tensor of shape (batch, target_seq_len, dm)
            containing the target
        :param training: boolean to determine if the model is training
        :param encoder_mask: padding mask to be applied to the encoder
        :param look_ahead_mask: look ahead mask to be applied to the decoder
        :param decoder_mask: padding mask to be applied to the decoder
        :return: tensor of shape (batch, target_seq_len, target_vocab)
            containing the transformer output
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask)

        final_output = self.linear(dec_output)
        # (batch_size, tar_seq_len, target_vocab_size)

        return final_output
