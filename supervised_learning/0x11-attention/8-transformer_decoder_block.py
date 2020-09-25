#!/usr/bin/env python3
"""contains the DecoderBlock class"""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        :param dm: dimensionality of the model
        :param h:  number of heads
        :param hidden: number of hidden units in the fully connected layer
        :param drop_rate: dropout rate
        """
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """
        :param x: tensor of shape (batch, target_seq_len, dm)
            containing the input to the decoder block
        :param encoder_output: tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder
        :param training: boolean to determine if the model is training
        :param look_ahead_mask: mask to be applied to the first
            multi head attention layer
        :param padding_mask: mask to be applied to the second
            multi head attention layer
        :return: tensor of shape (batch, target_seq_len, dm)
            containing the block’s output
        """
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1,
                                               encoder_output,
                                               encoder_output,
                                               padding_mask)

        # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        # (batch_size, target_seq_len, d_model)

        ffn_output = self.dense_hidden(out2)
        # (batch_size, target_seq_len, d_model)
        ffn_output = self.dense_output(ffn_output)

        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        # (batch_size, target_seq_len, d_model)

        return out3