#!/usr/bin/env python3
"""contains the Transformer class"""

import numpy as np
import tensorflow.compat.v2 as tf


def positional_encoding(max_seq_len, dm):
    """
    calculates the positional encoding for a transformer
    :param max_seq_len: integer representing the maximum sequence length
    :param dm: model depth
    :return: numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encoding vectors
    """
    # position
    t = np.arange(max_seq_len)[:, np.newaxis]

    # model depth
    index = np.arange(dm)[np.newaxis, :]
    dm_float = np.float32(dm)

    # angle
    W = 1 / (np.power(10000, (2 * (index // 2) / dm_float)))

    # argument
    Wt = (W * t)

    positional_vect = np.zeros((max_seq_len, dm))

    # sin to even indices
    positional_vect[:, 0::2] = np.sin(Wt[:, 0::2])

    # cos to odd indices
    positional_vect[:, 1::2] = np.cos(Wt[:, 1::2])

    return positional_vect


def sdp_attention(Q, K, V, mask=None):
    """
    calculates the scaled dot product attention
    :param Q: tensor with its last two dimensions as (..., seq_len_q, dk)
        containing the query matrix
    :param K: tensor with its last two dimensions as (..., seq_len_v, dk)
        containing the key matrix
    :param V: tensor with its last two dimensions as (..., seq_len_v, dv)
        containing the value matrix
    :param mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
        containing the optional mask, or defaulted to None
    :return: output, weights
        outputa tensor with its last two dimensions as
            (..., seq_len_q, dv)
            containing the scaled dot product attention
        weights a tensor with its last two dimensions as
            (..., seq_len_q, seq_len_v)
            containing the attention weights
    """
    dk = tf.shape(Q)[-1]
    dk_float = tf.cast(dk, tf.float32)

    scaled = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk_float)

    if mask is not None:
        scaled += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multihead attention for machine translation"""

    def __init__(self, dm, h):
        """
        Class constructor
        :param dm: integer representing the dimensionality of the model
        :param h: integer representing the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(dm / h)

        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)

        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is
        (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        :param self:
        :param Q: tensor of shape (batch, seq_len_q, dk)
            containing the input to generate the query matrix
        :param K: tensor of shape (batch, seq_len_v, dk)
            containing the input to generate the key matrix
        :param V: tensor of shape (batch, seq_len_v, dv)
            containing the input to generate the value matrix
        :param mask: always None
        :return: output, weights
            outputa tensor with its last two dimensions as (..., seq_len_q, dm)
            containing the scaled dot product attention
            weights a tensor with its last three dimensions as
                (..., h, seq_len_q, seq_len_v) containing the attention weights
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)  # (batch_size, seq_len, d_model)
        K = self.Wk(K)  # (batch_size, seq_len, d_model)
        V = self.Wv(V)  # (batch_size, seq_len, d_model)

        q = self.split_heads(Q, batch_size)
        # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(K, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(V, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, weights = sdp_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """create an encoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        :param dm: dimensionality of the model
        :param h: number of heads
        :param hidden: number of hidden units in the fully connected layer
        :param drop_rate: dropout rate
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # (batch_size, seq_len, dff)
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        :param x: tensor of shape (batch, input_seq_len, dm)
            containing the input to the encoder block
        :param training: boolean to determine if the model is training
        :param mask: mask to be applied for multi head attention
        :return: tensor of shape (batch, input_seq_len, dm)
            containing the block’s output
        """
        attn_output, _ = self.mha(x, x, x, mask)
        # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        # (batch_size, input_seq_len, d_model)

        ffn_output = self.dense_hidden(out1)
        # (batch_size, input_seq_len, d_model)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        # (batch_size, input_seq_len, d_model)

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """DecoderBlock class for machine translation"""

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


class Encoder(tf.keras.layers.Layer):
    """Encoder class for machine translation"""

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
        seq_len = x.shape[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)

        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """Decoder class for machine translation"""

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
        seq_len = x.shape[1]

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x


class Transformer(tf.keras.Model):
    """Transformer class for machine translation"""

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
