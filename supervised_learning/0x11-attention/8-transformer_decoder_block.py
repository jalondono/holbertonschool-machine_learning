#!/usr/bin/env python3
""" Transformer Decoder Block """
import tensorflow as tf
MultiHeadAttention = __import__('6-multi_head_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ create an encoder block for a transformer """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ dm - the dimensionality of the model
            h - the number of heads
            hidden - the number of hidden units in the fully connected layer
            drop_rate - the dropout rate
            public instance attributes:
            mha1 - the first MultiHeadAttention layer
            mha2 - the second MultiHeadAttention layer
            dense_hidden - the hidden dense layer with hidden units
            and relu activation
            dense_output - the output dense layer with dm units
            layernorm1 - the first layer norm layer, with epsilon=1e-6
            layernorm2 - the second layer norm layer, with epsilon=1e-6
            layernorm3 - the third layer norm layer, with epsilon=1e-6
            dropout1 - the first dropout layer
            dropout2 - the second dropout layer
            dropout3 - the third dropout layer
        """
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        # self.ffn = point_wise_feed_forward_network(d_model, dff)
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
            x - a tensor of shape (batch, target_seq_len, dm)containing the
                input to the decoder block
            encoder_output - a tensor of shape (batch, input_seq_len, dm)
                containing the output of the encoder
            training - a boolean to determine if the model is training
            look_ahead_mask - the mask to be applied to the first multi head
                attention layer
            padding_mask - the mask to be applied to the second multi head
                attention layer
            Returns: a tensor of shape (batch, target_seq_len, dm)
                containing the block’s output
        """
        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        print("x", x)
        print("encoder_out", encoder_output)
        print("attn1", attn1)
        print("out1", out1)
        # (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(out1,
                                               encoder_output,
                                               encoder_output,
                                               padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        # (batch_size, target_seq_len, d_model)
        out3 = self.dense_hidden(out2)
        ffn_output = self.dense_output(out3)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out4 = self.layernorm3(ffn_output + out2)

        return out4
