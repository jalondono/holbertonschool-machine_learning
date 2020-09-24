#!/usr/bin/env python3
""" Rnn Encoder """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ RNN Encoder class"""
    def __init__(self, vocab, embedding, units, batch):
        """
        Constructor class
        :param vocab:
        :param embedding:
        :param units:
        :param batch:
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros
        :return: a tensor of shape (batch, units)containing the initialized hidden states
        """
        initializer = tf.keras.initializers.Zeros()
        values = initializer(shape=(self.batch, self.units))
        return values

    def call(self, x, initial):
        """

        :param x: is a tensor of shape (batch, input_seq_len) containing
        the input to the encoder layer as word indices within the vocabulary
        :param initial: initial is a tensor of shape (batch, units) containing
        the initial hidden state
        :return: outputs, hidden
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
