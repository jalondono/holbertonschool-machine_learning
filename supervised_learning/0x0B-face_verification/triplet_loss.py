#!/usr/bin/env python3
""" Triplet Loss  Class"""
import tensorflow
import tensorflow.keras as K
import numpy as np


class TripletLoss(tensorflow.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        """
        Initialize Triplet Loss
        :param alpha: is the alpha value used to calculate
         the triplet loss
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def triplet_loss(self, inputs):
        """
        Calculate Triplet Loss
        :param inputs: is a list containing the anchor, positive
         and negative output tensors from the last layer of the model,
          respectively
        :return: Returns: a tensor containing the triplet loss values
        """
        A, P, N = inputs
        subtracted_1 = K.layers.Subtract()([A, P])
        subtracted_2 = K.layers.Subtract()([A, N])

        square_sub_1 = K.backend.square(subtracted_1)
        square_sub_2 = K.backend.square(subtracted_2)

        out1 = K.backend.sum(square_sub_1, axis=1)
        out2 = K.backend.sum(square_sub_2, axis=1)

        out3 = K.layers.Subtract()([out1, out2])
        loss = K.backend.maximum(out3 + self.alpha, 0)
        return loss
