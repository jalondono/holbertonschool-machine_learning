#!/usr/bin/env python3
""" Triplet Loss  Class"""
import tensorflow
import tensorflow.keras as K


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
