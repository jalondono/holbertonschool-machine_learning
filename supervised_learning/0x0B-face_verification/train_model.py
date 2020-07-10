#!/usr/bin/env python3
""" TrainModel Class """
import tensorflow as tf
from triplet_loss import TripletLoss


class TrainModel:
    def __init__(self, model_path, alpha):
        """
        Constructor class
        :param model_path: is the path to the base face
         verification embedding model
        :param alpha: is the alpha to use for the triplet
         loss calculation
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = tf.keras.models.load_model(model_path)
            print(self.base_model.summary())

        A = tf.keras.Input(shape=(96, 96, 3))
        P = tf.keras.Input(shape=(96, 96, 3))
        N = tf.keras.Input(shape=(96, 96, 3))

        net_01 = self.base_model(A)
        net_02 = self.base_model(P)
        net_03 = self.base_model(N)

        loss = TripletLoss(alpha)

        mix = [net_01, net_02, net_03]
        out = loss(mix)
        my_model = tf.keras.models.Model([A, P, N], out)
        my_model.compile(optimizer='adam')
        self.training_model = my_model
