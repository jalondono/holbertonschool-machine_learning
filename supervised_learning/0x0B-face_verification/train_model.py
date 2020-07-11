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

    def train(self, triplets, epochs=5, batch_size=32,
              validation_split=0.3, verbose=True):
        """
        that trains self.training_model:
        :param self:
        :param triplets: is a list of numpy.ndarrayscontaining the
         inputs to self.training_model
        :param epochs: is the number of epochs to train for
        :param batch_size:  is the batch size for training
        :param validation_split:  is the validation split for training
        :param verbose: is a boolean that sets the verbosity mode
        :return: the History output from the training
        """
        return self.training_model.fit(triplets,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       validation_split=validation_split,
                                       verbose=verbose)

    def save(self, save_path):
        """
        saves the base embedding model:
        :param save_path: is the path to save the model
        :return: the saved model
        """
        tf.keras.models.save_model(self.base_model, save_path)
        return self.base_model
