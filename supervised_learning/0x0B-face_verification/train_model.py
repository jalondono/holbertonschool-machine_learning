#!/usr/bin/env python3
""" TrainModel Class """
import tensorflow as tf
from triplet_loss import TripletLoss
import tensorflow.keras as K
import numpy as np


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

    @staticmethod
    def compute_tp_tn_fn_fp(y_true, y_pred):
        TP = sum((y_true == 1) & (y_pred == 1))
        TN = sum((y_true == 0) & (y_pred == 0))
        FN = sum((y_true == 1) & (y_pred == 0))
        FP = sum((y_true == 0) & (y_pred == 1))
        return TP, TN, FN, FP

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        calculates the F1 score of predictions
        :param y_pred:  a numpy.ndarray of shape (m,)
        containing the correct labels
        :return:The f1 score
        """
        TP, TN, FN, FP = TrainModel.compute_tp_tn_fn_fp(y_true, y_pred)
        if TP + FP == 0:
            return 0
        else:
            precision = TP / (TP + FP)

        if (TP + FN) == 0:
            return 0
        else:
            recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    @staticmethod
    def accuracy(y_true, y_pred):
        """
         calculates the Accuracy score of predictions
        :param y_pred: a numpy.ndarray of shape (m,)
        containing the correct labels
        :return: the accuracy
        """
        TP, TN, FN, FP = TrainModel.compute_tp_tn_fn_fp(y_true, y_pred)
        accuracy = (TP + TN) / (TP + FN + TN + FP)

        return accuracy

    def best_tau(self, images, identities, thresholds):
        """
        calculates the best tau to use for a maximal F1 score
        :param images: a numpy.ndarray of shape (m, n, n, 3)
         containing the aligned images for testing
        :param identities:  a list containing the identities of each image in images
        :param thresholds: a 1D numpy.ndarray of distance thresholds (tau) to test
        :return: (tau, f1, acc)
        """
        a = np.expand_dims(images[0], axis=0)
        pred = self.base_model.predict(images)
        print()