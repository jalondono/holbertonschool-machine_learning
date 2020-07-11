#!/usr/bin/env python3
""" Verification Class """
import tensorflow as tf
import tensorflow.keras as K


class FaceVerification:
    def __init__(self, model_path, database, identities):
        """
        Initialize Face Verification
        :param model_path: is the path to where the face
         verification embedding model is stored
        :param database: is a numpy.ndarray of shape (d, e)
        containing all the face embeddings in the database
        :param identities:  is a list of length d containing the
         identities corresponding to the embeddings in database
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.model = tf.keras.models.load_model(model_path)
            self.database = database
            self.identities = identities

    def embedding(self, images):
        """

        :param images:is a numpy.ndarray of shape (i, n, n, 3)
         containing the aligned images
        :return:
        """