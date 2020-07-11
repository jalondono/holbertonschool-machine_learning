#!/usr/bin/env python3
""" Verification Class """
import tensorflow as tf
import tensorflow.keras as K
import numpy as np


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
        calculates the face embedding of images
        :param images:is a numpy.ndarray of shape (i, n, n, 3)
         containing the aligned images
        :return:  a numpy.ndarray of shape (i, e) containing the
         embeddings where e is the dimensionality of the embeddings
        """
        predict = self.model.predict(images)
        return predict

    def verify(self, image, tau=0.5):
        """
        Verify
        :param image: is a numpy.ndarray of shape (n, n, 3) containing
         the aligned image of the face to be verify
        :param tau: is the maximum euclidean distance used for verification
        :return: (identity, distance), or (None, None) on failure
        """
        distances_aux = []
        image = image[np.newaxis, ...]
        prediction = self.model.predict(image)

        for idx, img in enumerate(self.database):
            distances_aux.append(np.sum(np.square(prediction - img)))

        distances = np.array(distances_aux)
        idx = np.argmin(distances)

        if distances[idx] < tau:
            return self.identities[idx], distances[idx]
        else:
            return None, None

