#!/usr/bin/env python3
""" FaceAlign Class"""
import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2


class FaceAlign:
    def __init__(self, shape_predictor_path):
        """
        Class constructor
        :param shape_predictor_path:  is the path to the dlib
         shape preditor model
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """

        :param image:  is a numpy.ndarray of rank 3 containing an
         image from which to detect a face
        :return:  a dlib.rectangle containing the boundary box for
         the face in the image, or None on failure
        """
        try:
            faces = self.detector(image, 1)
            area = 0

            for face in faces:
                if face.area() > area:
                    area = face.area()
                    rect = face

            if area == 0:
                rect = (dlib.rectangle(left=0, top=0, right=image.shape[1],
                                       bottom=image.shape[0]))

            return rect
        except RuntimeError:
            return None

    def find_landmarks(self, image, detection):
        """
        Find Landmarks
        :param image: is a numpy.ndarray of an image from
         which to find facial landmarks
        :param detection:  is a dlib.rectangle containing the
         boundary box of the face in the image
        :return: a numpy.ndarray of shape (p, 2)containing the
         landmark points, or None on failure
        """
        try:
            shape = self.shape_predictor(image, detection)
            a = np.ones((68, 2))
            for idx in range(68):
                a[idx] = (shape.part(idx).x, shape.part(idx).y)
            return a
        except RuntimeError:
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """
        Align Faces
        :param image: is a numpy.ndarray of rank 3 containing
         the image to be aligned
        :param landmark_indices:  is a numpy.ndarray of shape (3,)
         containing the indices of the three landmark points that
         should be used for the affine transformation
        :param anchor_points: is a numpy.ndarray of shape (3, 2)
         containing the destination points for the affine
         transformation, scaled to the range [0, 1]
        :param size: is the desired size of the aligned image
        :return: a numpy.ndarray of shape (size, size, 3) containing
         the aligned image, or None if no face is detected
        """
        rect = self.detect(image)
        coords = self.find_landmarks(image, rect)
        input_points = coords[landmark_indices]
        input_points = input_points.astype('float32')
        output_points = anchor_points * size
        warp_mat = cv2.getAffineTransform(input_points, output_points)
        warp_dst = cv2.warpAffine(image, warp_mat, (size, size))
        return warp_dst
