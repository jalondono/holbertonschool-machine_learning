#!/usr/bin/env python3
""" FaceAlign Class"""
import dlib
import matplotlib.pyplot as plt


class FaceAlign:
    def __init__(self, shape_predictor_path):
        """
        Class constructor
        :param shape_predictor_path:  is the path to the dlib
         shape predictor model
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
        current_area = 0
        idx_max_area = 0
        height, width, c = image.shape
        rectangle = dlib.rectangle(left=0,
                                   top=0,
                                   right=width,
                                   bottom=height)
        try:
            dets = self.detector(image, 1)
            max_area = 0
            if len(dets) == 0:
                return rectangle
            if len(dets) > 1:
                for idx, det in enumerate(dets):
                    x1, y1, x2, y2 = det
                    current_area = (x2 - x1) * (y2 - y1)
                    if current_area > max_area:
                        max_area = current_area
                        idx_max_area = idx
            return dets[idx_max_area]
        except Exception:
            return None
