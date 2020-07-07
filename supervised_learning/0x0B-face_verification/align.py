#!/usr/bin/env python3
""" FaceAlign Class"""
import dlib


class FaceAlign:
    def __init__(self, shape_predictor_path):
        """
        Class constructor
        :param shape_predictor_path:  is the path to the dlib
         shape predictor model
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
