#!/usr/bin/env python3
"""Initialize Yolo v3"""

import tensorflow.keras as K
import numpy as np


class Yolo():
    """
    Initialize Yolo v3
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        class constructor
        :param model_path: is the path to where a Darknet Keras model is stored
        :param classes_path: is the path to where the list of class names
         used for the Darknet model, listed in order of index, can be found
        :param class_t: is a float representing the box score threshold
         for the initial filtering step
        :param nms_t: is a float representing the IOU threshold for
         non-max suppression
        :param anchors: is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
         containing all of the anchor boxes:
        """
        # Load model
        self.model = K.models.load_model(model_path)
        # Load classes
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
