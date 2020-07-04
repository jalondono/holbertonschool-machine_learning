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

    def sigmoid(self, x):
        """ sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process Outputs
        :param outputs: is a list of numpy.ndarrays containing the
         predictions from the Darknet model for a single image:
        :param image_size: is a numpy.ndarray containing the image’s
         original size [image_height, image_width]
        :return: a tuple of (boxes, box_confidences, box_class_probs):
        """
        boxes = [pred[:, :, :, 0:4] for pred in outputs]
        box_confidences = []
        box_class_probs = []

        for idx, out in enumerate(boxes):

            box_confidences.append(boxes[idx][:, :, :, 4:5])
            box_class_probs.append(boxes[idx][:, :, :, 5:])

            for grid_h in range(out.shape[0]):
                for grid_w in range(out.shape[1]):

                    # center of bounding box tx
                    tx = out[grid_h, grid_w, :, 0]
                    # center of bounding box ty
                    ty = out[grid_h, grid_w, :, 1]

                    bx = ((self.sigmoid(tx) + grid_w) / out.shape[1])
                    by = ((self.sigmoid(ty) + grid_h) / out.shape[0])

                    # anchors
                    anchor = self.anchors[idx].astype(float)

                    # pw
                    pw = anchor[:, 0]
                    # ph
                    ph = anchor[:, 1]

                    anchor[:, 0] *= \
                        np.exp(out[grid_h, grid_w, :,
                               2]) / self.model.input.shape[1].value  # bw
                    anchor[:, 1] *= \
                        np.exp(out[grid_h, grid_w, :,
                               3]) / self.model.input.shape[2].value  # bh

                    out[grid_h, grid_w, :, 0] = \
                        (bx - (pw / 2)) * \
                        image_size[1]  # x1
                    out[grid_h, grid_w, :, 1] = \
                        (by - (ph / 2)) * \
                        image_size[0]  # y1
                    out[grid_h, grid_w, :, 2] = \
                        (bx + (pw / 2)) * \
                        image_size[1]  # x2
                    out[grid_h, grid_w, :, 3] = \
                        (by + (ph / 2)) * \
                        image_size[0]  # y2

        return boxes, box_confidences, box_class_probs
