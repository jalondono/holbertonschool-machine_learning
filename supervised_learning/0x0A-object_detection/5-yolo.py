#!/usr/bin/env python3
"""Initialize Yolo v3"""

import tensorflow.keras as K
import numpy as np
import glob
import cv2


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

            box_confidences.append(self.sigmoid(outputs[idx][:, :, :, 4:5]))
            box_class_probs.append(self.sigmoid(outputs[idx][:, :, :, 5:]))

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter Boxes
        :param boxes:  a list of numpy.ndarrays of shape
         (grid_height, grid_width, anchor_boxes, 4)
         containing the processed boundary boxes for
          each output, respectively
        :param box_confidences: a list of numpy.ndarrays of shape
         (grid_height, grid_width, anchor_boxes, 1) containing
         the processed box confidences for each output, respectively
        :param box_class_probs:a list of numpy.ndarrays of shape
         (grid_height, grid_width, anchor_boxes, classes)
          containing the processed box class probabilities
           for each output, respectively
        :return: a tuple of (filtered_boxes, box_classes, box_scores):
        """
        filtered_boxes = []
        full_conf_score = []
        threshold_score = self.class_t
        # Box Scores
        for box_conf, box_scor in zip(box_confidences, box_class_probs):
            full_conf_score.append(box_conf * box_scor)
        max_scores = [np.max(item, axis=3) for item in full_conf_score]
        max_scores = [item.reshape(-1) for item in max_scores]
        box_scores = np.concatenate(max_scores)
        idx_to_delete = np.where(box_scores < threshold_score)
        box_scores = np.delete(box_scores, idx_to_delete)

        # box_classes
        box_classes_list = [box.argmax(axis=3) for box in full_conf_score]
        box_classes_list = [box.reshape(-1) for box in box_classes_list]
        box_classes = np.concatenate(box_classes_list)
        box_classes = np.delete(box_classes, idx_to_delete)

        # filtered_boxes
        boxes_list = [box.reshape(-1, 4) for box in boxes]
        boxes = np.concatenate(boxes_list, axis=0)
        filtered_boxes = np.delete(boxes, idx_to_delete, axis=0)

        return filtered_boxes, box_classes, box_scores

    @staticmethod
    def iou(box1, box2):
        """calculates intersection over union
        (x1, y1, x2, y2)"""
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

        box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
        box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area

        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Non-max Suppression
        :param filtered_boxes:  a list of numpy.ndarrays of shape
         (grid_height, grid_width, anchor_boxes, 4)
         containing the processed boundary boxes for
          each output, respectively
        :param box_classes: a list of numpy.ndarrays of shape
         (grid_height, grid_width, anchor_boxes, 1) containing
         the processed box confidences for each output, respectively
        :param box_scores:a list of numpy.ndarrays of shape
         (grid_height, grid_width, anchor_boxes, classes)
          containing the processed box class probabilities
           for each output, respectively
        :return:
        """
        # find the unique classes
        # idx_sort = np.lexsort((-box_scores, box_classes))
        # ordered_class = np.array([box_classes[idx] for idx in idx_sort])
        # ordered_scores = np.array([box_scores[idx] for idx in idx_sort])
        # ordered_boxes = np.array([filtered_boxes[idx] for idx in idx_sort])
        #
        # unique, number_clases = np.unique(ordered_class, return_counts=True)
        # print()
        # inf = 0
        # class_of_class = []
        # for sup in number_clases:
        #     class_of_class.append(idx_sort[inf:sup + inf])
        #     inf = sup
        #
        # idx_to_delete = []
        # for class_box in class_of_class:
        #     i = 0
        #     for f_box in range(i, len(class_box)):
        #         j = i + 1
        #         for s_box in range(j, len(class_box)):
        #             box1 = filtered_boxes[class_box[f_box]]
        #             box2 = filtered_boxes[class_box[s_box]]
        #             iou = self.iou(box1, box2)
        #             if iou > self.nms_t:
        #                 idx_to_delete.append(class_box[s_box])
        #             else:
        #                 continue
        #         i += 1
        # ordered_boxes = np.delete(ordered_boxes, idx_to_delete, axis=0)
        # ordered_class = np.delete(ordered_class, idx_to_delete, axis=0)
        # ordered_scores = np.delete(ordered_scores, idx_to_delete, axis=0)
        # return ordered_boxes, ordered_class, ordered_scores

        index = np.lexsort((-box_scores, box_classes))

        box_predictions = np.array([filtered_boxes[i] for i in index])
        predicted_box_classes = np.array([box_classes[i] for i in index])
        predicted_box_scores = np.array([box_scores[i] for i in index])

        _, class_counts = np.unique(predicted_box_classes, return_counts=True)

        i = 0
        accumulated_count = 0

        for class_count in class_counts:
            while i < accumulated_count + class_count:
                j = i + 1
                while j < accumulated_count + class_count:
                    tmp = self.iou(box_predictions[i],
                                   box_predictions[j])
                    if tmp > self.nms_t:
                        box_predictions = np.delete(box_predictions, j,
                                                    axis=0)
                        predicted_box_scores = np.delete(predicted_box_scores,
                                                         j, axis=0)
                        predicted_box_classes = (np.delete
                                                 (predicted_box_classes,
                                                  j, axis=0))
                        class_count -= 1
                    else:
                        j += 1
                i += 1
            accumulated_count += class_count

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Load images
        :param folder_path: a string representing the path to the
        folder holding all the images to load
        :return: Returns a tuple of (images, image_paths):
        """
        img_paths = glob.glob(folder_path + '/*')
        img_files = [cv2.imread(path, 1) for path in img_paths]
        return img_files, img_paths

    def preprocess_images(self, images):
        """
         Preprocess images
        :param images: a list of images as numpy.ndarrays
        :return: Returns a tuple of (pimages, image_shapes)
        """
        input_w = self.model.input.shape[1].value
        input_h = self.model.input.shape[2].value

        pimages_list = []
        image_shapes_list = []

        for img in images:
            img_shape = img.shape[0], img.shape[1]
            image_shapes_list.append(img_shape)

            # Resize the images with inter-cubic interpolation
            dim = (input_w, input_h)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

            # Rescale in the range [0, 1]
            pimage = resized / 255
            pimages_list.append(pimage)

        pimages = np.array(pimages_list)
        image_shapes = np.array(image_shapes_list)

        return pimages, image_shapes
