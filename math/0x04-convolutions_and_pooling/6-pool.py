#!/usr/bin/env python3
"""  Pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
     performs pooling on images:
     :param mode: indicates the type of pooling
     :param kernel_shape: is a tuple of (kh, kw)
      containing the kernel shape for the pooling
     :param images: is a numpy.ndarray with shape (m, h, w)
      containing multiple grayscale images
     :param stride: is a tuple of (sh, sw)
     :return: a numpy.ndarray containing the convolved images
     """
    # image sizes
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    # Kernel Size
    kh = kernel_shape[0]
    kw = kernel_shape[1]

    # Strides
    sh = stride[0]
    sw = stride[1]

    # output size
    out_h = int(((h - kh) / sh) + 1)
    out_w = int(((w - kw) / sw) + 1)

    out_img = np.zeros((m, out_h, out_w, c))
    for x in range(out_w):
        for y in range(out_h):
            if mode == 'max':
                img = images[:, y * sh: y * sh + kh, x * sw: x * sw + kw]
                pixel = np.max(img, axis=(1, 2))
                out_img[:, y, x, :] = pixel
            if mode == 'avg':
                img = images[:, y * sh: y * sh + kh, x * sw: x * sw + kw]
                pixel = np.mean(img, axis=(1, 2))
                out_img[:, y, x, :] = pixel
    return out_img
