import cv2
import torch
import numpy as np


class ImageNormalize(object):
    def __init__(self):
        pass

    def __call__(self, image, bounding_box, label):
        return image / 255.0, bounding_box, label


class ImageNormalizeV2(object):
    def __call__(self, image, label):
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
        return image / 255.0, label


class ImageStandardize(object):
    def __call__(self, image, bounding_box=None, label=None):
        return (image - np.mean(image)) / np.std(image), bounding_box


class ImageStandardizeV2(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        return (image - np.mean(image)) / np.std(image), label


class ImageResize(object):
    def __init__(self, width, height):
        self.width, self.height = width, height

    def __call__(self, image, label=None):
        cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if label is not None:
            for bndbox in label:
                bndbox[0][:4:2] = (bndbox[0][:4:2]) / self.width
                bndbox[0][1:4:2] = (bndbox[0][1:4:2]) / self.height
        return image, label
