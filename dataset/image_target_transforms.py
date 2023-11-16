import cv2
import torch
import numpy as np

class ImageNormalize(object):
    def __init__(self):
        pass

    def __call__(self, image, bounding_box, label):
        return image / 255.0, bounding_box, label


class ImageStandardize(object):
    def __init__(self):
        pass

    def __call__(self, image, bounding_box, label):
        return (image - np.mean(image)) / np.std(image), bounding_box, label



class ImageResize(object):
    def __init__(self, width, height):
        self.width, self.height = width, height

    def __call__(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
