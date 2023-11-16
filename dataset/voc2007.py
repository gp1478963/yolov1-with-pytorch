import torch
from torch.utils.data import dataset
import json
import pathlib
import cv2
import os


class Voc2007Dataset(dataset.Dataset):
    def __init__(self, PASCAL_VOC=None, VOCtrainval=None, VOCtest=None, transform=None, target_transform=None,
                 train=True):
        super().__init__()
        self.PASCAL_VOC = PASCAL_VOC
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        if train:
            self.Voc = os.path.join(VOCtrainval, 'VOCdevkit', 'VOC2007', 'JPEGImages')
            self.JsonFile = 'pascal_train2007.json'
        else:
            self.Voc = os.path.join(VOCtest, 'VOCtest', 'VOC2007', 'JPEGImages')
            self.JsonFile = 'VOCtest_06-Nov-2007'
        self.getset()

    def __getitem__(self, index):
        image_path = os.path.join(self.Voc, self.images[index])
        bounding_box, label = self.boxes[index], self.labels[index]
        return image_path, bounding_box, label


    def __len__(self):
        return len(self.images)

    def getset(self):
        with open(os.path.join(self.PASCAL_VOC, 'PASCAL_VOC', self.JsonFile), 'r') as filt:
            self.images = []
            images_ids = []
            self.boxes = []
            self.labels = []

            json_format = json.load(filt)
            file_list = json_format['images']
            for per_file in file_list:
                self.images.append(per_file['file_name'])
                images_ids.append(per_file['id'])

            self.boxes = [[] for i in range(len(self.images))]
            self.labels = [[] for i in range(len(self.images))]
            for annotation in json_format['annotations']:
                image_id = annotation['image_id']
                index = images_ids.index(image_id)
                self.boxes[index].append(annotation['bbox'])
                self.labels[index].append(annotation['category_id'])

