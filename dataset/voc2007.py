import torch
from torch.utils.data import dataset
import json
import pathlib
import cv2
import os

CELL_SIZE = 1 / 7
CELL_COUNT = 7


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
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        height_original, width_original = image.shape[:2]
        bounding_box = torch.Tensor(bounding_box)
        bounding_box[:, [0, 2]] /= width_original
        bounding_box[:, [1, 3]] /= height_original

        if self.transform is not None:
            for transform_fn in self.transform:
                image = transform_fn(image)

        if self.target_transform is not None:
            for transform_fn in self.target_transform:
                image, bounding_box, label = transform_fn(image, bounding_box, label)

        return image, self.target_reshape(bounding_box, label)

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

    def target_reshape(self, boxes, labels):
        center_x, center_y = (boxes[:, 2] + boxes[:, 0]) / 2., (boxes[:, 3] + boxes[:, 1]) / 2.
        width, height = (boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])

        belong_cell_x = torch.tensor((center_x // CELL_SIZE), dtype=torch.long)
        belong_cell_y = torch.tensor((center_y // CELL_SIZE), dtype=torch.long)

        center_x_percentage_for_cell = (center_x - belong_cell_x * CELL_SIZE) / CELL_SIZE
        center_y_percentage_for_cell = (center_y - belong_cell_y * CELL_SIZE) / CELL_SIZE

        target = torch.zeros((CELL_COUNT, CELL_COUNT, 30), dtype=torch.float)
        for index_x, index_y, c_x, c_y, w, h, label in zip(belong_cell_x, belong_cell_y,
                                                           center_x_percentage_for_cell,
                                                           center_y_percentage_for_cell,
                                                           width, height, labels):
            target[index_x, index_y, [4, 9]] = 1.
            target[index_x, index_y, [0, 5]] = c_x
            target[index_x, index_y, [1, 6]] = c_y
            target[index_x, index_y, [2, 7]] = w
            target[index_x, index_y, [3, 8]] = h
            # label is begin with 1,so sub 1
            target[index_x, index_y, 10 + label - 1] = 1.

        return target
