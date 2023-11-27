import numpy
import torch
import cv2
from eval import yolov1_loss
import matplotlib
from torchvision.transforms import transforms

from dataset import voc2007
from net import yolonet
from dataset import image_target_transforms

IMAGE_PATH = 'E:\\voc2007\\VOCtest_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages\\000003.jpg'
IOU_THRESHOLD = 0.0006
CELL_COUNT = 7
CELL_SIZE = 1 / 7
IMAGE_SIZE = 448

CELL_PLEXS = 448 / 7

classicers = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

dataset_transforms = [image_target_transforms.ImageResize(width=448, height=448), transforms.ToTensor()]
target_dataset_transforms_l = [image_target_transforms.ImageNormalize()]

evaluate_obj = yolov1_loss.YoloV1Loss()

if __name__ == "__main__":
    dataset_obj = voc2007.Voc2007Dataset(
        PASCAL_VOC='D:\\image\\datasets\\VOC2007\\PASCAL_VOC',
        VOCtest='D:\\image\\datasets\\VOC2007\\VOCtest_06-Nov-2007',
        transform=dataset_transforms,
        target_transform=target_dataset_transforms_l, train=False)

    _, prediction = dataset_obj.__getitem__(0)
    prediction = torch.unsqueeze(prediction, dim=0)
    loss = evaluate_obj.forward(prediction, prediction)
    print(loss.data.cpu().numpy())




















