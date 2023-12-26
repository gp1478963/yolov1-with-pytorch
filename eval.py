import numpy
import torch

import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms

import yolov1_train
from dataset import image_target_transforms
from dataset import voc2007
from eval import yolov1_loss
from net import yolonet
from net import resnet_yolo
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

BATCH_SIZE = 32

if __name__ == '__main__':
    # yolo_model = resnet_yolo.resnet50(pretrained=False).to(device)
    # yolo_model.load_state_dict('./pth/model.pth')
    # yolo_model.eval()

    dataset_transforms = []
    target_dataset_transforms_l = [image_target_transforms.ImageNormalize()]

    dataset_obj = voc2007.Voc2007Dataset(
        PASCAL_VOC='C:\\Users\\gp147\\Downloads\\archive\\PASCAL_VOC',
        VOCtrainval='C:\\Users\\gp147\\Downloads\\archive\\VOCtrainval_06-Nov-2007',
        transform=dataset_transforms,
        target_transform=target_dataset_transforms_l, device=device, train=True)
    dataloader = DataLoader(dataset=dataset_obj, batch_size=BATCH_SIZE, shuffle=False)

    for _image, _label in dataloader:
        N = _image.shape[0]

        for index in range(N):
            label = _label[index]
            ground_truths_index = label[:, :, 4] > 0
            ground_truths = label[ground_truths_index]

            # 计算iou
            # 查看每个gt对应的最大iou
            # gt对应着最大iou > threshold, tp += 1
            # gt对应着最大iou < threshold, fn += 1
            # 剩下的所有的bbox , fp + rest
            # presion = tp / (tp + fp)
            # recall = tp(tp + fn)


