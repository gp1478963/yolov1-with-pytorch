import os

import torch.optim
import torch_directml
from torch.utils.data import DataLoader
from torchvision import transforms

import yolov1_train
from dataset import image_target_transforms
from dataset import voc2007
from eval import yolov1_loss
from net import yolonet
from net import resnet_yolo
from torchvision import models

# dataset_transforms = transforms.Compose([image_target_transforms.ImageStandardize()])
dataset_transforms = []
target_dataset_transforms_l = [image_target_transforms.ImageNormalize()]

if torch_directml.is_available():
    device = torch_directml.device(0)
else:
    device = 'cpu'


dataset_obj = voc2007.Voc2007Dataset(
    PASCAL_VOC='D:\\image\\datasets\\VOC2007\\PASCAL_VOC',
    VOCtrainval='D:\\image\\datasets\\voc2007\\VOCtrainval_06-Nov-2007',
    transform=dataset_transforms,
    target_transform=target_dataset_transforms_l, device=device)

dataset_obj_test = voc2007.Voc2007Dataset(
    PASCAL_VOC='D:\\image\\datasets\\voc2007\\PASCAL_VOC',
    VOCtest='D:\\image\\datasets\\voc2007\\VOCtest_06-Nov-2007',
    transform=dataset_transforms,
    target_transform=target_dataset_transforms_l, train=False, device=device)

PRETRAIN = True

EPOCH_STAGE_LIST = [(2, 1e-2), (6, 1e-3), (40, 1e-4), (5, 1e-5)]
BATCH_SIZE = 1

USE_RESNET = True
if USE_RESNET:
    yolo_model = resnet_yolo.resnet50(pretrained=False).to(device)
    dd = yolo_model.state_dict()
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    new_state_dict = resnet.state_dict()
    for st in new_state_dict.keys():
        if st in dd.keys() and not st.startswith('fc'):
            dd[st] = new_state_dict[st]
    yolo_model.load_state_dict(dd)
else:
    yolo_model = yolonet.YoloNet().to(device)
    if PRETRAIN and os.access('./pth/model.pth', os.F_OK):
        yolo_model.load_state_dict(torch.load('./pth/model.pth'))

evaluate_obj = yolov1_loss.YoloV1Loss(device=device).to(device)
optimum = torch.optim.SGD(yolo_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
dataloader = DataLoader(dataset=dataset_obj, batch_size=BATCH_SIZE, shuffle=True)

dataloader_eval = DataLoader(dataset=dataset_obj_test, batch_size=BATCH_SIZE, shuffle=False)
yolov1_train.train_stage(yolo_model, dataloader, dataloader_eval, optimum, evaluate_obj, device, EPOCH_STAGE_LIST)
torch.save(yolo_model.state_dict(), 'yolov1-with-pytorch.pth')
