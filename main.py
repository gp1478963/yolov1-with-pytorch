import os

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

# dataset_transforms = transforms.Compose([image_target_transforms.ImageStandardize()])
dataset_transforms = []
target_dataset_transforms_l = [image_target_transforms.ImageNormalize()]

device= 'gpu:0' if torch.cuda.is_available() else 'cpu'
# dataset_obj = voc2007.Voc2007Dataset(
#     PASCAL_VOC='D:\\image\\datasets\\VOC2007\\PASCAL_VOC',
#     VOCtrainval='D:\\image\\datasets\\voc2007\\VOCtrainval_06-Nov-2007',
#     transform=dataset_transforms,
#     target_transform=target_dataset_transforms_l, device=device)


CLASS_IERS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


class TargetTransformL:
    def __init__(self, CELL_COUNT, class_num, prior_box_num):
        self.CELL_COUNT = CELL_COUNT
        self.CELL_SIZE = 1 / self.CELL_COUNT
        self.class_num = class_num
        self.prior_box_num = prior_box_num

    def __call__(self, label):
        target = torch.zeros((self.CELL_COUNT, self.CELL_COUNT, self.prior_box_num * 5 + self.class_num))
        for item in label:
            box, classier, _ = item
            center_x, center_y = (box[0] + box[2]) * .5, (box[1] + box[3]) * .5
            belong_grid_x, belong_grid_y = int(center_x // self.CELL_COUNT), int(center_y // self.CELL_COUNT)
            if target[belong_grid_x, belong_grid_y, 4] == 0:
                dep = target[belong_grid_x, belong_grid_y]
                dep[:5] = torch.tensor([center_x, center_y, box[2], box[3], 1], dtype=torch.float32)
                dep[5:10] = dep[:5]
                dep[CLASS_IERS.index(classier)] = 1
        return target


transform_v2 = [image_target_transforms.ImageResize(448, 448),
                image_target_transforms.ImageNormalizeV2(),
                image_target_transforms.Convert2TorchTensor()]
train_val_data_set = voc2007.VOC2007DatasetV2(
    root_path='D:\\image\\datasets\\VOC2007\\VOCtrainval_06-Nov-2007',
    transform=transform_v2,
    target_transform=TargetTransformL(CELL_COUNT=7, class_num=20, prior_box_num=2),
    train=True,
    device=device
)

# dataset_obj_test = voc2007.Voc2007Dataset(
#     PASCAL_VOC='D:\\image\\datasets\\voc2007\\PASCAL_VOC',
#     VOCtest='D:\\image\\datasets\\voc2007\\VOCtest_06-Nov-2007',
#     transform=dataset_transforms,
#     target_transform=target_dataset_transforms_l, train=False, device=device)

PRETRAIN = True

EPOCH_STAGE_LIST = [(2, 1e-3), (15, 1e-3), (6, 1e-4), (5, 1e-5)]
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
dataloader = DataLoader(dataset=train_val_data_set, batch_size=BATCH_SIZE, shuffle=True)

# dataloader_eval = DataLoader(dataset=dataset_obj_test, batch_size=BATCH_SIZE, shuffle=False)
yolov1_train.train_stage(yolo_model, dataloader, None, optimum, evaluate_obj, device, EPOCH_STAGE_LIST)
torch.save(yolo_model.state_dict(), 'yolov1-with-pytorch.pth')
