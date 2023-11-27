import torch.optim
import torch_directml
from torch.utils.data import DataLoader
from torchvision import transforms

import yolov1_train
from dataset import image_target_transforms
from dataset import voc2007
from eval import yolov1_loss
from net import yolonet

# dataset_transforms = transforms.Compose([image_target_transforms.ImageStandardize()])
dataset_transforms = [image_target_transforms.ImageResize(width=448, height=448), transforms.ToTensor()]
target_dataset_transforms_l = [image_target_transforms.ImageNormalize()]


if torch_directml.is_available():
    device = torch_directml.device(0)
else:
    device = 'cpu'
device = 'cpu'
dataset_obj = voc2007.Voc2007Dataset(
    PASCAL_VOC='D:\\image\\datasets\\VOC2007\\PASCAL_VOC',
    VOCtrainval='D:\\image\\datasets\\VOC2007\\VOCtrainval_06-Nov-2007',
    transform=dataset_transforms,
    target_transform=target_dataset_transforms_l, device=device)


PRETRAIN = False

EPOCH_STAGE_LIST = [(1, 1e-3), (75, 1e-3), (30, 1e-3), (30, 1e-4)]
BATCH_SIZE = 1

yolo_model = yolonet.YoloNet().to(device)
if PRETRAIN:
    yolo_model.load_state_dict(torch.load('./pth/model.pth'))

evaluate_obj = yolov1_loss.YoloV1Loss(device=device).to(device)
optimum = torch.optim.SGD(yolo_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
dataloader = DataLoader(dataset=dataset_obj, batch_size=BATCH_SIZE, shuffle=True)

yolov1_train.train_stage(yolo_model, dataloader, optimum, evaluate_obj, device, EPOCH_STAGE_LIST)
torch.save(yolo_model, 'yolov1-with-pytorch.pth')


