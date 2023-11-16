from net import yolonet
import torch
from dataset import voc2007
from dataset import image_target_transforms

from torchvision import transforms
from torch.utils.data import DataLoader
# dataset_transforms = transforms.Compose([image_target_transforms.ImageStandardize()])
dataset_transforms = [image_target_transforms.ImageResize(width=448, height=448), transforms.ToTensor()]
target_dataset_transforms_l = [image_target_transforms.ImageNormalize()]

batch_size = 1
yolo_model = yolonet.YoloNet()
epoch_count = 1

dataset_obj = voc2007.Voc2007Dataset(
    PASCAL_VOC='D:\\image\\datasets\\VOC2007\\PASCAL_VOC',
    VOCtrainval='D:\\image\\datasets\\VOC2007\\VOCtrainval_06-Nov-2007',
    transform=dataset_transforms,
    target_transform=target_dataset_transforms_l)

dataloader = DataLoader(dataset=dataset_obj, batch_size=batch_size, shuffle=False)
for epoch in range(epoch_count):
    for image, target in dataloader:
        # output = yolo_model.forward(image)
        # print(output)
        print(target)
        break





# a, b, c = dataset_obj.__getitem__(3)

# input = torch.rand((1, 3, 448, 448))
#
# state_dict_c = yolo_model.state_dict()
# for name in state_dict_c:
#     print(name)
# output = yolo_model.forward(input)
# print(output)
