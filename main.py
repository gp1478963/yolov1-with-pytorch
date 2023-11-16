from net import yolonet
import torch
from dataset import voc2007

dataset_obj = voc2007.Voc2007Dataset(
    PASCAL_VOC='D:\\image\\datasets\\VOC2007\\PASCAL_VOC',
    VOCtrainval='D:\\image\\datasets\\VOC2007\\VOCtrainval_06-Nov-2007')

a, b, c = dataset_obj.__getitem__(3)
print(a, '\n', b, '\n', c, '\n')

# input = torch.rand((1, 3, 448, 448))
#
# yolo_model = yolonet.YoloNet()
# state_dict_c = yolo_model.state_dict()
# for name in state_dict_c:
#     print(name)
# output = yolo_model.forward(input)
# print(output)