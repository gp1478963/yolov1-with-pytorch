import torch
import numpy
import torchvision


def test_iou_calc_without_code():
    box1 = torch.tensor([[48, 240, 195, 371]], dtype=torch.float)
    print(box1)
    box2 = torch.tensor([[96, 250, 230, 321]], dtype=torch.float)
    print(box2)
    iou_num = torchvision.ops.box_iou(box1, box2)
    print(iou_num)

    box1[:, [0, 2]] = box1[:, [0, 2]] / 353
    box1[:, [1, 3]] = box1[:, [1, 3]] / 500
    box2[:, [0, 2]] = box2[:, [0, 2]] / 353
    box2[:, [1, 3]] = box2[:, [1, 3]] / 500

    box1[:, [0, 2]] = box1[:, [0, 2]] - 0.2
    box1[:, [1, 3]] = box1[:, [1, 3]] - 0.5
    box2[:, [0, 2]] = box2[:, [0, 2]] - 0.2
    box2[:, [1, 3]] = box2[:, [1, 3]] - 0.5

    print(box1)
    print(box2)
    iou_num = torchvision.ops.box_iou(box1, box2)
    print(iou_num)


if __name__ == '__main__':
    test_iou_calc_without_code()
