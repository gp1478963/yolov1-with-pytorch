import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import image_target_transforms

from dataset import voc2007


def grid_cell_absolut_location_(percent):
    y, x = torch.meshgrid(torch.range(0, 6), torch.range(0, 6), indexing='xy')
    percent[:, :, 1] = percent[:, :, 1]/7. + x/7.
    percent[:, :, 0] = percent[:, :, 0]/7. + y/7.

if __name__ == '__main__':
    dataset_transforms = []
    target_dataset_transforms_l = [image_target_transforms.ImageNormalize()]

    device = "cpu"
    dataset_obj = voc2007.Voc2007Dataset(
        PASCAL_VOC='C:\\Users\\gp147\\Downloads\\archive\\PASCAL_VOC',
        VOCtrainval='C:\\Users\\gp147\\Downloads\\archive\\VOCtrainval_06-Nov-2007',
        transform=dataset_transforms,
        target_transform=target_dataset_transforms_l, device=device, train=False)
    dataloader = DataLoader(dataset=dataset_obj, batch_size=1, shuffle=True)
    for _, target, image_original in dataloader:
        target = target.squeeze(0)
        grid_cell_absolut_location_(target)
        target = target.reshape(-1, 30)
        keep = (target[..., [4]] == 1).squeeze(1)
        boxes = target[keep][:, :4].contiguous()
        label_inter = target[keep][:, 10:].contiguous()
        image_original = torch.squeeze(image_original, 0).numpy()
        for box, label in zip(boxes, label_inter):
            center_x, center_y, width, height = box
            left_x, left_y = center_x - width / 2, center_y - height / 2
            right_x, right_y = center_x + width / 2, center_y + height / 2
            left_x, right_x = np.clip(left_x.numpy() *image_original.shape[1], a_min=0, a_max=1000), np.clip(right_x.numpy() *image_original.shape[1], a_min=0, a_max=1000)
            left_y, right_y = np.clip(left_y.numpy() * image_original.shape[0], a_min=0, a_max=1000), right_y.numpy() * image_original.shape[0]
            cv2.rectangle(image_original, (int(left_x), int(left_y)),
                          (int(right_x), int(right_y)),
                          color=(255, 0, 0), thickness=2)

        cv2.imshow('image', image_original)
        cv2.waitKey(0)
