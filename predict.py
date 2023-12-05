import numpy
import torch
import cv2

import matplotlib
import torch_directml
import torchvision.ops
from torchvision.transforms import transforms

from dataset import voc2007
from net import yolonet
from net import resnet_yolo
from dataset import image_target_transforms

IMAGE_PATH = 'D:\\image\\datasets\\voc2007\\VOCtest_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages\\000004.jpg'
IOU_THRESHOLD = 0.1
CELL_COUNT = 7
CELL_SIZE = 1 / 7
IMAGE_SIZE = 448

CELL_PLEXS = 448 / 7

classicers = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

if torch_directml.is_available():
    device = torch_directml.device(0)
else:
    device = 'cpu'

dataset_transforms = [image_target_transforms.ImageResize(width=448, height=448), transforms.ToTensor()]
target_dataset_transforms_l = [image_target_transforms.ImageNormalize()]

if __name__ == "__main__":
    # define yolo net model
    print('python predict program...')
    model = resnet_yolo.resnet50(False).to(device)
    # model = yolonet.YoloNet().to(device)
    state_dict_origin = model.state_dict()
    model.load_state_dict(torch.load('./pth/model.pth'))
    model.eval()

    # dataset_obj = voc2007.Voc2007Dataset(
    #     PASCAL_VOC='E:\\voc2007\\PASCAL_VOC',
    #     VOCtest='E:\\voc2007\\VOCtest_06-Nov-2007',
    #     transform=dataset_transforms,
    #     target_transform=target_dataset_transforms_l, train=False)

    # load image with 3 channels, resize image to (448, 448)
    image_ori = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    image_ori = cv2.resize(image_ori, (448, 448), interpolation=cv2.INTER_CUBIC)
    image, _, _ = image_target_transforms.ImageNormalize()(image_ori, "", "")

    image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    prediction = model.forward(image_tensor)

    # _, prediction = dataset_obj.__getitem__(0)

    # prediction = torch.unsqueeze(prediction, 0)
    # for prediction_ in prediction.squeeze(0).reshape(-1, 30):
    #     print(prediction_)

    index_clun = numpy.array(range(CELL_COUNT))
    grid_y, grid_x = numpy.meshgrid(index_clun, index_clun, indexing='xy')

    boxs = []
    lable = []
    score = []
    for gridx, gridy in zip(grid_x, grid_y):
        for g_x, g_y in zip(gridx, gridy):
            # get prediction
            prediction_item = prediction[0, g_x, g_y]

            if prediction_item[4] >= prediction_item[9] and prediction_item[4] > IOU_THRESHOLD:
                coord = prediction_item[:4]
                score.insert(-1, prediction_item[4].data.cpu())
            elif prediction_item[9] >= prediction_item[4] and prediction_item[9] > IOU_THRESHOLD:
                coord = prediction_item[5:9]
                score.insert(-1, prediction_item[9].data.cpu())
            else:
                continue

            c_x, c_y = coord[0], coord[1]
            if c_x <= 0 or c_y <= 0:
                score.remove(-1)
                continue

            width, height = coord[2] * coord[2] * IMAGE_SIZE, coord[3] * coord[3] * IMAGE_SIZE
            if width <= 0 or height <= 0:
                score.remove(-1)
                continue

            c_x, c_y = (g_x + c_x) * CELL_PLEXS, (g_y + c_y) * CELL_PLEXS
            top_left_x, top_left_y = numpy.clip(int(c_x - width / 2), 0, 448), numpy.clip(int(c_y - height / 2), 0, 448)
            bottom_right_x, bottom_right_y = int(c_x + width / 2), int(c_y + height / 2)
            boxs.insert(-1, [top_left_x, top_left_y, bottom_right_x, bottom_right_y])
            # cv2.rectangle(image_ori, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y),
            #               color=(255, 0, 0), thickness=2)

            # 取得类别信息
            classer, probolity = torch.max(prediction_item[10:], dim=0)
            lable.insert(-1, classicers[probolity])
            # cv2.putText(image_ori, classicers[probolity], (top_left_x, top_left_y),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 0, 0))



    boxs = torch.from_numpy(numpy.array(boxs)).float()

    boxs[boxs < 0] = 0
    score = torch.from_numpy(numpy.array(score))

    keep_indices = torchvision.ops.nms(boxs, score, 0.001)

    boxs = boxs.int()

    for index_clun in keep_indices:
        box = boxs[index_clun, :]
        cv2.rectangle(image_ori, (int(box[0].data), int(box[1].data)),
                      (int(box[2].data), int(box[3].data)),
                      color=(255, 0, 0), thickness=2)
        cv2.putText(image_ori, lable[int(index_clun.data)], (int(box[0].data), int(box[1].data)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 0, 0))

    cv2.imshow('image', image_ori)
    cv2.waitKey(0)

    print('python predict program done.')
