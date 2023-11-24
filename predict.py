import numpy
import torch
import cv2

import matplotlib

from net import yolonet
from dataset import image_target_transforms

IMAGE_PATH = 'D:\\image\\01faab58bcfe3fa801219c776168a6.jpg@1280w_1l_2o_100sh.jpg'
IOU_THRESHOLD = 0.75
CELL_COUNT = 7
CELL_SIZE = 1/7
IMAGE_SIZE = 448

CELL_PLEXS = 448/7


if __name__ == "__main__":
    # define yolo net model
    print('python predict program...')
    model = yolonet.YoloNet()
    state_dict_origin = model.state_dict()
    model.load_state_dict(torch.load('./pth/yolo.pth'))
    model.eval()

    # load image with 3 channels, resize image to (448, 448)
    image_ori = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    image_ori = cv2.resize(image_ori, (448, 448), cv2.INTER_CUBIC)
    image, _, _ = image_target_transforms.ImageNormalize()(image_ori, "", "")

    image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float()
    prediction, stds = model.forward(image_tensor)

    # for prediction_ in prediction.squeeze(0).reshape(-1, 30):
    #     print(prediction_)

    index_clun = numpy.array(range(CELL_COUNT))
    grid_y, grid_x = numpy.meshgrid(index_clun, index_clun, indexing='xy')
    for gridx, gridy in zip(grid_x, grid_y):
        for g_x, g_y in zip(gridx, gridy):
            # get prediction
            prediction_item = prediction[0, g_x, g_y]

            if prediction_item[4] > prediction_item[9]:
                coord = prediction_item[:4]
            else:
                coord = prediction_item[5:9]

            c_x, c_y = coord[0], coord[1]
            if c_x <= 0 or c_y <= 0:
                continue

            width, height = coord[2]*IMAGE_SIZE, coord[3]*IMAGE_SIZE
            if width <= 0 or height <= 0:
                continue

            c_x, c_y = (g_x + c_x) * CELL_PLEXS, (g_y + c_y) * CELL_PLEXS
            top_left_x, top_left_y = int(c_x - width), int(c_y - height)
            bottom_right_x, bottom_right_y = int(c_x + width), int(c_y + height)
            cv2.rectangle(image_ori, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y),
                          color=(255, 0, 0), thickness=2)
    cv2.imshow('image', image_ori)
    cv2.waitKey(0)












    print('python predict program done.')






















