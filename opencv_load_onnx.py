import cv2
import torch
from torchvision import transforms

from dataset import image_target_transforms



dataset_transforms = [image_target_transforms.ImageResize(width=448, height=448), transforms.ToTensor()]
target_dataset_transforms_l = [image_target_transforms.ImageNormalize()]
IMAGE_PATH = 'D:\\image\\datasets\\VOC2007\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages\\000112.jpg'
image_ori = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
image_ori = cv2.resize(image_ori, (448, 448), interpolation=cv2.INTER_CUBIC)
image, _, _ = image_target_transforms.ImageNormalize()(image_ori, "", "")
image_tensor = torch.from_numpy(image).permute( 0, 1 ,2 ).float().numpy()

yolo_net = cv2.dnn.readNetFromONNX('./pth/lmodel_cpu.onnx')
blob = cv2.dnn.blobFromImage(image_tensor)
yolo_net.setInput(blob)
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
out = yolo_net.forward()
print(out)







