import copy

import torch
import onnx
import onnxscript
from torchvision import transforms

from dataset import image_target_transforms
from net import resnet_yolo

print(torch.__version__)
print(onnxscript.__version__)

from onnxscript import opset18  # opset 18 is the latest (and only) supported version for now

import onnxruntime

print(onnxruntime.__version__)
device = 'cpu'

def utils_save_list(list_, dst):
    file = open(dst, 'a')
    list_ = list_[0]
    for i in range(len(list_)):
        for j in range(len(list_[i])):
            for k in range(len(list_[i][j])):
                str_all = ''
                for s in range(len(list_[i][j][k])):
                    str_all = str_all + str(list_[i][j][k][s]) + '\t'
                file.write(str_all + '\n')
    file.close()

def convert2_onnx():
    # load torch model
    model = resnet_yolo.resnet50(False).to(device)
    state_dict_origin = model.state_dict()
    model.load_state_dict(torch.load('./pth/lmodel.pth', map_location=torch.device('cpu')))
    model.eval()

    input_shape = torch.randn(1, 3, 448, 448, requires_grad=True)
    out_shape = torch.rand(1, 7, 7, 30)

    torch.onnx.export(model=model, args=input_shape, f='./pth/lmodel_' + device + '.onnx', export_params=True,
                      verbose=True,
                      input_names=['input_tensor'],
                      output_names=['output_tensor']
                      )

def onnx_checker():
    onnx_model = onnx.load('./pth/lmodel.onnx')
    onnx.checker.check_model(onnx_model)

def do_inference():
    import cv2
    dataset_transforms = [image_target_transforms.ImageResize(width=448, height=448), transforms.ToTensor()]
    target_dataset_transforms_l = [image_target_transforms.ImageNormalize()]
    IMAGE_PATH = 'D:\\image\\datasets\\VOC2007\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages\\000112.jpg'
    image_ori = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    image_ori = cv2.resize(image_ori, (448, 448), interpolation=cv2.INTER_CUBIC)
    image, _, _ = image_target_transforms.ImageNormalize()(image_ori, "", "")
    image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float().numpy()


    onnx_model = onnx.load("./pth/lmodel_cpu.onnx")
    for initializer in onnx_model.graph.initializer:
        print(initializer.name)
        if initializer.name == 'onnx::Conv_597':
            W = onnx.numpy_helper.to_array(initializer)
            print(initializer.name, W.shape)
            print(W)


    ort_session = onnxruntime.InferenceSession("./pth/lmodel_cpu.onnx", providers=["CPUExecutionProvider"])



    print(ort_session.get_inputs()[0].name)
    print(ort_session.get_outputs()[0].name)
    ort_inputs = {ort_session.get_inputs()[0].name: image_tensor}
    output_result = ort_session.run(None, input_feed=ort_inputs)
    utils_save_list(output_result, './out/lmodel_cpu_onnx_inference_resule_for_000112.txt')
    print(output_result)




if __name__ == '__main__':
    print('start...')
    # convert2_onnx()
    # onnx_checker()
    do_inference()
    print('done...')
    pass
