import torch
import torchvision
from torch import nn
import numpy


class YoloNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # 1
        self.conv1_0 = torch.nn.Conv2d(in_channels=3, out_channels=64,
                                       kernel_size=7, stride=2, padding=3)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        # 2
        self.conv2_0 = torch.nn.Conv2d(in_channels=64, out_channels=192,
                                       kernel_size=3, stride=1, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

        # 3
        self.conv3_0 = torch.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1)
        self.conv3_1 = torch.nn.Conv2d(in_channels=128, out_channels=256,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.conv3_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.conv3_3 = torch.nn.Conv2d(in_channels=256, out_channels=512,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        # 4
        self.conv4_0 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.conv4_1 = torch.nn.Conv2d(in_channels=256, out_channels=512,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.conv4_2 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.conv4_3 = torch.nn.Conv2d(in_channels=256, out_channels=512,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.conv4_4 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.conv4_5 = torch.nn.Conv2d(in_channels=256, out_channels=512,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.conv4_6 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.conv4_7 = torch.nn.Conv2d(in_channels=256, out_channels=512,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.conv4_8 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        self.conv4_9 = torch.nn.Conv2d(in_channels=512, out_channels=1024,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2)

        # 5
        self.conv5_0 = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
        self.conv5_1 = torch.nn.Conv2d(in_channels=512, out_channels=1024,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.conv5_2 = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
        self.conv5_3 = torch.nn.Conv2d(in_channels=512, out_channels=1024,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.conv5_4 = torch.nn.Conv2d(in_channels=1024, out_channels=1024,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.maxpool5 = torch.nn.MaxPool2d(kernel_size=2)

        # 6
        self.conv6_0 = torch.nn.Conv2d(in_channels=1024, out_channels=1024,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.conv6_1 = torch.nn.Conv2d(in_channels=1024, out_channels=1024,
                                       kernel_size=3, stride=1, padding='same', padding_mode='zeros')

        # 7 fc
        self.fc7_0 = torch.nn.Linear(in_features=(1024 * 7 * 7), out_features=1024)
        self.dropout7 = torch.nn.Dropout(p=0.5)
        self.fc7_1 = torch.nn.Linear(in_features=1024, out_features=(30 * 7 * 7))

        self.weights_init()

    def forward(self, x):
        N = x.size()[0]
        stds = []
        # 1
        x = self.maxpool1(self.relu(self.conv1_0(x)))
        # stds.append(x.std())

        # 2
        x = self.maxpool2(self.relu(self.conv2_0(x)))
        # stds.append(x.std())

        # 3
        x = self.relu(self.conv3_0(x))
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.maxpool2(x)
        # stds.append(x.std())

        # 4
        x = self.relu(self.conv4_0(x))
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.relu(self.conv4_4(x))
        x = self.relu(self.conv4_5(x))
        x = self.relu(self.conv4_6(x))
        x = self.relu(self.conv4_7(x))
        x = self.relu(self.conv4_8(x))
        x = self.relu(self.conv4_9(x))
        x = self.maxpool2(x)
        # stds.append(x.std())

        # 5
        x = self.relu(self.conv5_0(x))
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.relu(self.conv5_4(x))
        x = self.maxpool2(x)
        # stds.append(x.std())

        # 6
        x = self.relu(self.conv6_0(x))
        x = self.relu(self.conv6_1(x))
        # stds.append(x.std())

        # 7 fc
        x = self.fc7_0(nn.Flatten()(x))
        # A dropout layer with rate = .5 after the first
        # connected layer prevents co-adaptation between layers
        x = self.dropout7(x)
        # stds.append(x.std())
        x = self.fc7_1(x).reshape(N, 7, 7, 30)
        return x, stds

    def weights_init(self):
        for modl in self.modules():
            if isinstance(modl, nn.Conv2d):
                nn.init.normal_(modl.weight.data,
                                std=numpy.sqrt(1 / (modl.kernel_size[0] * modl.kernel_size[
                                    1] * modl.in_channels * modl.out_channels)))
            if isinstance(modl, nn.Linear):
                nn.init.normal_(modl.weight.data, std=numpy.sqrt(1 / (modl.in_features * modl.out_features)))
        # pass
