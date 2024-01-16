import torch
import torchvision
import torch_directml
import numpy
import visdom
import time
import numpy as np
import os
# wind = visdom.Visdom()
# wind.line([0.], [0.], win='loss', opts=dict(title='loss'))


def train_im(model, train_loader, test_loader, optimizer, criterion, device, epoch_count, learning_rate, index):
    set_lr(optimizer, learning_rate)
    averg_loss = []
    for epoch in range(epoch_count):
        model.train()
        loss_list = []
        index = 0
        for image, _, _, target in train_loader:
            index += 1
            output = model.forward(image)
            class_loss, confidence_loss, coordance_los, total_loss = criterion.forward(output, target)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_list.append(total_loss.data.cpu())
            # averg_loss.append(numpy.sum(loss_list) / len(loss_list))
            # wind.line([numpy.sum(loss_list) / len(loss_list)], [index], win='loss', update='append')
            print('train epoch <{0}/{1}>,progress<{2}/{3}>  current batch:class_loss:[{4:.4f}], '
                  'confidence_loss:[{5:.4f}], '
                  'coordance_los:[{6:.4f}],total loss:[{7:.4f}],  average loss:[{8:.4f}]'
                  .format(epoch + 1, epoch_count, index, train_loader.__len__(),
                          class_loss, confidence_loss, coordance_los, total_loss.data.cpu(),
                          numpy.sum(loss_list) / len(loss_list)))

        index = 0
        loss_list = []

        # for image, target in test_loader:
        #     model.eval()
        #     index += 1
        #     output = model.forward(image)
        #     class_loss, confidence_loss, coordance_los, total_loss = criterion.forward(output, target)
        #     index = index + 1
        #     loss_list.append(total_loss.data.cpu())
        #     # averg_loss.append(numpy.sum(loss_list) / len(loss_list))
        #     # wind.line([numpy.sum(loss_list) / len(loss_list)], [index], win='loss', update='append')
        #     print(
        #         'eval epoch <{0}/{1}>,progress<{2}/{3}>  current batch:class_loss:[{4:.4f}], '
        #         'confidence_loss:[{5:.4f}], '
        #         'coordance_los:[{6:.4f}],total loss:[{7:.4f}],  average loss:[{8:.4f}]'
        #         .format(epoch + 1, epoch_count, index, train_loader.__len__(),
        #                 class_loss, confidence_loss, coordance_los, total_loss.data.cpu(),
        #                 numpy.sum(loss_list) / len(loss_list)))
        if os.access('pth/model.pth', os.F_OK):
            os.remove('pth/model.pth')
        torch.save(model.state_dict(), 'pth/model.pth')


def train_stage(model, train_loader, test_loader, optimizer, criterion, device, epoch_dict):
    index = 0
    model.to(device)
    for epoch_count, learning_rate in epoch_dict:
        train_im(model, train_loader, test_loader, optimizer, criterion, device, epoch_count, learning_rate, index)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
