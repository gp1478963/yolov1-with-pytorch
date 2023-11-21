import torch
import torchvision
import torch_directml
import numpy
import visdom
import time
import numpy as np

wind = visdom.Visdom()
wind.line([0.], [0.], win='loss', opts=dict(title='loss'))
def train_im(model, train_loader, optimizer, criterion, device, epoch_count, learning_rate, index):

    set_lr(optimizer, learning_rate)
    averg_loss = []
    for epoch in range(epoch_count):
        loss_list = []

        for image, target in train_loader:
            output, stds = model.forward(image)
            total_loss = criterion.forward(output, target)
            index = index + 1
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_list.append(total_loss.data.cpu())
            # averg_loss.append(numpy.sum(loss_list) / len(loss_list))
            print(index)
            wind.line([numpy.sum(loss_list) / len(loss_list)], [index], win='loss', update='append')
            # print(average_loss)


def train_stage(model, train_loader, optimizer, criterion, device, epoch_dict):
    index = 0
    model.to(device)
    model.train()
    for epoch_count, learning_rate in epoch_dict:
        train_im(model, train_loader, optimizer, criterion, device, epoch_count, learning_rate, index)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
