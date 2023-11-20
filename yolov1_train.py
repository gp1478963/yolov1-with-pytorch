import torch
import torchvision
import torch_directml
import numpy


def train_im(model, train_loader, optimizer, criterion, device, epoch_count, learning_rate):
    set_lr(optimizer, learning_rate)
    for epoch in range(epoch_count):
        loss_list = []
        for image, target in train_loader:
            output, stds = model.forward(image)
            total_loss = criterion.forward(output, target)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_list.append(total_loss.data.cpu())
            print(numpy.sum(loss_list) / len(loss_list))


def train_stage(model, train_loader, optimizer, criterion, device, epoch_dict):
    model.to(device)
    for epoch_count, learning_rate in epoch_dict:
        train_im(model, train_loader, optimizer, criterion, device, epoch_count, learning_rate)


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
