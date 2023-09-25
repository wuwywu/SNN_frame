from settings import *
from utils import *
from models import CifarNet2
from datasets import cifar10

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, criterion, optimizer, epoch, train_loader, writer):
    model.train()   # 开启训练
    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(device)
        outputs = model(images)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1).to(device)
        loss = criterion(outputs, labels_)
        # 更新参数空间
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 可视化
        if (i+1)%100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epochs, i+1, len(train_loader), loss.item() ))
            
            # 添加标量到tensorboard中
            writer.add_scalar('Train Loss /i', loss, i + 1 + len(train_loader) * epoch)

def test(model, criterion, epoch, test_loader, writer):
    model.eval()                # 开启评估
    correct = 0
    total = 0
    _s = "conv1.weight"         # 保存可学习参数的直方图
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.float().to(device)
            outputs = model(images)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1).to(device)
            loss = criterion(outputs, labels_)
            _, predicted = outputs.cpu().max(1) # max()输出(值，索引)
            total += float(labels.size(0))
            correct += float(predicted.eq(labels).sum().item())

    acc = 100. * correct / total
    writer.add_scalar('Test Loss /epoch', loss, epoch)
    writer.add_scalar('Test Acc /epoch', acc, epoch) 

    for i, (name, param) in enumerate(model.named_parameters()):
        if _s in name:
            writer.add_histogram(name, param, epoch)

    # 可视化(准确率)
    print('Epoch [%d/%d], acc: %.5f'%(epoch+1, num_epochs, acc))

def main():
    train_loader = cifar10(train=True)
    test_loader = cifar10(train=False)

    snn = CifarNet2() # 创建模型
    snn.to(device)
    criterion = nn.MSELoss() # loss函数
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)    # 优化器
    # 使用tensorboard可视化
    board_dir = r"./log2_cifar10"+"/time_window="+str(time_window)+"_num_epochs="+str(num_epochs)+"_encodelayer_way="+str(encodelayer_way)+"_"+noise_type+"Noise="+str(D_noise)+"_bn_fn="+str(bn_fn)
    print(board_dir)
    writer = SummaryWriter(logdir=board_dir)
    for epoch in range(num_epochs):
        train(snn, criterion, optimizer, epoch, train_loader, writer)
        test(snn, criterion, epoch, test_loader, writer)

        lr_scheduler(optimizer, epoch, learning_rate, 40)

    writer.close()


if __name__ == "__main__":
    main()
    
