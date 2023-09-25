# -*- coding: utf-8 -*-
'''
NMNISTNet 导入到网络中的大小为(N, 2, 34, 34, time_window)
'''
from models import NMNISTNet
from settings import *
from utils import *
# 导入数据集处理程序
import sys
sys.path.append("./NMNIST")
from dataset import NMNIST

import os
import torch
import torch.nn as nn
# import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, criterion, optimizer, epoch, train_loader, writer):
    model.train()   # 开启训练
    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(device)
        labels_ = labels.to(device)
        outputs = model(images)
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
            labels_ = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels_)
            _, predicted = outputs.cpu().max(1) # max()输出(值，索引)
            total += float(labels.size(0))
            labels_ = np.argmax(labels, axis=1)
            correct += float(predicted.eq(labels_).sum().item())

    acc = 100. * correct / total
    writer.add_scalar('Test Loss /epoch', loss, epoch)
    writer.add_scalar('Test Acc /epoch', acc, epoch) 

    for i, (name, param) in enumerate(model.named_parameters()):
        if _s in name:
            writer.add_histogram(name, param, epoch)


def main():
    # 训练集的地址
    src_path_train = r"./NMNIST/Train"
    src_path_test  = r"./NMNIST/Test"

    # 基于torch.utils.data.Dataset写的一个可迭代数据集(需要删除，直接删掉文件夹，不要删文件)
    path_train = r"./NMNIST/data/NMNIST_npy/Train"+"_time_window="+str(time_window)+"_DT="+str(DT)
    
    path_test = r"./NMNIST/data/NMNIST_npy/Test"+"_time_window="+str(time_window)+"_DT="+str(DT)

    # 判断文件夹是否存在(train)
    if not os.path.exists(path_train):
        # 如果不存在，创建文件夹
        os.makedirs(path_train)
        train_dataset = NMNIST(True, time_window, DT)
        train_dataset.preprocessing(src_path_train, path_train)
    else:   
        train_dataset = NMNIST(True, time_window, DT, path_train)

    # shuffle-->是否随机打乱数据集，num_workers-->是否多线程处理数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 判断文件夹是否存在(test)
    if not os.path.exists(path_test):
        # 如果不存在，创建文件夹
        os.makedirs(path_test)
        test_dataset = NMNIST(False, time_window, DT)
        test_dataset.preprocessing(src_path_test, path_test)
    else:   
        test_dataset = NMNIST(False, time_window, DT, path_test)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # print(len(test_dataset))
    # print(list(test_loader)[0][0].shape)
    # print(list(test_loader)[0][1].shape)
    # print(len(test_loader))

    snn = NMNISTNet() # 创建模型
    snn.to(device)
    criterion = nn.MSELoss() # loss函数
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)    # 优化器
    
    # 使用tensorboard可视化
    board_dir = r"./log_nmnist"+"/time_window="+str(time_window)+"_num_epochs="+str(num_epochs)+"_"+noise_type+"Noise="+str(D_noise)
    print(board_dir)
    writer = SummaryWriter(logdir=board_dir)

    for epoch in range(num_epochs):
        train(snn, criterion, optimizer, epoch, train_loader, writer)
        test(snn, criterion, epoch, test_loader, writer)

        lr_scheduler(optimizer, epoch, learning_rate, 20)

    writer.close()
    

if __name__ == "__main__":
    main()

   