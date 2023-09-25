from snn_layer import *
from setting import *
from utils import *

import os
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


def main():
    data_path = r".\MNIST"
    # 训练集
    # transforms.ToTensor() --> 把0~255int 转换为 0~1 tensorfloat
    transform_train = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),    # p(=0.5)概率水平翻转
                                        #   transforms.Normalize(0.4914, 0.247)
                                    ])
    # 基于torch.utils.data.Dataset写的一个可迭代数据集
    train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=False, transform=transform_train)
    # shuffle-->是否随机打乱数据集，num_workers-->是否多线程处理数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 测试集
    transform_test = transforms.Compose([transforms.ToTensor(),
                                        #   transforms.RandomHorizontalFlip(),    # p(=0.5)概率水平翻转
                                        #   transforms.Normalize(0.4914, 0.247)
                                    ])
    test_dataset = torchvision.datasets.MNIST(root= data_path, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # plt.imshow(test_set[100][0].numpy().transpose(1,2,0))
    # plt.show()

    snn = SCNN() # 创建模型
    snn.to(device)
    criterion = nn.MSELoss() # loss函数
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

    # 使用tensorboard可视化
    writer = SummaryWriter(logdir="./log_MNIST")
    for epoch in range(num_epochs):
        train(snn, criterion, optimizer, epoch, train_loader, writer)
        test(snn, criterion, epoch, test_loader, writer)

        lr_scheduler(optimizer, epoch, learning_rate, 40)

    writer.close()
    

if __name__ == "__main__":
    main()




