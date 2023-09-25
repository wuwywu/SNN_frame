import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from settings import *

# 手写数据集
class MNISTNet(nn.Module):  # Example net for MNIST
    def __init__(self):
        super(MNISTNet, self).__init__()
        if bn_fn:
            self.bn1 = tdBatchNorm(15)
            self.bn2 = tdBatchNorm(40)
        else:
            self.bn1 = None
            self.bn2 = None
        self.conv1 = nn.Conv2d(1, 15, 5, 1, 2, bias=None)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(15, 40, 5, 1, 2, bias=None)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(7 * 7 * 40, 300)
        self.fc2 = nn.Linear(300, 10)

        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.pool1_s = tdLayer(self.pool1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.pool2_s = tdLayer(self.pool2)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)

        self.spike1 = LIFSpike()
        self.spike2 = LIFSpike()
        self.spike3 = LIFSpike()
        self.spike4 = LIFSpike()
        self.spike5 = LIFSpike()
        self.spike6 = LIFSpike()

        # 1、直流编码;2、泊松编码
        self.encodelayer = encodelayer(schemes=encodelayer_way)   

    def forward(self, x):
        # x --> (N, C, H, W)
        x = self.encodelayer(x)
        # x --> (N, C, H, W, T)
        x = self.conv1_s(x)
        x = self.spike1(x)
        x = self.pool1_s(x)
        x = self.spike2(x)
        x = self.conv2_s(x)
        x = self.spike3(x)
        x = self.pool2_s(x)
        x = self.spike4(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike5(x)
        x = self.fc2_s(x)
        x = self.spike6(x)
        out = torch.sum(x, dim=2) / time_window   # [N, neurons, steps]
        return out
    

# 手写数据集（神经形态）
class NMNISTNet(nn.Module):  # Example net for N-MNIST
    def __init__(self):
        super(NMNISTNet, self).__init__()
        if bn_fn:
            self.bn1 = tdBatchNorm(20)
            self.bn2 = tdBatchNorm(50)
        else:
            self.bn1 = None
            self.bn2 = None
        self.conv1 = nn.Conv2d(2, 20, 3, 1, padding=0)  # 数据集(N, P, H, W, T)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(7 * 7 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.pool1_s = tdLayer(self.pool1, self.bn2)
        self.conv2_s = tdLayer(self.conv2)
        self.pool2_s = tdLayer(self.pool2)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)

        self.spike1 = LIFSpike()
        self.spike2 = LIFSpike()
        self.spike3 = LIFSpike()
        self.spike4 = LIFSpike()
        self.spike5 = LIFSpike()
        self.spike6 = LIFSpike()

    def forward(self, x): 
        # x --> (N, P, H, W)
        # x = self.encodelayer(x)
        # x --> (N, p, H, W, T)
        x = self.conv1_s(x)
        x = self.spike1(x)
        x = self.pool1_s(x)
        x = self.spike2(x)
        x = self.conv2_s(x)
        x = self.spike3(x)
        x = self.pool2_s(x)
        x = self.spike4(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike5(x)
        x = self.fc2_s(x)
        x = self.spike6(x)
        out = torch.sum(x, dim=2) / time_window  # [N, neurons, time_window]
        return out
    

# CifarNet 1
class CifarNet1(nn.Module):  # Example net for CIFAR10
    def __init__(self):
        super(CifarNet1, self).__init__()
        #self.conv0 = nn.Conv2d(1, 1, 5, 2)
        self.conv0 = nn.Conv2d(3, 128, 3, 1, 1, bias=None)
        self.bn0 = tdBatchNorm(128)
        self.conv1 = nn.Conv2d(128, 256, 3, 1, 1, bias=None)
        self.bn1 = tdBatchNorm(256)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(256, 512, 3, 1, 1, bias=None)
        self.bn2 = tdBatchNorm(512)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(512, 1024, 3, 1, 1, bias=None)
        self.bn3 = tdBatchNorm(1024)
        self.conv4 = nn.Conv2d(1024, 512, 3, 1, 1, bias=None)
        self.bn4 = tdBatchNorm(512)
        self.fc1 = nn.Linear(8 * 8 * 512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        self.conv0_s = tdLayer(self.conv0, self.bn0)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.pool1_s = tdLayer(self.pool1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.pool2_s = tdLayer(self.pool2)
        self.conv3_s = tdLayer(self.conv3, self.bn3)
        self.conv4_s = tdLayer(self.conv4, self.bn4)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)
        self.fc3_s = tdLayer(self.fc3)

        # Nspike = 10
        # 因为这样就集中在一个列表中了，tensorboard的图中显示可能会不正确
        # self.spike = [LIFSpike() for _ in range(Nspike)]
        self.spike0 = LIFSpike()
        self.spike1 = LIFSpike()
        self.spike2 = LIFSpike()
        self.spike3 = LIFSpike()
        self.spike4 = LIFSpike()
        self.spike5 = LIFSpike()
        self.spike6 = LIFSpike()
        self.spike7 = LIFSpike()
        self.spike8 = LIFSpike()
        self.spike9 = LIFSpike()

        # 1、直流编码;2、泊松编码
        self.encodelayer = encodelayer(schemes=encodelayer_way)

    def forward(self, x):
        # x --> (N, C, H, W)
        x = self.encodelayer(x)
        # x --> (N, C, H, W, T)
        x = self.conv0_s(x)
        x = self.spike0(x)
        x = self.conv1_s(x)
        x = self.spike1(x)
        x = self.pool1_s(x)
        x = self.spike2(x)
        x = self.conv2_s(x)
        x = self.spike3(x)
        x = self.pool2_s(x)
        x = self.spike4(x)
        x = self.conv3_s(x)
        x = self.spike5(x)
        x = self.conv4_s(x)
        x = self.spike6(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike7(x)
        x = self.fc2_s(x)
        x = self.spike8(x)
        x = self.fc3_s(x)
        x = self.spike9(x)
        out = torch.sum(x, dim=2) / time_window  # [N, neurons, time_window]
        return out
    
    def reset_parameters(self):
        # 重新初始化模型的参数
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                # print(module)


# CifarNet 2
class CifarNet2(nn.Module):  # Example net for CIFAR10
    def __init__(self):
        super(CifarNet2, self).__init__()
        if bn_fn:
            self.bn1 = tdBatchNorm(128)
            self.bn2 = tdBatchNorm(128)
        else:
            self.bn1 = None
            self.bn2 = None
        self.conv1 = nn.Conv2d(3, 128, 5, 1, 2, bias=None)     
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(128, 128, 5, 1, 2, bias=None)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(8192, 512)
        self.fc2 = nn.Linear(512, 10)

        self.conv1_s = tdLayer(self.conv1, self.bn1)  # , self.bn1
        self.pool1_s = tdLayer(self.pool1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)  # , self.bn2
        self.pool2_s = tdLayer(self.pool2)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)

        self.spike1 = LIFSpike()
        self.spike2 = LIFSpike()
        self.spike3 = LIFSpike()
        self.spike4 = LIFSpike()
        self.spike5 = LIFSpike()
        self.spike6 = LIFSpike()

        # 1、直流编码;2、泊松编码
        self.encodelayer = encodelayer(schemes=encodelayer_way)   

    def forward(self, x):
        # x --> (N, C, H, W)
        x = self.encodelayer(x)
        # x --> (N, C, H, W, T)
        x = self.conv1_s(x)
        x = self.spike1(x)
        x = self.pool1_s(x)
        x = self.spike2(x)
        x = self.conv2_s(x)
        x = self.spike3(x)
        x = self.pool2_s(x)
        x = self.spike4(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike5(x)
        x = self.fc2_s(x)
        x = self.spike6(x)
        out = torch.sum(x, dim=2) / time_window   # [N, neurons, steps]
        return out
    
    def reset_parameters(self):
        # 重新初始化模型的参数
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                # print(module)
    


if __name__ == "__main__":

    # snn = MNISTNet()
    snn = CifarNet1()
    snn.train()   # 开启训练
    # print(list(snn.named_parameters()))
    # print(snn)

    # input = torch.rand((1, 1, 28, 28))
    input = torch.rand((1, 3, 32, 32))
    # snn(input)

    # board_dir = r"./log_mnist"+"/model"
    board_dir = r"./log_cifar10"+"/model"
    with SummaryWriter(logdir=board_dir) as writer:
        writer.add_graph(snn, input)
