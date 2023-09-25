# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from setting import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SpikeAct(torch.autograd.Function):
    """ 定义脉冲激活函数，并根据论文公式进行梯度的近似。
        Implementation of the spiking activation function with an approximation of gradient.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = abs(input - thresh) < lens
        hu = hu.float() / (2 * lens)
        return grad_input * hu
    
# 使用 apply 将方法绑定
Spike_Act = SpikeAct.apply

# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = Spike_Act(mem)          # Spike_Act : approximation firing function
    return  mem, spike

# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 32, 1, 1, 3),
           (32, 32, 1, 1, 3),]
# kernel size
cfg_kernel = [28, 14, 7]
# fc layer
cfg_fc = [128, 10]

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])   # 7x7x32 --> 128
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])  # 128 --> 10

    def forward(self, input, time_window = 20):
        # 卷积层
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        # 全连接层
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing

            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)  # Nx32x28x28
            
            x = F.avg_pool2d(c1_spike, 2)   # Nx32x14x14

            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem, c2_spike)   # Nx32x14x14

            x = F.avg_pool2d(c2_spike, 2)   # Nx32x7x7

            x = x.view(batch_size, -1)  # Nx(32x7x7)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike
        
        outputs = h2_sumspike / time_window    # firing rate of output
        return outputs


if __name__ == "__main__":

    snn = SCNN()
    print(list(snn.named_parameters()))
    # print(snn)

    # input = torch.rand((1, 1, 28, 28))
    # # snn(input)
    # with SummaryWriter(logdir="./run") as writer:
    #     writer.add_graph(snn, input)


