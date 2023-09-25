# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from settings import *

torch.manual_seed(1)   # 设置随机种子

class SpikeAct(torch.autograd.Function):
    """ 定义脉冲激活函数，并根据论文公式进行梯度的近似。
        Implementation of the spiking activation function with an approximation of gradient.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()     # LIF的阈值

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = abs(input - thresh) < lens
        hu = hu.float() / (2 * lens)
        return grad_input * hu
    
# 使用 apply 将方法绑定
Spike_Act = SpikeAct.apply


# noise产生器
class noise_types:
    def __init__(self, type="white"):
        self.type = type  
        if type == "color" : self.color_init = False

    def __call__(self, size, device):
        if self.type=="white":   noise = torch.normal(mean=0., std=np.sqrt(2*D_noise*delta_t), size=size, device=device)  
        if self.type=="color": 
            if self.color_init is False : 
                self.noise = delta_t*torch.normal(mean=0., std=np.sqrt(D_noise*lam_color), size=size, device=device)
                self.color_init = True 
            else:
                self.noise = self.noise - delta_t*lam_color*self.noise \
                    +lam_color*torch.normal(mean=0., std=np.sqrt(2*D_noise*delta_t), size=size, device=device)
                self.noise = delta_t*self.noise

            noise = self.noise

        return noise


# membrane potential update
def mem_update(mem, spike, W_mul_o_t1_n, noise):
    mem = mem * decay * (1. - spike) + W_mul_o_t1_n + noise
    spike = Spike_Act(mem)          # Spike_Act : approximation firing function
    return  mem, spike


class LIFSpike(nn.Module):
    """对带有时间维度的张量进行一次LIF神经元的发放模拟，可以视为一个激活函数，用法类似ReLU。
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """
    def __init__(self):
        super(LIFSpike, self).__init__()
        self.noise_m = noise_types(type=noise_type)     # 确定噪声类型

    def forward(self, x):
        # 输入数据(N, C, H, W) --> (N, C, H, W, T)
        u   = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(time_window):
            noise = self.noise_m(u.shape, u.device)
            u, out[..., step] = mem_update(u, out[..., max(step-1, 0)], x[..., step], noise)
        return out


class tdLayer(nn.Module):
    """将普通的层转换到时间域上。输入张量需要额外带有时间维，此处时间维在数据的最后一维上。前传时，对该时间维中的每一个时间步的数据都执行一次普通层的前传。
            Converts a common layer to the time domain. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data. When forwarding, a normal layer forward is performed for each time step of the data in that time dimension.

    Args:
        layer (nn.Module): 需要转换的层。
            The layer needs to convert.
        bn (nn.Module): 如果需要加入BN，则将BN层一起当做参数传入。
            If batch-normalization is needed, the BN layer should be passed in together as a parameter.
    """
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn

    def forward(self, x):
        # 输入数据(N, C, H, W) --> (N, C, H, W, T)
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (time_window,), device=x.device) # 卷积后的维度
        for step in range(time_window):
            x_[..., step] = self.layer(x[..., step])

        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class encodelayer(nn.Module):
    # 参考文献：Spike timing reshapes robustness against attacks in spiking neural networks
    def __init__(self, schemes=1):
        super(encodelayer, self).__init__()
        if schemes==1:  # 直流编码
            self.encoding = self.current_coding

        if schemes==2:  # 泊松编码
            self.encoding = self.Poisson_coding

        if schemes==3:  # Rate-Syn coding
            self.encoding = self.Rate_Syn

        if schemes==4:  # TTFS coding (Time-to-first-spike coding)
            self.encoding = self.TTFS
 
    def forward(self, x):
        return self.encoding(x) # (N, C, H, W, T)

    def current_coding(self, x):
        # necessary for general dataset: broadcast input
        x, _ = torch.broadcast_tensors(x, torch.zeros( (time_window,) + x.shape, device=x.device))  # 函数无法在最后一个维度上进行广播
        return x.permute(1, 2, 3, 4, 0)

    def Poisson_coding(self, x):
        x_ = torch.zeros(x.shape + (time_window,), device=x.device)
        for step in range(time_window):
            x_[...,step] = x > torch.rand(x.shape, device=x.device)
        return x_
    
    def Rate_Syn(self, x):
        x_ = torch.zeros(x.shape + (time_window,), device=x.device)
        t = ((1-x)*time_window).int()
        for step in range(time_window):
            x_step = torch.where(step>=t, 1., 0.)
            x_[...,step] = x_step
        return x_

    def TTFS(self, x):
        x_ = torch.zeros(x.shape + (time_window,), device=x.device)
        t = ((1-x)*time_window).int()
        for step in range(time_window):
            x_step = torch.where(step==t, 1., 0.)
            x_[...,step] = x_step
        return x_


# Threshold-dependent batch normalization
class tdBatchNorm(nn.BatchNorm2d):
    """tdBN的实现。相关论文链接：https://arxiv.org/pdf/2011.05280。具体是在BN时，也在时间域上作平均；并且在最后的系数中引入了alpha变量以及Vth。
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * thresh * (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return input


if __name__=="__main__":
    # encode = encodelayer(schemes=2)
    # input = torch.rand((1, 1, 28, 28))
    # # print(input)
    # print(encode(input).shape)

    # 测试噪声
    # noise_m = noise_types(type="white")  # color, white
    # n = 1000
    # noise = np.zeros(n)
    # for i in range(n):
    #     noise[i] = noise_m((1,), device="cpu").numpy()   # torch.device("cpu")
    
    # plt.plot(range(1,1001), noise)
    # plt.show()

    # 编码测试
    encodelayer = encodelayer(schemes=3)
    # x = torch.tensor([0.5])
    x = torch.rand(4,1,1,1)
    print(encodelayer(x))

