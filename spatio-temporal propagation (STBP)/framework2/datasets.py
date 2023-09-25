import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from settings import *

# ===================================== cifat-10 ===================================== 
def cifar10(train=True):
    data_path = r".\cifar-10"
    # =============== 训练集 ===============
    if train:
        # transforms.ToTensor() --> 把0~255int 转换为 0~1 tensorfloat
        transform_train = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomHorizontalFlip(),    # p(=0.5)概率水平翻转
                                            #   transforms.Normalize(0.4914, 0.247)
                                        ])
        train_dataset = torchvision.datasets.CIFAR10(root= data_path, train=True, download=False, transform=transform_train)
        # shuffle-->是否随机打乱数据集，num_workers-->是否多线程处理数据
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # print(train_dataset.class_to_idx)   # 打印出类型索引对应表
        ## 先去掉transform
        # print(train_dataset[5][1])
        # plt.imshow(train_dataset[5][0])
        # plt.show()
        return train_loader
    # =============== 测试集 ===============
    else:
        transform_test = transforms.Compose([transforms.ToTensor(),
                                        #   transforms.RandomHorizontalFlip(),    # p(=0.5)概率水平翻转
                                        #   transforms.Normalize(0.4914, 0.247)
                                    ])
        test_dataset = torchvision.datasets.CIFAR10(root= data_path, train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return test_loader

# ===================================== cifat-10 ===================================== 


# ===================================== MNIST ===================================== 
def mnist(train=True):
    data_path = r".\MNIST"
    # =============== 训练集 ===============
    if train:
        # transforms.ToTensor() --> 把0~255int 转换为 0~1 tensorfloat
        transform_train = transforms.Compose([transforms.ToTensor(),
                                            # transforms.RandomHorizontalFlip(),    # p(=0.5)概率水平翻转
                                            #   transforms.Normalize(0.4914, 0.247)
                                        ])
        # 基于torch.utils.data.Dataset写的一个可迭代数据集
        train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=False, transform=transform_train)
        # shuffle-->是否随机打乱数据集，num_workers-->是否多线程处理数据
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        return train_loader
    # =============== 测试集 ===============
    else:
        transform_test = transforms.Compose([transforms.ToTensor(),
                                        #   transforms.RandomHorizontalFlip(),    # p(=0.5)概率水平翻转
                                        #   transforms.Normalize(0.4914, 0.247)
                                    ])
        test_dataset = torchvision.datasets.MNIST(root= data_path, train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        return test_loader

# ===================================== MNIST ===================================== 



if __name__=="__main__":
    cifar10() 

