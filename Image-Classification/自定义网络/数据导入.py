import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt


# 数据处理类
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


Batch_size = 256

# 数据导入和处理
'''
torchvision.datasets中的download=True为加载，如果在root下能找到本地数据集，则进行加载（生成一个加载文件夹）；否则从网上下载数据集并加载。
download=False不管本地是否数据集，不加载也不下载。
'''
trainset = datasets.CIFAR10(root='D:\LearningModels\Datasets\CIFAR10', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='D:\LearningModels\Datasets\CIFAR10', train=False,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_size, shuffle=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(classes)
classes = trainset.classes
print(classes)
#classes1 = trainset.clase_to_idx
# classes1
# print(classes1)

# shape = trainset.data.shape
# print(shape)
#
# print(type(trainset.data))
# print(type(trainset))
#
#
#
# plt.imshow(trainset.data[0])
# plt.show()
im,label = iter(trainloader).next()
#
# l = [1, 2, 3]
# for i in iter(l):
#       print(i)
#
# a = 3/2
# b = 8//5
# print(a,b)
def imshow(img):
    img = img / 2 + 0.5
    img = np.transpose(img.numpy(),(1,2,0))
    plt.imshow(img)
plt.figure(figsize=(8,12))

imshow(torchvision.utils.make_grid(im[:32]))
plt.show()

imshow(im[1])

