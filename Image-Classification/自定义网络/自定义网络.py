import torch
import torch.nn as nn
import torchsummary as ts
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import test
import cv2
from PIL import Image


## 数据导入
# 数据处理类
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


Batch_size = 128

trainset = datasets.CIFAR10(root='D:\LearningModels\Datasets\CIFAR10', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='D:\LearningModels\Datasets\CIFAR10', train=False,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_size, shuffle=True)
classes = trainset.classes
print(classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Mynet(nn.Module):
    def __init__(self,num_classes=10):
        super(Mynet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,padding=1),

            nn.Conv2d(128,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,padding=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*9*9,2048),
            nn.ReLU(True),
            nn.Linear(2048,num_classes)
        )
    def forward(self,x):
        out = self.features(x)
        print(x.shape)
        print(out.shape)
        out = out.view(out.size(0),-1)
        print(out.shape)
        out = self.classifier(out)
        print(out.shape)
        return out
net = Mynet().to(device)
ts.summary(net,(3,32,32),device=device)
# print(net)

# test_x = torch.randn(1,3,32,32).to(device)
# test_y = net(test_x)
# print(test_y.shape)

# optimizer = optim.SGD(net.parameters(),lr = 1e-1,momentum=0.9,weight_decay=5e-4)
# criterion = nn.CrossEntropyLoss()
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=5,min_lr = 0.000001)
# import time
# epoch = 2
# import os
# if not os.path.exists('./model'):
#     os.makedirs('./model')
# else:
#     print("文件已存在")
# save_path = './model/Mynet.pth'



# from utils import train
# from utils import plot_history
# Acc, Loss, Lr = train(net, trainloader, testloader, epoch, optimizer, criterion, scheduler, save_path, verbose = True)
# plot_history(epoch ,Acc, Loss, Lr)

# test.test_AccuracyAll(testloader,device,save_path) #测试总的准确率
# test.test_AccuracySim(testloader,device,classes,save_path) #测试每一类的准确率
# test.SelectToTest(testloader,classes,device,save_path) #选取一部分图像看检测结果

#

# 选取本地图片进行检测
# img_path = 'D:/暂存/1.jpg'
# img = Image.open(img_path).convert('RGB')
# # print(img.size,img.mode,img.format)
# # img.show()
# img = transform(img)
# img = img.to(device)
# img = img.unsqueeze(0)
# # test.LocalTest(save_path,img,classes,device,transform)


# import requests
# url1 = "https://cn.bing.com/images/search?view=detailV2&ccid=WxsV9Lcp&id=E94CBFD58482E2CACA73C643E4A48EC6F8CF1A06&thid=OIP.WxsV9LcpTwwAd04Z1K0aqAHaFj&mediaurl=https%3a%2f%2ftse1-mm.cn.bing.net%2fth%2fid%2fR-C.5b1b15f4b7294f0c00774e19d4ad1aa8%3frik%3dBhrP%252bMaOpORDxg%26riu%3dhttp%253a%252f%252fnewssrc.onlinedown.net%252fimgs%252f20170619%252f20170619_065005_549.jpg%26ehk%3d6M3ecP%252bof144KJ%252bHKIGA8fdLzQi57Ba0W45uybLpv9c%253d%26risl%3d%26pid%3dImgRaw%26r%3d0&exph=898&expw=1196&q=A380&simid=608043429539688568&FORM=IRPRST&ck=70445361A1B009391E9B5ED2E12BE6A2&selectedIndex=0&ajaxhist=0&ajaxserp=0"
# url1 = "https://cn.bing.com/images/search?view=detailV2&ccid=5oJ4jLwn&id=F26EC809617C0F80C2C47AA534D72AA29FA6D65F&thid=OIP.5oJ4jLwngv-SGfB1Y5MrIgHaEK&mediaurl=https%3A%2F%2Ftse1-mm.cn.bing.net%2Fth%2Fid%2FR-C.e682788cbc2782ff9219f07563932b22%3Frik%3DX9amn6Iq1zSleg%26riu%3Dhttp%253a%252f%252fcdn.feeyo.com%252fpic%252f20140927%252f201409271046298816.jpg%26ehk%3DantJNnR0V3jIpXswNY1Du0dzYlHi8zTEBwTjKNeiTA8%253d%26risl%3D%26pid%3DImgRaw%26r%3D0&exph=900&expw=1600&q=A380&simid=608005500674183542&form=IRPRST&ck=811C21F0FCF3D6F3019077DE4CD64C45&selectedindex=3&ajaxhist=0&ajaxserp=0&vt=0&sim=11"
# url1 = "https://cn.bing.com/images/search?view=detailV2&ccid=REaQZkAP&id=DF99F1CFAC7E0001637B2F6A5F3712A92B0BBF08&thid=OIP.REaQZkAPf59ecPZILHk3MQHaHa&mediaurl=https%3a%2f%2fpic3.zhimg.com%2fv2-44469066400f7f9f5e70f6482c793731_r.jpg%3fsource%3d1940ef5c&exph=2448&expw=2448&q=%e7%8b%97&simid=608024823733952418&FORM=IRPRST&ck=B1359E3B5A9BFC7114C1671A956246EE&selectedIndex=0&ajaxhist=0&ajaxserp=0"
#
# response = requests.get(url1)
# # print(reponse)
# # reponse_json = reponse.json()
# # print("test:",reponse.text)
# # print(reponse_json)
# img = Image.open(response)





