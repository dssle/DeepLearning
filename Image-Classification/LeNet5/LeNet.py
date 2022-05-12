import torch
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
from torch import optim

# from  train_test import train


transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 128

trainset = datasets.CIFAR10(root='D:\LearningModels\Datasets\CIFAR10',train=True,download=True,transform=transform)
testset = datasets.CIFAR10(root='D:\LearningModels\Datasets\CIFAR10',train=False,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True)
classes = trainset.classes



class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10),
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


# net = LeNet5().to('cuda')
# ts.summary(net,(3,32,32))

device = "cuda" if torch.cuda.is_available() else 'cpu'
net = LeNet5().to(device)
#
#
optimizer = optim.SGD(net.parameters(),lr=1e-1,momentum=0.9,weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=5,min_lr=0.000001)
#
# if not os.path.exists('./model'):
#     os.makedirs('./model')
# else:
#     print("文件已存在")
# save_path = './model/LeNet5.pth'
#
#
# epoch = 20
# # Acc,Loss,lr = train(net,trainloader,testloader,epoch,optimizer,criterion,scheduler,save_path,writer=None,verbose=True)
# Acc,Loss,lr = train(net,trainloader,testloader,epoch,optimizer,criterion,scheduler,save_path,verbose=True)
# plot_history(epoch,Acc=Acc,Loss=Loss,lr=lr)

# test.test_AccuracyAll(testloader,device,save_path)
# test.test_AccuracySim(testloader,device,classes,save_path)
# test.SelectToTest(testloader,classes,device,save_path)

# img_path = 'D:/暂存/f1e7-hexfcvk7658542.jpg'
# img = Image.open(img_path).convert("RGB")
# test.LocalTest(save_path,img,classes,device,transform)

