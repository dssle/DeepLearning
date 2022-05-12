import torch
import torch.nn as nn
import torchsummary as ts


class AlexNet (nn.Module):
    def __init__(self,num_classes=10,init_weight=False):
        super(AlexNet,self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(384,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )

        self.clssifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6,2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048,num_classes)
        )
        if init_weight:
            self._initialize_weight()

    def forward(self,x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = x.view(x.size(0),-1)
        output = self.clssifier(x)
        return output

    def _initialize_weight(self):
        # 用m遍历Sequential，如果有卷积层则初始化，否则初始化全连接层
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)
# 实例化并可视化网络
# device  = 'cuda' if torch.cuda.is_available() else 'cpu'
# net = AlexNet(10,True).to(device)
# ts.summary(net,(3,224,224))
# pass







