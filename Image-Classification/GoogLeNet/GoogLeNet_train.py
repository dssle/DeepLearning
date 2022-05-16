# from GoogLeNet_model import GoogLeNet
from torchvision.models import GoogLeNet
import os.path
import sys
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import time
from GoogLeNet_utils import get_acc,plot_history1,write_result

def train(batch_size,epoch,data_root,model_path,record_result=False):

    # 数据增强
    data_transform = {
        'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]),
        'val':transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]),
    }

    # 路径设置
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    else:
        print('model_path is already exit')
    model_savepath = os.path.join(model_path,'Mynet.pth')

    # 配置数据集
    # trainset = datasets.CIFAR10(root=data_root,train=True,download=True,transform=data_transform['train'])
    # testset = datasets.CIFAR10(root=data_root,train=False,download=True,transform=data_transform['val'])

    trainset = datasets.ImageFolder(root=os.path.join(data_root,'train'),transform=data_transform['train'])
    testset = datasets.ImageFolder(root=os.path.join(data_root, 'val'),transform=data_transform['val'])

    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True)

    # 查看图片
    # im, label = iter(trainloader).next()
    # plt.figure()
    # imshow(torchvision.utils.make_grid(im[:32]))
    # plt.show()

    # 选择硬件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using %s device." % device)


    # 实例化网络
    # net = GoogLeNet(num_classes=5,aux_logits=True,init_weights=True)
    # net = net.to(device)

    # 迁移学习
    net = GoogLeNet(num_classes=5)
    # 导出模型权重
    model_dict = net.state_dict()
    # 导入预训练模型权重
    pretrain_model = torch.load('./model_pre/googlenet_pre.pth')
    # 排除分类器最后连接层的权重
    # 排除主分类器最后一层全连接层，辅助分类器最后一层全连接层的权重和偏置，其他更换为预训练模型权重
    del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
                "aux2.fc2.weight", "aux2.fc2.bias",
                 "fc.weight", "fc.bias"]
    # 提取所需的预训练模型权重
    pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
    # 更新模型权重
    model_dict.update(pretrain_dict)
    # 网络载入新的权重
    net.load_state_dict(model_dict)
    net = net.to(device)

    # 设置优化器、损失函数和学习率优化器
    optimizer = optim.SGD(net.parameters(),lr=3e-3,momentum=0.9,weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=3,min_lr=0.000001)

    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []
    lr_list = []
    best_acc = 0

    for i in range(epoch):
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0


        net.train()

        pbar_trainloader = tqdm(trainloader,file=sys.stdout,desc="epoch:{:0>3}|{:<3}-->{:<5} ".format(i+1,epoch,'train'))
        for step, data in enumerate(pbar_trainloader):
            img, labels = data
            img = img.to(device)
            labels = labels.to(device)

            # 梯度清零、清除缓存
            optimizer.zero_grad()
            if hasattr(torch.cuda,"cuda.cache"):
                torch.cuda.empty_cache()

            logits,Aux_logits1,Aux_logits2 = net(img)
            # 带有辅助分类器的loss函数
            loss1 = criterion(logits,labels)
            loss2 = criterion(logits, labels)
            loss3 = criterion(logits, labels)
            loss = loss1 + loss2 * 0.3 + loss3 * 0.3

            loss.backward()
            optimizer.step()

            train_loss += loss.data
            # 取主分类器的输出来求acc
            train_acc += get_acc(logits,labels)

        train_loss = train_loss / len(trainloader)
        train_acc = train_acc * 100 / len(trainloader)

        lr = optimizer.param_groups[0]['lr']

        train_loss_list.append(train_loss.item())
        train_acc_list.append(train_acc.item())
        lr_list.append(lr)

        scheduler.step(train_loss)

        if testloader is not None:
            net.eval()
            with torch.no_grad():
                pbar_testloader = tqdm(testloader,file=sys.stdout,desc="epoch:{:0>3}|{:<3}-->{:<5}".format(i+1, epoch, 'val'))
                for step, data in enumerate(pbar_testloader):
                    img, labels = data
                    img = img.to(device)
                    labels = labels.to(device)

                    # 梯度清零、清除缓存
                    optimizer.zero_grad()
                    if hasattr(torch.cuda, "cuda.cache"):
                        torch.cuda.empty_cache()

                    output = net(img)
                    loss = criterion(output, labels)

                    test_loss += loss.data
                    test_acc += get_acc(output, labels)

                test_loss = test_loss / len(testloader)
                test_acc = test_acc * 100 / len(testloader)

                test_loss_list.append(test_loss.item())
                test_acc_list.append(test_acc.item())

                # 保存模型
        if test_acc > best_acc:
            torch.save(net,model_savepath)
            best_acc = test_acc
    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['test_acc'] = test_acc_list
    Loss['train_loss'] = train_loss_list
    Loss['test_loss'] = test_loss_list
    Lr = lr_list

    # 将所有结果参数封装成字典，并放进列表，写入txt文件
    if record_result:
        List = []
        write_time = time.asctime(time.localtime())
        Batch_size = {'batch_size':batch_size}
        Epoch = {'epoch':epoch}

        List.append(Batch_size)
        List.append(Epoch)
        List.append(Loss)
        List.append(Acc)
        List.append(Lr)
        file_name = write_result(List,write_time)
        print('The result is wroten in {}'.format(file_name))
    return Loss,Acc,Lr

if __name__ == '__main__':


    batch_size = 8
    epoch = 2
    data_root = 'D:/StudyDatasets/flower_photos/dataset'
    model_path = './model'

    Loss,Acc,Lr = train(batch_size,epoch,data_root,model_path,record_result=True)
    plot_history1(epoch, Acc, Loss, Lr)


