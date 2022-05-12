import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import tqdm

def get_acc(outputs, label):
    total = outputs.shape[0] # outputs是128张图的10分类结果（还没激活为概率）(128,10),shape[0]其实就是batch_size
    probs, pred_y = outputs.data.max(dim=1) # probs为10类中概率最大的，pred_y为最大概率对应标签，dim=1等价于axis=1；
    correct = (pred_y == label).sum().data # 如果预测的分类正确则让correct加1
    return correct / total # 返回准确率Accuracy

def train(net, trainloader, testloader, epoches, optimizer , criterion, scheduler , path = './model.pth', writer = None ,verbose = False):
    # 配置硬件
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    train_acc_list, test_acc_list = [],[]
    train_loss_list, test_loss_list = [],[]
    loss_list = []
    lr_list  = []
    # 开始进入epoch
    # 虽然为epoch内，但是以下操作都是一个batch一个batch来操作
    for i in range(epoches): # 开始循环epoches次
        start = time.time()  # 系统当前时间
        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0
        if torch.cuda.is_available():
            # 网络放到gpu中
            net = net.to(device)
        # 启用dropout和BN
        net.train()

        # 提取数据并放入gpu
        for step,data in enumerate(trainloader,start=0): # step是batch的个数-1，其实就是iteration的次数
            im,label = data # data一个元素是数据集的数据和标签，data是超2维的list，[0]上是以batch_size为单位的tensor，每个tensor是一个batch的图片(128,3,32,32)；[1]上是标签
            im = im.to(device)
            label = label.to(device)
            print("iteration=",step)
            # 梯度清零 显存clean
            optimizer.zero_grad() # 梯度清零，让这次训练的梯度不受上次的影响 #当然也可以释放内存
            if hasattr(torch.cuda, 'empty_cache'): # 删除显存中的不必要参数，释放显存
                torch.cuda.empty_cache()


            # 前向传播，算y_hat
            outputs = net(im)
            # 计算loss
            loss = criterion(outputs,label)
            # backward（计算梯度）
            loss.backward()
            # 更新参数
            optimizer.step()
            # 更新loss函数值
            train_loss += loss.data

            # probs, pred_y = outputs.data.max(dim=1) # 得到概率
            # # 正确的个数
            # train_acc += (pred_y==label).sum().item()
            # # 总数
            # total += label.size(0)

            # 更新准确率函数
            train_acc += get_acc(outputs,label)
            # 打印训练进度
            rate = (step + 1) / len(trainloader)
            a = "*" * int(rate * 50) # 以50个*作为进度条
            b = "." * (50 - int(rate * 50))
            print('\r train {:3d}|{:3d}{:3.0f}%  [{}->{}] '.format(i+1,epoches,int(rate*100),a,b),end='')
        # 计算一个epoch内的平均loss和准确率
        train_loss = train_loss / len(trainloader)
        train_acc = train_acc * 100 / len(trainloader)
        if verbose:
            train_acc_list.append(train_acc.item()) # 记录准确率
            train_loss_list.append(train_loss.item()) # 记录loss函数值
    #     print('train_loss:{:.6f} train_acc:{:3.2f}%' .format(train_loss ,train_acc),end=' ')
        # 记录学习率
        lr = optimizer.param_groups[0]['lr'] # 从优化器中提取lr
        if verbose:
            lr_list.append(lr)
        # 根据loss来更新学习率
        scheduler.step(train_loss)

        # 浅浅val一下
        if testloader is not None:
            net.eval() # 如果要测试，就切换成eval模式
            with torch.no_grad():
                for step,data in enumerate(testloader,start=0):
                    im,label = data
                    im = im.to(device)
                    label = label.to(device)
                    # 释放显存
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    outputs = net(im)
                    loss = criterion(outputs,label) # 因为val过程中也要计算loss，所以测试集里面也要有标签
                    test_loss += loss.data
                    # probs, pred_y = outputs.data.max(dim=1) # 得到概率
                    # test_acc += (pred_y==label).sum().item()
                    # total += label.size(0)
                    test_acc += get_acc(outputs,label)
                    rate = (step + 1) / len(testloader)
                    a = "*" * int(rate * 50)
                    b = "." * (50 - int(rate * 50))
                    print('\r test  {:3d}|{:3d} {:^3.0f}%  [{}->{}] '.format(i+1,epoches,int(rate*100),a,b),end='')
            test_loss = test_loss / len(testloader)
            test_acc = test_acc * 100 / len(testloader)
            if verbose:
                test_loss_list.append(test_loss.item())
                test_acc_list.append(test_acc.item())
            end = time.time()
            print(
                '\rEpoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Test Loss:{:>.6f}  Test Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                    i + 1, epoches, train_loss, train_acc, test_loss, test_acc,lr), end='')
        else:
            end = time.time()
            print('\rEpoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(i+1,epoches,train_loss,train_acc,lr),end = '')
        time_ = int(end - start) # 训练加上测试的运行时间（单位s）
        h = time_ / 3600
        m = time_ % 3600 /60
        s = time_ % 60
        time_str = "\tTime %02d:%02d" % ( m, s)
        # ====================== 使用 tensorboard ==================
        # 画出三个标量图，放在tensorboard中
        if writer is not None:
            writer.add_scalars('Loss', {'train': train_loss,
                                    'valid': test_loss}, i+1)
            writer.add_scalars('Acc', {'train': train_acc ,
                                   'valid': test_acc}, i+1)
            writer.add_scalars('Learning Rate',lr,i+1)
        # =========================================================
        # 打印所用时间
        print(time_str)
        # 如果取得更好的准确率，就保存模型
        if test_acc > best_acc:
            torch.save(net,path)
            best_acc = test_acc
    Acc = {}
    Loss = {}
    Acc['train_acc'] = train_acc_list
    Acc['test_acc'] = test_acc_list
    Loss['train_loss'] = train_loss_list
    Loss['test_loss'] = test_loss_list
    Loss['loss'] = loss_list
    Lr = lr_list
    return Acc, Loss, Lr

def plot_history(epoches, Acc, Loss, lr):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    plt.figure('loss')
    epoch_list = range(1,epoches + 1)
    plt.plot(epoch_list, Loss['train_loss'])
    plt.plot(epoch_list, Loss['test_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss Value')
    plt.legend(['train', 'test'], loc='upper left')

    plt.figure()
    plt.plot(epoch_list, Acc['train_acc'])
    plt.plot(epoch_list, Acc['test_acc'])
    plt.xlabel('epoch')
    plt.ylabel('Acc Value')
    plt.legend(['train', 'test'], loc='upper left')

    plt.figure()
    plt.plot(epoch_list, lr)
    plt.xlabel('epoch')
    plt.ylabel('Train LR')
    plt.show()


