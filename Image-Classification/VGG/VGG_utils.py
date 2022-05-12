import os.path

import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5
    img = np.transpose(img.numpy(),(1,2,0))
    plt.imshow(img)

def get_acc(outputs, label):
    total = outputs.shape[0] # outputs是128张图的10分类结果（还没激活为概率）(128,10),shape[0]其实就是batch_size
    probs, pred_y = outputs.data.max(dim=1) # probs为10类中概率最大的，pred_y为最大概率对应标签，dim=1等价于axis=1；
    correct = (pred_y == label).sum().data # 如果预测的分类正确则让correct加1
    return correct / total # 返回准确率Accuracy

# 画图直接版本，直接从train的输出取结果输入
def plot_history1(epoches, Acc, Loss, lr):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'


    plt.figure('loss')
    epoch_list = range(1,epoches + 1)
    # print('epoch_list:',epoch_list)
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

# 画图简介版，先将结果保存至txt文件，再从txt中提取结果输入至画图函数，程序结束后还能画
def plot_history2():
    with open("./results/result.txt","r") as file:
        Str = file.read()
        List = eval(Str)

        epoches = List[1]
        Loss = List[2]
        Acc = List[3]
        lr = List[4]


    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'


    plt.figure('loss')
    epoch_list = range(1,epoches['epoch'] + 1)
    # print('epoch_list:',epoch_list)
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

# 将一个数分为两个相近的数相乘
def tran(num_A):
    num_B = 1
    while num_B < num_A :
        num_A /= 2
        num_B *= 2
    return int(num_A),int(num_B)

# 计算程序执行时间
def cal_time(start_time,end_time):
    dur_time = end_time - start_time
    hours = dur_time // 3600
    mins = dur_time % 3600 // 60
    second = dur_time % 3600 % 60
    print('Program is completed, using {:.1f}h {:.1f}m {:.1f}s'.format(hours,mins,second))

# 按顺序生成txt文件，并将传入的list写入txt文件，注意如果中间有txt被删除则在中间生成，如1.txt，3.txt，那么会生成2.txt而不是4.txt
def write_result(list):
    dir = './results/'
    file_No = 1
    while(True):
        file_name = "{}result{}.txt".format(dir, file_No)
        if not os.path.exists(file_name):
            with open(file_name,'w') as file:
                list = str(list)
                file.write(list)
                break
        else:
            file_No += 1
