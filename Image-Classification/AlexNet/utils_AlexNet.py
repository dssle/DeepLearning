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

def plot_history(epoches, Acc, Loss, lr):
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

# 将一个数分为两个相近的数相乘
def tran(num_A):
    num_B = 1
    while num_B < num_A :
        num_A /= 2
        num_B *= 2
    return int(num_A),int(num_B)