import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from utils_AlexNet import tran
import os
import PIL.Image as Image
import torch.nn.functional as F

# 整个测试集的准确率，batch_size用于配置数据集，data_root数据集路径，model_path模型所在文件夹（没到模型路径）
def AccuracyAll(batch_size,data_root,model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = datasets.ImageFolder(root=os.path.join(data_root, 'val'), transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    model_savepath = os.path.join(model_path,'Mynet.pth')

    net = torch.load(model_savepath)
    correct = 0  # 定义预测正确的图片数，初始化为0
    total = 0  # 总共参与测试的图片数，也初始化为0

    for step, data in enumerate(testloader):  # 循环每一个batch
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        net.eval()  # 把模型转为test模式
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        outputs = net(images)  # 输入网络进行测试

        # outputs.data是一个batch_size * classes的张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)  # 更新测试图片的数量
        correct += (predicted == labels).sum()  # 更新正确分类的图片的数量
    print('Accuracy of the network on the %g test images: %.2f %%' % (total,100 * correct / total))


# 整个测试集每一类的准确率，class_num是数据集标签种类数量，batch_size用于配置数据集，data_root数据集路径，model_path模型所在文件夹（没到模型路径）
def AccuracySim(batch_size,classes_num,data_root,model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = datasets.ImageFolder(root=os.path.join(data_root, 'val'), transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    classes = testset.classes
    model_savepath = os.path.join(model_path,'Mynet.pth')

    net = torch.load(model_savepath)
    # 定义2个存储每类中测试正确的个数的 列表，初始化为0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    net.eval()
    with torch.no_grad():
        for step, data in enumerate(testloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            # 4组(batch_size)数据中，输出于label相同的，标记为1，否则为0
            c = (predicted == labels)
            for i in range(len(images)):  # 因为每个batch都有4张图片，所以还需要一个4的小循环
                label = labels[i]  # 对各个类的进行各自累加
                class_correct[label.data] += c[i]
                class_total[label] += 1

    for i in range(classes_num):
        print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# 选取指定的batch来查看每张图的预测情况，batch_No确定要看哪个batch，batch_size用于配置数据集，data_root数据集路径，model_path模型所在文件夹（没到模型路径）
def ShowPart(batch_No,batch_size,data_root,model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = datasets.ImageFolder(root=os.path.join(data_root, 'val'), transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    classes = testset.classes

    model_savepath = os.path.join(model_path,'Mynet.pth')

    net = torch.load(model_savepath)
    correct = 0  # 定义预测正确的图片数，初始化为0
    total = 0  # 总共参与测试的图片数，也初始化为0

    for step,data in enumerate(testloader):
        if step + 1 == batch_No:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            net.eval()  # 把模型转为test模式
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            outputs = net(images)  # 输入网络进行测试

            # outputs.data是一个batch_size * classes的张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # 更新测试图片的数量
            correct += (predicted == labels).sum()  # 更新正确分类的图片的数量
    print('Accuracy of the network on the No.%g batch with %g test images: %.2f %%' % (batch_No,total,100 * correct / total))

    predicted = predicted.cpu()
    labels = labels.cpu()
    images = images.cpu()

    fig = plt.figure(figsize=(16, 16))
    for idx in range(batch_size):
        # tran函数用于将batch_size分为相近两数相乘，画图的时候可以有合适的长宽比
        a, b = tran(batch_size)
        # 划定b * a的网格
        ax = fig.add_subplot(a, b, idx + 1, xticks=[], yticks=[])

        # 处理一下图像RGB值并从tensor形式转换为numpy形式
        img = images[idx] / 2 + 0.5
        img = np.transpose(img.numpy(), (1, 2, 0))
        plt.imshow(img)
        # imshow(images[idx])

        # 每个网格图片取标题
        ax.set_title("{}, ({})".format(classes[predicted[idx].item()], classes[labels[idx].item()]),
                     color=("green" if predicted[idx].item() == labels[idx].item() else "red"),fontsize=10)

    plt.show()

# 输入本地图片进行判断，classes是标签的类别（列表），data_root数据集路径，model_path模型所在文件夹（没到模型路径）
def LocalTest(classes,img_path,model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img = Image.open(img_path).convert('RGB')
    img.show()
    img = data_transform(img).to(device)
    img = img.unsqueeze(0)

    model_savepath = os.path.join(model_path, 'Mynet.pth')
    net = torch.load(model_savepath).to(device)
    net.eval()
    output = net(img)

    prob = F.softmax(output, dim=1)

    prob =prob.cpu()
    prob = prob.detach().numpy()
    # 取消科学计数法输出，改用平常的小数
    np.set_printoptions(suppress=True)

    print("概率：", prob)
    _, pred = torch.max(output.data, 1)
    print("类别：", pred.item())
    pred_class = classes[pred.item()]
    print('分类：', pred_class)




