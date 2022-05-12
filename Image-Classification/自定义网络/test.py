import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def test_AccuracyAll(testloader,device,path):

    net = torch.load(path)
    correct = 0  # 定义预测正确的图片数，初始化为0
    total = 0  # 总共参与测试的图片数，也初始化为0
    # testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    for data in testloader:  # 循环每一个batch
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        net.eval()  # 把模型转为test模式
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        outputs = net(images)  # 输入网络进行测试

        # outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)  # 更新测试图片的数量
        correct += (predicted == labels).sum()  # 更新正确分类的图片的数量

    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))


def test_AccuracySim(testloader,device,classes,path):
    net = torch.load(path)
    # 定义2个存储每类中测试正确的个数的 列表，初始化为0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    # testloader = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True, num_workers=2)
    net.eval()
    with torch.no_grad():
        for data in testloader:
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

    for i in range(10):
        print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def imshow(img):
    img = img / 2 + 0.5
    img = np.transpose(img.numpy(),(1,2,0))
    plt.imshow(img)

def SelectToTest(testloader,classes,device,path):
    net = torch.load(path)
    net = net.to(device)
    dataiter = iter(testloader)
    images, labers = dataiter.next()

    images = images.to(device)
    labers = labers.to(device)
    outputs = net(images)
    _, preds = torch.max(outputs, 1)

    correct = torch.sum(preds == labers.data).item()

    preds = preds.cpu()
    labels = labers.cpu()
    images = images.cpu()
    print("Accuracy Rate = {}%".format(correct / len(images) * 100))

    fig = plt.figure(figsize=(25, 25))
    for idx in np.arange(64):
        ax = fig.add_subplot(8, 8, idx + 1, xticks=[], yticks=[])
        # fig.tight_layout()
        #     plt.imshow(im_convert(images[idx]))
        imshow(images[idx])
        ax.set_title("{}, ({})".format(classes[preds[idx].item()], classes[labels[idx].item()]),
                     color=("green" if preds[idx].item() == labels[idx].item() else "red"))

    plt.show()

def LocalTest(model_path,img,classes,device,transform):
    import torch.nn.functional as F
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = transform(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    net = torch.load(model_path)
    net.eval()

    output = net(img)
    prob = F.softmax(output, dim=1)
    print("概率：", prob)
    value, pred = torch.max(output.data, 1)
    print("类别：", pred.item())
    pred_class = classes[pred.item()]
    print('分类：', pred_class)