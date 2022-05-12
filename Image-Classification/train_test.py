import torch
import tqdm as tqdm
import matplotlib.pyplot as plt

def get_acc(outputs,labels):
     total = outputs.shape[0]
     probs,pred = torch.max(outputs.data,1)
     acc = (pred == labels).sum()
     return acc/total


def train(net, trainloader, testloader, epoch, optimizer, criterion, schedule, path, verbose=False):
    device =  "cuda"  if torch.cuda.is_available() else 'cpu'
    # 创建空列表，保存每个epoch的数据：train的loss、acc、lr，val的loss、acc
    train_loss_list ,test_loss_list= [], []
    train_acc_list, test_acc_list = [], []
    lr_list = []

    # 开始进入epoch，0~epoch-1训练
    for i in range(epoch):
        # 定义将要保存的数据
        train_loss = 0
        test_loss = 0
        test_acc = 0
        best_acc = 0
        lr = 0

        # 加载模型
        net = torch.load(path).to(device)

        net = net.train()

        # 设置进度条
        trainloader = tqdm.tqdm(trainloader)
        trainloader.set_description("epoch"+str(i+1) +'|'+ str(epoch))
        # 以batch为单位读取数据
        for step,data in enumerate(trainloader):
            img, label = data
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            # 清除缓存
            if hasattr(torch.cuda,"empty_cache"):
                torch.cuda.empty_cache()

            outputs = net(img)

            loss = criterion(outputs,label)

            loss.backward()

            optimizer.step()

            # 计算一个epoch的平均loss和acc
            train_loss += loss.data
            test_acc += get_acc(outputs,label)

            train_loss = train_loss/len(trainloader)
            test_acc = test_acc/len(trainloader)

        # 记录每个epoch的平均loss和acc
        if verbose:
            train_loss_list.append(train_loss.item())
            train_acc_list.append(test_acc.item())

            lr = optimizer.param_groups[0]['lr']
            lr_list.append(lr)

        # 更新优化器
        schedule.step(train_loss)


        # val
        if testloader is not None:
            net.eval()
            with torch.no_grad():

                for data in testloader:
                    img, label = data
                    img = img.to(device)
                    label = label.to(device)
                    if hasattr(torch.cuda,'cuda.cache'):
                        torch.cuda.empty_cache()

                    outputs = net(img)

                    loss = criterion(outputs)

                    test_loss += loss.data
                    test_acc += get_acc(outputs,label)

            test_loss = test_loss / len(testloader)
            test_acc = test_acc / len(testloader)

            if verbose:
                test_loss_list.append(train_loss.item())
                test_acc_list.append(test_acc)

            if test_acc > best_acc:
                torch.save(net,path)
                best_acc = test_acc

    # 把数据列表放入字典中
    Acc = {}
    Loss = {}
    Acc['acc_train'] = train_acc_list
    Acc['acc_test'] = test_acc_list
    Loss['loss_train'] = train_loss_list
    Loss['loss_test'] = test_loss_list
    Lr = lr_list
    return Acc, Loss, Lr

def plot_history(Loss, Acc, Lr, epochs):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    plt.figure()
    epoch_list = range(epochs+1)
    plt.plot(epoch_list, Loss['loss_train'])
    plt.plot(epoch_list, Loss['loss_test'],)
    plt.xlabel('epoch')
    plt.ylabel('Loss Value')
    plt.legend(['train','test'], loc = "upper left")

    plt.figure()
    plt.plot(epoch_list, Acc['acc_train'])
    plt.plot(epoch_list, Acc['acc_test'])
    plt.xlabel('epoch')
    plt.ylabel('Acc Value')
    plt.legend(['train','test'], loc = "upper left")

    plt.figure()
    plt.plot(epoch_list, Lr)
    plt.xlabel('epoch')
    plt.ylabel('Lr Value')
    plt.legend(['lr'], loc = "upper left")

    plt.show()












