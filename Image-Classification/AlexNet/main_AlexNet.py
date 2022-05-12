from train_AlexNet import train
from utils_AlexNet import plot_history
from test_AlexNet import AccuracyAll,AccuracySim,ShowPart,LocalTest
import os
if __name__ == '__main__':
    batch_size = 32
    epoch = 30
    data_root = 'D:/StudyDatasets/flower_photos/dataset'
    model_path = './model/'


    ## train
    # Loss,Acc,Lr = train(batch_size,epoch,data_root,model_path)
    # print('train_loss:',Loss['train_loss'],'\n\r','test_loss:',Loss['test_loss'])
    # print('train_acc:',Acc['train_acc'],'\n\r','test_acc:',Acc['test_acc'])
    # plot_history(epoch, Acc, Loss, Lr)

    ## test

    # AccuracyAll(batch_size,data_root,model_path)

    # classes_num = 5
    # AccuracySim(batch_size,classes_num,data_root,model_path)

    # batch_No = 1
    # ShowPart(batch_No,batch_size,data_root,model_path)

    img_path = './data/3.jfif'
    classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    LocalTest(classes,img_path,model_path)