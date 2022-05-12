from resnet_train import train
from resnet_utils import plot_history
from resnet_test import  AccuracyAll,AccuracySim,ShowPart

if __name__  ==  '__main__':
    batch_size = 32
    epoch = 2
    data_root = 'D:/StudyDatasets/flower_photos/dataset'
    model_path = './model'

    Loss,Acc,Lr = train(batch_size,epoch,data_root,model_path)
    # plot_history(epoch,Acc,Loss,Lr)

    AccuracyAll(batch_size,data_root,model_path)

    classes_num = 5
    AccuracySim(batch_size,classes_num,data_root,model_path)

    # batch_no = 2
    # ShowPart(batch_no,batch_size,data_root,model_path)