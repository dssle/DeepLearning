from VGG_train import train
from VGG_utils import plot_history1,plot_history2,cal_time,write_result
from VGG_test import  AccuracyAll,AccuracySim,ShowPart
import time

if __name__  ==  '__main__':
    batch_size = 8
    epoch = 1
    data_root = 'D:/StudyDatasets/flower_photos/dataset'
    model_path = './model'

    start_time = time.time()

    Loss,Acc,Lr = train(batch_size,epoch,data_root,model_path,record_result=True)
    plot_history2()
    #
    # AccuracyAll(batch_size,data_root,model_path)
    #
    # classes_num = 5
    # AccuracySim(batch_size,classes_num,data_root,model_path)


    end_time = time.time()
    cal_time(start_time,end_time)





    # batch_no = 2
    # ShowPart(batch_no,batch_size,data_root,model_path)