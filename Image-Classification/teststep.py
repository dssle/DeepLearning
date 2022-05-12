import numpy as np
import torch

# a_1 = [[1,2,3,4],
#        [5,6,7,8]]
# a_2 = np.array([[1,2,3,4],
#        [5,6,7,8]])
# a_3 = torch.tensor([[1,2,3,4],
#        [5,6,7,8]])
# print()

# a = [1,2,3,4,5,6]
# # for i in iter(a):
# #     print(i)
# b = iter(a).next()
# print(b)

# a=[1,2,3]
# it=iter(a) #创建迭代器对象
# next(it)   #输出迭代器下一项
# next(it)
# next(it)
# #输出：
# #1
# #2
# #3

# import torch
# device = torch.device('cuda')
# print(type(device))
# print(device)

import torch
import torch.nn as nn
import torchsummary as ts
#
# pred_y = [1,2,3,4]
# label = [1,2,2,4]
# correct = (pred_y == label).sum().data
# print(correct)

# for i in range(20):
#     print(f'{i} \n')

# import time
# print((time.time()))
# m = 10
# s = 5
# time_str = "\tTime %02d:%02d" % ( m, s)
# print(time_str)
#
# Acc = {}
# Loss = {}
# train_acc_list = 10
# Acc['train_acc'] = train_acc_list
# print(Acc)

# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
#
# writer = SummaryWriter()
# a = []
# for x in range(20):
#     a.append(x)
# a = np.array(a)
#
# for x in range(20):
#     writer.add_scalar('lalala',a,x)

# import numpy as np
# from torch.utils.tensorboard import SummaryWriter  # 也可以使用 tensorboardX
# # from tensorboardX import SummaryWriter  # 也可以使用 pytorch 集成的 tensorboard
#
# writer = SummaryWriter()
# for epoch in range(100):
#     writer.add_scalar('add_scalar/squared', np.square(epoch), epoch)
# #     writer.add_scalars("add_scalars/trigonometric",
# # {'xsinx': epoch * np.sin(epoch/5), 'xcosx': epoch* np.cos(epoch/5), 'xtanx': np.tan(epoch/5)}, epoch)
#
# writer.close()

# class ok :
#     def sum(self,aa,bb):
#         return aa+bb
#     yayayaya = 1234
# a = [1,2,3,4,5]
# b = {
#     "yes": 1234,
# }
# c = (1,2,3)
# d = ok()
# pass

import time
# from tqdm import tqdm
# pbar = tqdm(range(10))
# for a in pbar:
#     pbar.set_description("开始执行")
#     time.sleep(1)
#     pass

# from tqdm import tqdm
# for a in tqdm(range(10)):
#     time.sleep(1)
#     pass


# class_correct = list(0. for i in range(10))
# print(class_correct)
# import torch
# import numpy as np
# # class_correct.squeeze()
# predicted = torch.tensor([1,2,2,4,45,65,7,32,54,634563])
# labels =torch.tensor([1,2,3,4,5,46,67,47,6,8])
# a = [0,0]
# c = (predicted == labels).squeeze()
# d = (predicted == labels)
# a[0] += c[1]
# print(c,type(c),c[1])
# print(d,type(d),d[1])
# print(a)
#
#
# a = [0,0]
# b = torch.tensor([True,False])
# a[0] += b[0]
# a[1] += b[1]
# print(a)
# # 一个tensor里面有True/False布尔型，取一个与数相加，则True相当于1，False相当于0，返回相加结果，类型为tensor
# b = torch.tensor([True,False])
# c = 1
# c +=b[0]
# print(c)

# import torch
# a = torch.tensor([1,2,3,4,5,6])
# b = torch.tensor([1,2,4,3])
# c = torch.sum(a == b)
# d = (a==b).sum()
# print(c)
# print(d)
#
# import matplotlib.pyplot as plt
# plt.figure("图片名称")

# print(a.view(2,3))
# i =10
# print(str(i))


# a = torch.tensor([1])
# b = torch.tensor([2])
# a += b
# print(a)
# import tqdm
# i = 1
# epoch = 9
# # print(f'epoch:{:0>3}|{:<3}-->{}'.format(i,epoch+1,'train'))
# print('yes')
#
# for a in tqdm.tqdm(range(9)):
#     tqdm.tqdm.write()

# import os
# cwd = "./model11111111"
# if not os.path.exists(cwd):
#     os.makedirs(cwd)
# #
# data_root = os.path.join(cwd, "dataset")
# os.makedirs(data_root)
# print(data_root)

# a = (.8254 * 63 + .8315 * 89 +.4688 * 64 + .8841 * 69 + .7595 *79 ) / 364
# print(a)

# A = 128
# B = 1
# def tran(A,B):
#     while B < A :
#         A /= 2
#         B *= 2
#     return A,B
# a = []
# for _ in range(1,1000):
#     a.append(_)
# print(a)
# c,*b = a
# print(b)

# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.current_device())


# def sum(a,b,**kwargs):
#     kwargs
#     print(kwargs,type(kwargs))
#     return a+b
#
# c = sum(10,20,yes = "yes",ok = 1)

# from pathlib import Path
# import glob
# def increment_dir(dir, comment=''):
#     # Increments a directory runs/exp1 --> runs/exp2_comment
#     n = 0  # number
#     dir = str(Path(dir))  # os-agnostic
#     d = sorted(glob.glob(dir + '*'))  # directories
#     if len(d):
#         update_NUM = 0
#         for x in d:
#             if '_' in x:
#                 tmp_NUM = int(x[len(dir):])
#                 if tmp_NUM > update_NUM:
#                     update_NUM = tmp_NUM
#         n = update_NUM + 1
#     return dir + str(n) + ('_' + comment if comment else '')
#
# increment_dir('./VGG/results/yes')
# import os
# def write_result(list):
#     dir = './'
#     file_No = 1
#     while(True):
#         file_name = "{}result{}.txt".format(dir, file_No)
#         if not os.path.exists(file_name):
#             with open(file_name,'w') as file:
#                 list = str(list)
#                 file.write(list)
#                 break
#         else:
#             file_No += 1
# list = [1,2,3]
# write_result(list)

import time

write_time = time.asctime(time.localtime())
write_time2 = time.asctime()
# write_time3 = time.asctime(time.time())
# print(write_time,'\n',type(write_time2))

# List = ['yes']
# file = open('./result1.txt','a')
# file.write(str(List))
# file.write('\n')
# file.write(str([write_time]))
# file.close()
#
file = open('./result1.txt','r')
# for file1 in file.readline():
# file1 = file.readline()
file1 = file.readlines()
# print(eval(file1[0]))
for i in file1:
    file2 = eval(i)
    print(file2)
print(file1)
print('1')
# print(type(file2))
# print(file2)
pass

# Str = file.readlines()
# for line in Str:
#     List = eval(line)