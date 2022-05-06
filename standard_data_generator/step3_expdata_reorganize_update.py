import os
import numpy as np
import random
import pickle

expdata_dir = f'D:/project/seg_from_zhen/expdata/ctseg'
img_dir = os.path.join(expdata_dir, 'img')
mask1_dir = os.path.join(expdata_dir, 'mask1')
mask2_dir = os.path.join(expdata_dir, 'mask2')

n_case = len(os.listdir(img_dir))
obj_list = np.arange(n_case)
random.shuffle(obj_list)

sel_topk = 1000
alldata_list = obj_list[:sel_topk]

data1_list = alldata_list[:int(0.5*len(alldata_list))]
data2_list = alldata_list[int(0.5*len(alldata_list)):]

print (len(alldata_list), len(data1_list), len(data2_list))

data1_n_train = int(len(data1_list) * 0.6)
data1_n_valid = int(len(data1_list) * 0.8 )
data1_train_list = data1_list[:data1_n_train]
data1_valid_list = data1_list[data1_n_train:data1_n_valid]
data1_test_list = data1_list[data1_n_valid:]

data2_n_train = int(len(data2_list) * 0.6)
data2_n_valid = int(len(data2_list) * 0.8)
data2_train_list = data2_list[:data2_n_train]
data2_valid_list = data2_list[data2_n_train:data2_n_valid]
data2_test_list = data2_list[data2_n_valid:]
#
# # construct experiment dataset
# task1_trainlist = []
# task2_trainlist = []
# task3_trainlist = []
#
# task1_validlist = []
# task2_validlist = []
# task3_validlist = []
#
# task1_testlist = []
# task2_testlist = []
# task3_testlist = []
#
# for item in data1_train_list:
#     img_path = os.path.join(img_dir, '%d.npy'%item)
#     mask1_path = os.path.join(mask1_dir, '%d.npy'%item)
#     task1_trainlist.append((img_path, mask1_path))
#
# for item in data2_train_list:
#     img_path = os.path.join(img_dir, '%d.npy'%item)
#     mask2_path = os.path.join(mask2_dir, '%d.npy'%item)
#     task2_trainlist.append((img_path, mask2_path))
#
# for item in data1_valid_list:
#     img_path = os.path.join(img_dir, '%d.npy'%item)
#     mask1_path = os.path.join(mask1_dir, '%d.npy'%item)
#     task1_validlist.append((img_path, mask1_path))
#
# for item in data2_valid_list:
#     img_path = os.path.join(img_dir, '%d.npy'%item)
#     mask2_path = os.path.join(mask2_dir, '%d.npy'%item)
#     task2_validlist.append((img_path, mask2_path))
#
# for item in data1_test_list:
#     img_path = os.path.join(img_dir, '%d.npy'%item)
#     mask1_path = os.path.join(mask1_dir, '%d.npy'%item)
#     task1_testlist.append((img_path, mask1_path))
#
# for item in data2_test_list:
#     img_path = os.path.join(img_dir, '%d.npy'%item)
#     mask2_path = os.path.join(mask2_dir, '%d.npy'%item)
#     task2_testlist.append((img_path, mask2_path))
#
# task3_trainlist = task1_trainlist + task2_trainlist
# task3_validlist = task1_validlist + task2_validlist
# task3_testlist = task1_testlist + task2_testlist
#
# modelData_dir = 'C:/Users/zhi.qiao/PycharmProjects/pythonProject/dataPreprocess/ctseg_data/n1'
# pickle.dump(task1_trainlist, open(os.path.join(modelData_dir,'task1_train_list.pkl'), 'wb'))
# pickle.dump(task1_validlist, open(os.path.join(modelData_dir,'task1_valid_list.pkl'), 'wb'))
# pickle.dump(task1_testlist, open(os.path.join(modelData_dir,'task1_test_list.pkl'), 'wb'))
# print (len(task1_trainlist), len(task1_validlist), len(task1_testlist))
# pickle.dump(task2_trainlist, open(os.path.join(modelData_dir,'task2_train_list.pkl'), 'wb'))
# pickle.dump(task2_validlist, open(os.path.join(modelData_dir,'task2_valid_list.pkl'), 'wb'))
# pickle.dump(task2_testlist, open(os.path.join(modelData_dir,'task2_test_list.pkl'), 'wb'))
# print (len(task2_trainlist), len(task2_validlist), len(task2_testlist))
# pickle.dump(task3_trainlist, open(os.path.join(modelData_dir,'task3_train_list.pkl'), 'wb'))
# pickle.dump(task3_validlist, open(os.path.join(modelData_dir,'task3_valid_list.pkl'), 'wb'))
# pickle.dump(task3_testlist, open(os.path.join(modelData_dir,'task3_test_list.pkl'), 'wb'))
# print (len(task3_trainlist), len(task3_validlist), len(task3_testlist))