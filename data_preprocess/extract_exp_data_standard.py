import pickle
import numpy as np
import random
import os
import csv

case_list = []
z0 = 0
formal_covid_dict = pickle.load(open('./data_preprocess/formal_covid_dict.pkl','rb'))
for key, value in formal_covid_dict.items():
  for image_name, info in value['image_dict'].items():
    if 'PA' in info['type'] or 'AP' in info['type']:
      if value['class']['COVID-19'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 0))
      if value['class']['pneumonia_virus'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 1))
      if value['class']['pneumonia_bacteria'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 2))
      if value['class']['normal'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 3))
      z0 += 1
print(len(case_list))

z1 = 0
formal_kaggle_dict = pickle.load(open('./data_preprocess/formal_kaggle_dict.pkl','rb'))
for key, value in formal_kaggle_dict.items():
  for image_name, info in value['image_dict'].items():
    if 'PA' in info['type'] or 'AP' in info['type']:
      if value['class']['COVID-19'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 0))
      if value['class']['pneumonia_virus'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 1))
      if value['class']['pneumonia_bacteria'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 2))
      if value['class']['normal'] == 1:
          case_list.append((info['path'], key+'_'+image_name, 3))
      z1 += 1
print (len(case_list))
print (z0, z1)
random.shuffle(case_list)

np = len(case_list)
p1 = int(0.7*np)
p2 = int(0.8*np)

train_list = case_list[:p1]
valid_list = case_list[p1:p2]
test_list = case_list[p2:]

random.shuffle(train_list)
random.shuffle(valid_list)
random.shuffle(test_list)

exp_data_id = 'standard_data_multiclass_0325'
exp_data_dir = os.path.join('./data_preprocess', exp_data_id)
os.mkdir(exp_data_dir)

train_s = [0,0,0,0]
test_s = [0,0,0,0]
valid_s = [0,0,0,0]
for x in train_list:
  train_s[x[2]] += 1
for x in valid_list:
  valid_s[x[2]] += 1
for x in test_list:
  test_s[x[2]] += 1
print (train_s)
print ('N of Train', len(train_list), 'covid:%d'%train_s[0], 'pneumonia_virus:%d'%train_s[1], 'pneumonia_bacteria:%d'%train_s[2], 'normal:%d'%train_s[3])
print ('N of Valid', len(valid_list), 'covid:%d'%valid_s[0], 'pneumonia_virus:%d'%valid_s[1], 'pneumonia_bacteria:%d'%valid_s[2], 'normal:%d'%valid_s[3])
print ('N of Test', len(test_list), 'covid:%d'%test_s[0], 'pneumonia_virus:%d'%test_s[1], 'pneumonia_bacteria:%d'%test_s[2], 'normal:%d'%test_s[3])

with open(os.path.join(exp_data_dir, 'data_statistic.csv'),'w') as f:
  csv_writer = csv.writer(f)
  csv_writer.writerow(['N of Train', len(train_list), 'covid:%d'%train_s[0], 'pneumonia_virus:%d'%train_s[1], 'pneumonia_bacteria:%d'%train_s[2], 'normal:%d'%train_s[3]])
  csv_writer.writerow(['N of Valid', len(valid_list), 'covid:%d'%valid_s[0], 'pneumonia_virus:%d'%valid_s[1], 'pneumonia_bacteria:%d'%valid_s[2], 'normal:%d'%valid_s[3]])
  csv_writer.writerow(['N of Test', len(test_list), 'covid:%d'%test_s[0], 'pneumonia_virus:%d'%test_s[1], 'pneumonia_bacteria:%d'%test_s[2], 'normal:%d'%test_s[3]])

train_path = os.path.join(exp_data_dir, 'exp_train_list.pkl')
valid_path = os.path.join(exp_data_dir, 'exp_valid_list.pkl')
test_path = os.path.join(exp_data_dir, 'exp_test_list.pkl')

if os.path.exists(train_path):
  os.remove(train_path)

if os.path.exists(valid_path):
  os.remove(valid_path)

if os.path.exists(test_path):
  os.remove(test_path)

pickle.dump(train_list, open(train_path,'wb'))
pickle.dump(valid_list, open(valid_path,'wb'))
pickle.dump(test_list, open(test_path,'wb'))

print ('finished')