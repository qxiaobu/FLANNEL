import pickle
import numpy as np
import random
import os
import csv

positive_case_list = []
negative_case_list_covid = []
negative_case_list_xray = []

formal_covid_dict = pickle.load(open('./data_preprocess/formal_covid_dict.pkl','rb'))

for key, value in formal_covid_dict.items():
  for image_name, info in value['image_dict'].items():
    if 'AP' or 'PA' in info['type']:
      if value['class']['COVID-19'] == '1':
          positive_case_list.append((info['path'], key+'_'+image_name, 1))
      else:
          negative_case_list_covid.append((info['path'], key+'_'+image_name, 0))
      
formal_xray_dict = pickle.load(open('./data_preprocess/formal_xray_dict.pkl','rb'))
for key, value in formal_xray_dict.items():
  for image_name, info in value['image_dict'].items():
    if 'AP' or 'PA' in info['type']:
      negative_case_list_xray.append((info['path'], key+'_'+image_name, 0))

print (len(positive_case_list))
print (len(negative_case_list_covid))
print (len(negative_case_list_xray))

random.shuffle(negative_case_list_xray)

negative_case_list = negative_case_list_covid + negative_case_list_xray[:3000]

random.shuffle(positive_case_list)
random.shuffle(negative_case_list)

np = len(positive_case_list)
nn = len(negative_case_list)

p1 = int(0.5*np)
p2 = int(0.75*np)

n1 = int(0.5*nn)
n2 = int(0.75*nn)

valid_p_list_standard = positive_case_list[p1:p2]
test_p_list_standard = positive_case_list[p2:]

train_p_list = []
p_list = []
for case in positive_case_list[:p1]:
  p_list.append(case+('original',))
  p_list.append(case+('fz_horizontal',))
  p_list.append(case+('fz_vertical',))
  p_list.append(case+('random_crop1',))
  p_list.append(case+('random_crop2',))
  p_list.append(case+('scale_0.5',))
  p_list.append(case+('scale_2',))
  p_list.append(case+('gaussian_0_1',))
  p_list.append(case+('gaussian_05_1',))
  p_list.append(case+('gaussian_50_1',))
train_p_list = p_list

valid_p_list = []
p_list = []
for case in positive_case_list[p1:p2]:
  p_list.append(case+('original',))
  p_list.append(case+('fz_horizontal',))
  p_list.append(case+('fz_vertical',))
  p_list.append(case+('random_crop1',))
  p_list.append(case+('random_crop2',))
  p_list.append(case+('scale_0.5',))
  p_list.append(case+('scale_2',))
  p_list.append(case+('gaussian_0_1',))
  p_list.append(case+('gaussian_05_1',))
  p_list.append(case+('gaussian_50_1',))
valid_p_list = p_list
  
test_p_list = []
p_list = []
for case in positive_case_list[p2:]:
  p_list.append(case+('original',))
  p_list.append(case+('fz_horizontal',))
  p_list.append(case+('fz_vertical',))
  p_list.append(case+('random_crop1',))
  p_list.append(case+('random_crop2',))
  p_list.append(case+('scale_0.5',))
  p_list.append(case+('scale_2',))
  p_list.append(case+('gaussian_0_1',))
  p_list.append(case+('gaussian_05_1',))
  p_list.append(case+('gaussian_50_1',))
test_p_list = p_list

train_n_list = negative_case_list[:n1]
valid_n_list = negative_case_list[n1:n2]
test_n_list = negative_case_list[n2:]

train_list = train_p_list + train_n_list
valid_list = valid_p_list + valid_n_list
test_list = test_p_list + test_n_list
valid_list_standard = valid_p_list_standard + valid_n_list
test_list_standard = test_p_list_standard + test_n_list

random.shuffle(train_list)

exp_data_id = 'standard_data_augmentation_0405'
exp_data_dir = os.path.join('./data_preprocess', exp_data_id)
os.mkdir(exp_data_dir)

#print ('N of Train', len(train_list), 'N of Positive', len(train_p_list), 'N of Negative', len(train_n_list))
#print ('N of Valid', len(valid_list), 'N of Positive', len(valid_p_list), 'N of Negative', len(valid_n_list))
#print ('N of Test', len(test_list), 'N of Positive', len(test_p_list), 'N of Negative', len(test_n_list))
#
with open(os.path.join(exp_data_dir, 'data_statistic.csv'),'w') as f:
  csv_writer = csv.writer(f)
  csv_writer.writerow(['N of Train', len(train_list), 'N of Positive', len(train_p_list), 'N of Negative', len(train_n_list)])
  csv_writer.writerow(['N of Valid', len(valid_list), 'N of Positive', len(valid_p_list), 'N of Negative', len(valid_n_list)])
  csv_writer.writerow(['N of Test', len(test_list), 'N of Positive', len(test_p_list), 'N of Negative', len(test_n_list)])
  csv_writer.writerow(['N of Valid Standard', len(valid_list_standard), 'N of Positive', len(valid_p_list_standard), 'N of Negative', len(valid_n_list)])
  csv_writer.writerow(['N of Test Standard', len(test_list_standard), 'N of Positive', len(test_p_list_standard), 'N of Negative', len(test_n_list)])
  

train_path = os.path.join(exp_data_dir, 'exp_train_list.pkl')
valid_path = os.path.join(exp_data_dir, 'exp_valid_list.pkl')
test_path = os.path.join(exp_data_dir, 'exp_test_list.pkl')
valid_path_standard = os.path.join(exp_data_dir, 'exp_valid_list_standard.pkl')
test_path_standard = os.path.join(exp_data_dir, 'exp_test_list_standard.pkl')

if os.path.exists(train_path):
  os.remove(train_path)

if os.path.exists(valid_path):
  os.remove(valid_path)

if os.path.exists(test_path):
  os.remove(test_path)

if os.path.exists(valid_path_standard):
  os.remove(valid_path_standard)

if os.path.exists(test_path_standard):
  os.remove(test_path_standard)

pickle.dump(train_list, open(train_path,'wb'))
pickle.dump(valid_list, open(valid_path,'wb'))
pickle.dump(test_list, open(test_path,'wb'))
pickle.dump(valid_list_standard, open(valid_path_standard,'wb'))
pickle.dump(test_list_standard, open(test_path_standard,'wb'))

print ('finished')