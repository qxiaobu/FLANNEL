import os
import pickle

data_dir = './chest_xray'
data_type = ['train','val','test']
statistic_dict = {}
data_dict = {}

for dt in data_type:
  current_dir = os.path.join(data_dir, dt)
  normal_current_dir = os.path.join(current_dir, 'NORMAL')
#  data_dict[dt] = {}
  for filename in os.listdir(normal_current_dir):
    ws = filename.split('.')[0].split('-')
    if ws[0] == 'IM':
      patient_id = 'IM'+ws[1]
      subject_id = ws[2]
    elif ws[1] == 'IM':
      patient_id = 'IM'+ws[2]
      subject_id = ws[3]
    dtype = 'normal'
    image_path = os.path.join(normal_current_dir, filename)
    if data_dict.get(patient_id+'_'+subject_id) is None:
        data_dict[patient_id+'_'+subject_id] = {'class':{
                                              'COVID-19':0,
                                              'pneumonia_virus':0,
                                              'pneumonia_bacteria':0,
                                              'normal':0
                                              },
                                              'image_dict':{}}
    data_dict[patient_id+'_'+subject_id]['class']['normal'] = 1
    data_dict[patient_id+'_'+subject_id]['image_dict'][filename] = {
        'path':image_path, 
        'type':'AP'
    }
  abnormal_current_dir = os.path.join(current_dir, 'PNEUMONIA')
  for filename in os.listdir(abnormal_current_dir):
    ws = filename.split('.')[0].split('_')
    patient_id = ws[0]
    subject_id = ws[2]
    dtype = ws[1]
    image_path = os.path.join(abnormal_current_dir, filename)
    if data_dict.get(patient_id+'_'+subject_id) is None:
        data_dict[patient_id+'_'+subject_id] = {'class':{
                                              'COVID-19':0,
                                              'pneumonia_virus':0,
                                              'pneumonia_bacteria':0,
                                              'normal':0
                                              },
                                              'image_dict':{}}
    if  dtype in 'pneumonia_virus':
      data_dict[patient_id+'_'+subject_id]['class']['pneumonia_virus'] = 1
    if  dtype in 'pneumonia_bacteria':
      data_dict[patient_id+'_'+subject_id]['class']['pneumonia_bacteria'] = 1
    data_dict[patient_id+'_'+subject_id]['image_dict'][filename] = {
        'path':image_path, 
        'type':'AP'
    }

print ('finished')

print (dt)
y0 = 0
y1 = 0
y2 = 0
y3 = 0
z0 = 0
z1 = 0
z2 = 0
z3 = 0
i = 0
j = 0
for key, value in data_dict.items():
  for jpg_name, jpg_info in value['image_dict'].items():
    y0 += value['class']['COVID-19']
    y1 += value['class']['pneumonia_virus']
    y2 += value['class']['pneumonia_bacteria']
    y3 += value['class']['normal']
    j += 1
    if 'PA' in jpg_info['type'] or 'AP' in jpg_info['type']:
      i += 1
      z0 += value['class']['COVID-19']
      z1 += value['class']['pneumonia_virus']
      z2 += value['class']['pneumonia_bacteria']
      z3 += value['class']['normal']
print (i, j)
print (y0, y1, y2, y3)
print (z0, z1, z2, z3)
    
#saved_path = './data_preprocess/formal_kaggle_dict.pkl'
#if os.path.exists(saved_path):
#  os.remove(saved_path)
#pickle.dump(data_dict, open(saved_path,'wb'))
#print ('finish')

#print (i, j)
#5856 5856
#print (y0, y1, y2, y3)
#0 1493 2780 1583
#print (z0, z1, z2, z3)
#0 1493 2780 1583