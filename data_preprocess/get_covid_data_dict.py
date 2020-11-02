import csv
import os
import pickle

data_root_dir = './covid-chestxray-dataset-master'
image_root_dir = './covid-chestxray-dataset-master/images'
info_file_name = 'metadata.csv'

info_path = os.path.join(data_root_dir, info_file_name)

data_dict = {}

covid = ['COVID-19']
pneumonia_virus = ['SARS']
pneumonia_bacteria = ['Pneumocystis','Streptococcus']
normal = ['No Finding']
x0 = 0
x1 = 0
x2 = 0
x3 = 0
n_ct = 0
n_ards = 0
with open(info_path,'r') as f:
  csv_reader = csv.reader(f)
  i = 0 
  for row in csv_reader:
    if i == 0:
      i += 1
      continue
    patient_id = row[0]
    subject_id = row[1]
    view = row[6]
    image_name = row[10]
    disease = row[4]
    modality = row[7]
    if 'ray' not in modality:
      n_ct += 1
      continue
    jpg_path = os.path.join(image_root_dir, image_name)
    if os.path.exists(jpg_path) and 'AP' in view:    
      if data_dict.get(patient_id+'_'+subject_id) is None:
          data_dict[patient_id+'_'+subject_id] = {'class':{
                                              'COVID-19':0,
                                              'pneumonia_virus':0,
                                              'pneumonia_bacteria':0,
                                              'normal':0
                                              },
                                              'image_dict':{}}
      if disease == 'ARDS':
        n_ards += 1
        continue
      if disease in covid:
        data_dict[patient_id+'_'+subject_id]['class']['COVID-19'] = 1
        x0 += 1
      if disease in pneumonia_virus:
        data_dict[patient_id+'_'+subject_id]['class']['pneumonia_virus'] = 1
        x1 += 1
      if disease in pneumonia_bacteria:
        data_dict[patient_id+'_'+subject_id]['class']['pneumonia_bacteria'] = 1
        x2 += 1
      if disease in normal:
        data_dict[patient_id+'_'+subject_id]['class']['normal'] = 1
        x3 += 1
      data_dict[patient_id+'_'+subject_id]['image_dict'][image_name] = {
        'path':jpg_path, 
        'type':view
      }


y0 = 0
y1 = 0
y2 = 0
y3 = 0
z0 = 0
z1 = 0
z2 = 0
z3 = 0
v0 = 0
v1 = 0
v2 = 0
v3 = 0
w0 = 0
w1 = 0
w2 = 0
w3 = 0
i = 0
j = 0
ap_list = []
pa_list = []
for key, value in data_dict.items():
  for jpg_name, jpg_info in value['image_dict'].items():
    print(jpg_info['type'])
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
      if 'PA' in jpg_info['type']:
        pa_list.append(jpg_name)
        v0 += value['class']['COVID-19']
        v1 += value['class']['pneumonia_virus']
        v2 += value['class']['pneumonia_bacteria']
        v3 += value['class']['normal']
      if 'AP' in jpg_info['type']:
        ap_list.append(jpg_name)
        w0 += value['class']['COVID-19']
        w1 += value['class']['pneumonia_virus']
        w2 += value['class']['pneumonia_bacteria']
        w3 += value['class']['normal']
                 
print (x0, x1, x2, x3)
print (i, j)
print (y0, y1, y2, y3)
print (z0, z1, z2, z3)
print (v0, v1, v2, v3)
print (w0, w1, w2, w3)
#pickle.dump(data_dict, open('./data_preprocess/formal_covid_dict_ap.pkl','wb'))
##pickle.dump(pa_list, open('pa_list.pkl','wb'))
###saved_path = './data_preprocess/formal_covid_dict.pkl'
###if os.path.exists(saved_path):
###  os.remove(saved_path)
###pickle.dump(data_dict, open(saved_path,'wb'))
###print ('finish')
##
###             Xray	
###新冠	          111	 
###非新冠病毒肺炎	 11	  
###细菌肺炎	        8	  
###无症状	         1	 
###总计	          131	  
###158=23例CT图像+4例ARDS无法划分肺炎类别		
