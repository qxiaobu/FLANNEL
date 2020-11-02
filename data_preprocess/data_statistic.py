#import os
#
#check_dir = './mimic-cxr-2.0.0.physionet.org/files'
#
#rule = 'dcm' #jpg
#
#c = 0
#for i in range(10, 11):
#  sub_folder_dir = os.path.join(check_dir, 'p%d'%i)
##  print (sub_folder_dir)
#  for patient_id in os.listdir(sub_folder_dir):
#      patient_dir = os.path.join(sub_folder_dir, patient_id)
##      print (patient_dir)
#      if os.path.isdir(patient_dir):
#        for subject_id in os.listdir(patient_dir):
#          subject_dir = os.path.join(patient_dir, subject_id)
#  #        print (subject_dir)
#          if os.path.isdir(subject_dir):
#            for file in os.listdir(subject_dir):
#              if rule in file:
#                c += 1
#
#print ('N of JPG images:', c)

import os
import pickle

p_set = set([])
kd = pickle.load(open('./data_preprocess/formal_kaggle_dict.pkl','rb'))
for key in kd.keys():
  print (key)
  p_set.add(key.split('_')[0])
print (len(p_set))
cd = pickle.load(open('./data_preprocess/formal_covid_dict_ap.pkl','rb'))
for key in cd.keys():
  p_set.add(key.split('_')[0])  
print (len(p_set))