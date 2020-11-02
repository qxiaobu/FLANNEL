import os
from PIL import Image
import numpy as np
import torch
import csv
import random
from explore_version_03.data.base_dataset import BaseDataset
# add data augumentation 
#from torch.utils.data import DataLoader

model_list = ['densenet161','inception_v3', 'resnet152','resnext101_32x8d','vgg19_bn']
run_types = ['train','valid','test']



def parse_feature_data_dict(f_dir, cv, runtype):
  labels = []
  all_predict_v = []
  fff = True
  for model in model_list:
    predict_v = []
    sub_dir = f_dir%(model, cv)
    filename = 'result_detail_%s_%s_%s.csv'%(model, runtype, cv)
    print(filename)
    filepath = os.path.join(sub_dir, filename)
    with open(filepath,'r') as f:
      csv_reader = csv.reader(f)
      for row in csv_reader:
        xxx = np.array([float(x) for x in row[4:]])
        predict_v.append(xxx)
        if fff:
          cur_labels = np.array([float(row[0]),float(row[1]),float(row[2]),float(row[3])])
          labels.append(np.argmax(cur_labels))
    all_predict_v.append(np.array(predict_v))
    fff = False
  all_matrix = []
  features = np.concatenate(all_predict_v, 1)
  labels = np.array(labels)
  print (np.shape(features), np.shape(labels))
  return features.astype(float), labels


#cvs = ['cv1','cv2','cv3','cv4','cv5']
def parse_data_dict(f_dir, cv, runtype):
  labels = []
  all_predict_v = []
  fff = True
#  model = 'densenet161'
#  for cv in cvs:
  for model in model_list:
    predict_v = []
    sub_dir = f_dir%(model, cv)
    filename = 'result_detail_%s_%s_%s.csv'%(model, runtype, cv)
    print(filename)
    filepath = os.path.join(sub_dir, filename)
    with open(filepath,'r') as f:
      csv_reader = csv.reader(f)
      for row in csv_reader:
#        xxx = np.array([float(row[0]),float(row[1]),float(row[2]),float(row[3])])
        xxx = np.exp(np.array([float(row[0]),float(row[1]),float(row[2]),float(row[3])]))
        xxx = xxx/np.sum(xxx)
#        print (xxx)
        predict_v.append(xxx)
        if fff:
          cur_labels = np.array([float(row[4]),float(row[5]),float(row[6]),float(row[7])])
          labels.append(np.argmax(cur_labels))
    all_predict_v.append(np.array(predict_v))
    fff = False
#  all_matrix = []
#  for i in range(4):
#    temp_matrix = []
#    for j in range(5):
#      temp_matrix.append(all_predict_v[j][:, i])
#    temp_matrix = np.array(temp_matrix).transpose(1, 0)
#    print (np.shape(temp_matrix))
#    all_matrix.append(temp_matrix)
  features_l = np.concatenate(all_predict_v, 1)
  
#  all_predict_v = []
#  for model in model_list:
#    predict_v = []
#    sub_dir = './explore_version_03/results/%s_20200407_multiclass_%s'%(model, cv)
#    filename = 'result_detail_%s_%s_%s.csv'%(model, runtype, cv)
#    print(filename)
#    filepath = os.path.join(sub_dir, filename)
#    with open(filepath,'r') as f:
#      csv_reader = csv.reader(f)
#      for row in csv_reader:
#        xxx = np.array([float(x) for x in row[4:]])
#        predict_v.append(xxx)
#    all_predict_v.append(np.array(predict_v))
#  features_f = np.concatenate(all_predict_v, 1)
#  features = np.concatenate([features_l, features_f], 1)
#  print (np.shape(features))
  labels = np.array(labels)
#  print (np.shape(features), np.shape(labels))
  return features_l.astype(float), labels

def parse_data_dict_sampling(f_dir, cv, runtype):
  labels = []
  all_predict_v = []
  fff = True
  for model in model_list:
    predict_v = []
    sub_dir = f_dir%(model, cv)
    filename = 'result_detail_%s_%s_%s.csv'%(model, runtype, cv)
    print(filename)
    filepath = os.path.join(sub_dir, filename)
    with open(filepath,'r') as f:
      csv_reader = csv.reader(f)
      for row in csv_reader:
        xxx = np.exp(np.array([float(row[0]),float(row[1]),float(row[2]),float(row[3])]))
        xxx = xxx/np.sum(xxx)
        predict_v.append(xxx)
        if fff:
          cur_labels = np.array([float(row[4]),float(row[5]),float(row[6]),float(row[7])])
          labels.append(np.argmax(cur_labels))
    all_predict_v.append(np.array(predict_v))
    fff = False
  features_l = np.concatenate(all_predict_v, 1)
  labels = np.array(labels)
  new_data_normal = []
  new_data_covid = []
  for i in range(len(labels)):
    if labels[i]==0:
      new_data_normal.append((features_l[i], labels[i]))
    else:
      new_data_covid.append((features_l[i], labels[i]))
  new_data = new_data_covid * 10 + new_data_normal
  random.shuffle(new_data)
  features = []
  labels = []
  for i in range(len(new_data)):
    features.append(new_data[i][0])
    labels.append(new_data[i][1])
  features = np.array(features)
  labels = np.array(labels)
  return features.astype(float), labels

class EnsembleDatasetSampling(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, run_type = 'train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        features, label_list = parse_data_dict_sampling(opt.data_dir, opt.cv, run_type)  # get image paths
        self.features = torch.from_numpy(features)
        self.label_list = torch.from_numpy(label_list)
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        feature = self.features[index]
        label = self.label_list[index]
        
        return {'A': feature, 'B': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.features)
      
    def get_label_distri(self):
        counts = np.array([0.,0.,0.,0.])
        for item in self.label_list:
          counts[item] += 1.
        counts = 1000./counts
        return torch.from_numpy(np.array([counts]))
	
class EnsembleDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, run_type = 'train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        features, label_list = parse_data_dict(opt.data_dir, opt.cv, run_type)  # get image paths
        self.features = torch.from_numpy(features)
        self.label_list = torch.from_numpy(label_list)
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        feature = self.features[index]
        label = self.label_list[index]
        
        return {'A': feature, 'B': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.features)
      
    def get_label_distri(self):
        counts = np.array([0.,0.,0.,0.])
        for item in self.label_list:
          counts[item] += 1.
        counts = 1000./counts
        return torch.from_numpy(np.array([counts]))

if __name__ == "__main__":
  cdir = './explore_version_03/results/%s_20200407_multiclass_%s'
  myset = EnsembleDataset(cdir)
  myloader = DataLoader(dataset=myset, batch_size=2, shuffle=False)
  for data in myloader:
    print(data)
    
  