import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_augumentation
#from data.image_folder import make_dataset, make_label_dict
from data.pairwise_fileread import pairwise_data_dict
from PIL import Image
import numpy as np
import torch

# add data augumentation 

class PairwiseDataset(BaseDataset):
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
        self.data_dir = opt.data%run_type  # get the image data directory
        self.image_paths, self.label_list, self.info_list = pairwise_data_dict(self.data_dir)  # get image paths
        self.run_type = run_type
        
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
        (xray_path_1, xray_path_2) = self.image_paths[index]
        xray_1 = Image.open(xray_path_1).convert('L')
        xray_2 = Image.open(xray_path_2).convert('L')
        
        (label_1, label_2) = self.label_list[index]
        
        label3 = self.info_list[index]
        transform_params = get_params(self.opt, xray_1.size)
        xray_transform = get_transform(self.opt, transform_params, grayscale=True, run_type = self.run_type)
        xray_1 = xray_transform(xray_1)
        xray_1 = torch.cat((xray_1, xray_1, xray_1), 0)
        xray_2 = xray_transform(xray_2)
        xray_2 = torch.cat((xray_2, xray_2, xray_2), 0)
        xray = torch.stack([xray_1, xray_2], dim=0)
        
        return {'A': xray, 'B': np.array([label_1, label_2]), 'info': label3}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
      
#    def get_label_distri(self):
#        counts = np.array([0.,0.,0.,0.])
#        for item in self.label_list:
#          counts[item] += 1.
#        counts = 1000./counts
#        return torch.from_numpy(np.array([counts]))
          
