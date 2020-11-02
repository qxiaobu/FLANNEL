import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_augumentation
#from data.image_folder import make_dataset, make_label_dict
from data.image_folder import parse_data_dict
from PIL import Image
import numpy as np

# add data augumentation 

class XrayDataset(BaseDataset):
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
        self.image_paths, self.label_list, self.info_list = parse_data_dict(self.data_dir)  # get image paths

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
        xray_path = self.image_paths[index]
        xray = Image.open(xray_path).convert('L')
        label = self.label_list[index]
        info = self.info_list[index]
        if label == 0:
          transform_params = get_params(self.opt, xray.size)
          xray_transform = get_transform(self.opt, transform_params, grayscale=True)
          xray = xray_transform(xray)
        else:
          transform_t = info[1]
          transform_params = get_params(self.opt, xray.size)
          xray_transform = get_transform_augumentation(self.opt, transform_params, transform_type = transform_t, grayscale=True)
          xray = xray_transform(xray)
        return {'A': xray, 'B': label, 'info': info}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
