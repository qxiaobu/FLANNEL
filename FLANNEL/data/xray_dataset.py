import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, make_label_dict
from PIL import Image
import numpy as np

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
        self.dir_run = os.path.join(opt.data, run_type)  # get the image directory
        self.data_paths = make_dataset(self.dir_run)  # get image paths
        self.label_dict, self.label_sets = make_label_dict(opt.label_file)

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
        xray_path = self.data_paths[index]
        xray = Image.open(xray_path).convert('L')
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, xray.size)
        xray_transform = get_transform(self.opt, transform_params, grayscale=True)
        xray = xray_transform(xray)
        filename = xray_path.split('/')[-1].split('\\')[-1]
        label = self.label_sets[filename]
        id_label = self.label_dict[label]
        xray_label = np.zeros(len(self.label_dict))
        xray_label[id_label] = 1
        return {'A': xray, 'B': id_label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data_paths)
