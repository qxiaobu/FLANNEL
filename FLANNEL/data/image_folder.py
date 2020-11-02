"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
import csv
import pickle

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def parse_data_dict(path):
    print (path)
    image_data_list = pickle.load(open(path, 'rb'))
    path_list = []
    label_list = []
    info_list = []
    for x in image_data_list:
      path_list.append(x[0])
      label_list.append(x[2])
      if len(x) == 4:
        info_list.append((x[1], x[3]))
      else:
        info_list.append(x[1])
    return path_list, label_list, info_list
  
def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a val directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def make_label_dict(label_file):
    label_dict = {}
    label_sets = {}
    assert os.path.isfile(label_file), '%s is not a val file' % dir
    with open(label_file, 'r') as f:
        csv_reader = csv.reader(f)
        index = 0
        for row in csv_reader:
            if index == 0:
                index += 1
                continue
            label_sets[row[9]] = row[4]
            if label_dict.get(row[4]) is None:
                label_dict[row[4]] = len(label_dict)
            index += 1
    return label_dict, label_sets

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
