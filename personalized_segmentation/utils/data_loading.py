import csv
import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pickle

class BasicDataset(Dataset):

    def __init__(self, datalist, scale: float = 1.0, mask_suffix: str = '', dtype: str = ''):
        self.dtype = dtype
        self.datalist = datalist
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

    def __len__(self):
        return len(self.datalist)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        (img_path, feat_path, value_path) = self.datalist[idx]
        iid = img_path.split('\\')[-1].split('.')[0]
        img = self.load(img_path)
        feat = self.load(feat_path) if feat_path != '' else np.zeros([1])
        value = self.load(value_path)


        img = self.preprocess(img, self.scale, is_mask=False)
        feat = self.preprocess(feat, self.scale, is_mask=False) if feat != np.zeros([1]) else feat
        value = self.preprocess(value, self.scale, is_mask=True)
        label = value.copy()
        label[label!=0]=1

        return {
            'img_path': img_path,
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'feat':  torch.as_tensor(feat.copy()).float().contiguous(),
            'value': torch.as_tensor(value.copy()).long().contiguous(),
            'label': torch.as_tensor(label.copy()).long().contiguous(),
        }


class treeDataset(BasicDataset):
    def __init__(self, datalist, scale=1):
        super().__init__(datalist, scale, mask_suffix='_mask', dtype = 'list')


def get_datalist(datalist_path):
    with open(datalist_path, 'r', newline='') as f:
        csv_reader = csv.reader(f)
        datalist = []
        for idx, row in enumerate(csv_reader):
            # print (idx, row)
            if idx > 0 and len(row) > 0:
                datalist.append(row)
    return datalist

# local test
if __name__ == '__main__':
    datalist_path = 'C:/Users/zhi.qiao/PycharmProjects/pythonProject/Pytorch-UNet-master/personalized_segmentation/obj_database/0/datalist.csv'
    datalist = get_datalist(datalist_path)
    td = treeDataset(datalist)
    for key, value in td[0].items():
        print (key, value)
    # print (td.datalist[0])