from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# mask1_path = 'D:/project/seg_from_zhen/origindata/mask_1/patient1_z_1.png'
# mask1 = Image.open(mask1_path)
# mask1 = np.array(mask1)
# print (set(list(mask1.flatten())))
format_size = (224,224)
source_data_dir = f'D:/project/seg_from_Zhen/parsed'
img_dir = os.path.join(source_data_dir, 'images')
mask1_dir = os.path.join(source_data_dir, 'mask_1')
mask2_dir = os.path.join(source_data_dir, 'mask_2')

saved_img_dir = f'D:/project/seg_from_zhen/expdata/ctseg/img'
saved_mask1_dir = f'D:/project/seg_from_zhen/expdata/ctseg/mask1'
saved_mask2_dir = f'D:/project/seg_from_zhen/expdata/ctseg/mask2'

typ1_dis = []
typ2_dis = []

for idx, fname in enumerate(os.listdir(img_dir)):
    print (idx, fname)
    # sfname = fname.split('.')[0]
    img_path = os.path.join(img_dir, fname)
    mask1_path = os.path.join(mask1_dir, fname)
    mask2_path = os.path.join(mask2_dir, fname)
    img = Image.open(img_path)
    img = img.resize(format_size)
    mask1 = Image.open(mask1_path)
    mask1 = mask1.resize(format_size)
    mask1 = np.array(mask1)
    mask1[mask1!=0] = 1
    print (set(list(mask1.flatten())))
    mask2 = Image.open(mask2_path)
    mask2 = mask2.resize(format_size)
    mask2 = np.array(mask2)
    mask2[mask2!=0] = 1
    typ1_dis.append(np.count_nonzero(mask1))
    typ2_dis.append(np.count_nonzero(mask2))
    # print (np.shape(img))
    img = np.stack([img, img, img])
    # print (np.shape(img))
    img = img.transpose(1,2,0)
    # print(np.shape(img))
    # print (np.min(mask1), np.max(mask1))
    # break
    np.save(os.path.join(saved_img_dir, '%d.npy'%idx), img)
    np.save(os.path.join(saved_mask1_dir, '%d.npy' % idx), mask1)
    np.save(os.path.join(saved_mask2_dir, '%d.npy' % idx), mask2)

print (np.mean(typ1_dis), np.mean(typ2_dis))

# img_path = 'D:/project/seg_from_zhen/images/Egyptian_Mau_127.jpg'
#
# # mask_path = 'D:/project/seg_from_zhen/annotations/trimaps/Egyptian_Mau_127.png'
# img = Image.open(img_path)
# # mask = Image.open(mask_path)
# # print (mask.size)
# img = img.resize(format_size)
#
# # mask = mask.resize(format_size)
# # print (mask.size)
# # mask = np.array(mask)
#
# # # Data type 1
# # mask[mask!=1] = 0
#
# ## Data type 2
# # mask[mask==2] = 0
# # mask[mask==3] = 1
# # print (mask)
# # print (np.max(mask), np.min(mask))
# # print (np.count_nonzero(mask), np.size(mask))
# # mask = Image.fromarray(mask)
# plt.imshow(img)
# plt.show()
#
#

