import argparse
import pickle

from utils.data_loading import treeDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
from pathlib import Path
# from utils.dice_score import dice_loss
from modellib import adaptUnet
# import torch.nn.functional as F
import os.path
import numpy as np
from sklearn.cluster import SpectralClustering
import joblib
import csv

def train_unit(datalist, config=None, model_saved_dir=None, middata_saved_dir=None):
    # config information
    img_scale = config.img_scale
    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    amp = config.amp
    device = config.device
    n_save_checkpoint = config.n_save_checkpoint
    # ['img_data', 'feat_data', 'label_data']
    # create data
    train_set = treeDataset(datalist, img_scale)
    n_train = len(train_set)
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # # create model
    net = adaptUnet(n_channels=3, n_classes=2, bilinear=True)
    # # control training
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion_classification = nn.BCEWithLogitsLoss()
    criterion_regression = nn.MSELoss()
    # training process
    train_loss = []
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            i = 0
            for batch in train_loader:
                i += 1
                images = batch['image']
                feat = batch['feat']
                true_value = batch['value']
                true_label = batch['label']

                if len(feat.size()) == 2:
                    images = images.to(device=device, dtype=torch.float32)
                    true_value = true_value.to(device=device, dtype=torch.float32)
                    true_label = true_label.to(device=device, dtype=torch.float32)
                    with torch.cuda.amp.autocast(enabled=amp):
                        value_pred = net([images])
                else:
                    images = images.to(device=device, dtype=torch.float32)
                    true_value = true_value.to(device=device, dtype=torch.float32)
                    feat = feat.to(device=device, dtype=torch.float32)
                    true_label = true_label.to(device=device, dtype=torch.float32)
                    with torch.cuda.amp.autocast(enabled=amp):
                        value_pred = net([images, feat])
                loss = criterion_regression(torch.squeeze(value_pred, 1), true_value)
                # print (i, loss)
                epoch_loss += loss.detach().numpy()
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

        if epoch % n_save_checkpoint == 0:
            torch.save(net.state_dict(), os.path.join(model_saved_dir, 'checkpoint_epoch.{}.pth'.format(epoch + 1)))

        train_loss.append(epoch_loss/float(i))
        # print (train_loss)

    with open(os.path.join(model_saved_dir, 'log.csv'), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['train_loss'])
        for row in train_loss:
            csv_writer.writerow([row])


    net.eval()
    img_path_set = []
    value_true_set = []
    value_pred_set = []
    for idx, batch in enumerate(train_loader):
        img_path = batch['img_path']
        img_path_set += img_path
        images = batch['image']
        feat = batch['feat']
        true_value = batch['value']

        if len(feat.size()) == 2:
            images = images.to(device=device, dtype=torch.float32)
            # true_value = true_value.to(device=device, dtype=torch.float32)
            with torch.cuda.amp.autocast(enabled=amp):
                value_pred = net([images])
        else:
            images = images.to(device=device, dtype=torch.float32)
            # true_value = true_value.to(device=device, dtype=torch.float32)
            feat = feat.to(device=device, dtype=torch.float32)
            with torch.cuda.amp.autocast(enabled=amp):
                value_pred = net([images, feat])

        value_true_set.append(true_value.detach().numpy())
        value_pred_set.append(np.squeeze(value_pred.detach().numpy(), 1))

    value_trues = np.concatenate(value_true_set, 0)
    value_pres = np.concatenate(value_pred_set, 0)
    next_value = value_trues - value_pres
    next_value_dir = os.path.join(middata_saved_dir, 'next_value')
    if os.path.isdir(next_value_dir) is False:
        os.mkdir(next_value_dir)
    cur_pred_dir = os.path.join(middata_saved_dir, 'cur_pred')
    if os.path.isdir(cur_pred_dir) is False:
        os.mkdir(cur_pred_dir)

    updated_datalist = []
    for idx in range(len(img_path_set)):
        img_path = img_path_set[idx]
        iid = img_path.split('\\')[-1].split('.')[0]
        tuple = (img_path,)
        np.save(os.path.join(cur_pred_dir, '%s.npy'%iid), value_pres[idx, :, :])
        tuple += (os.path.join(cur_pred_dir, '%s.npy'%iid),)
        np.save(os.path.join(next_value_dir, '%s.npy' % iid), next_value[idx, :, :])
        tuple += (os.path.join(next_value_dir, '%s.npy' % iid),)
        updated_datalist.append(tuple)

    return updated_datalist, next_value

def write_datalist(datalist, cur_database_dir):
    import csv
    datalist_path = os.path.join(cur_database_dir, 'datalist.csv')
    with open(datalist_path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['image_path', 'last_path', 'label_path'])
        for row in datalist:
            csv_writer.writerow(row)

def createModel_point(datalist, config=None, layerid=None):

    if len(datalist) < 20 or len(layerid.split('.'))>3:
        return 0
    cur_database_dir = os.path.join(config.obj_database_dir, layerid)
    cur_modelbase_dir = os.path.join(config.obj_modelbase_dir, layerid)
    if os.path.isdir(cur_database_dir) is False:
        os.makedirs(cur_database_dir)
    if os.path.isdir(cur_modelbase_dir) is False:
        os.makedirs(cur_modelbase_dir)
    write_datalist(datalist, cur_database_dir)
    updated_datalist, updated_residual = train_unit(datalist, config, cur_modelbase_dir, cur_database_dir)
    pickle.dump(updated_datalist, open(os.path.join(cur_database_dir, 'updated_datalist.pkl'),'wb'))
    pickle.dump(updated_residual, open(os.path.join(cur_database_dir, 'updated_residual.pkl'),'wb'))
    # updated_datalist = pickle.load(open('./updated_datalist.pkl','rb'))
    # updated_residual = pickle.load(open('./updated_residual.pkl','rb'))
    # print (len(updated_residual))
    n_case = len(updated_residual)
    feat = updated_residual.reshape([n_case, -1])
    ratio0 = 0
    ratio1 = 0
    n = 0
    best_sc = (None, 0, 0)
    while n<5 and (ratio0<0.3 or ratio1<0.3):
        sc = SpectralClustering(n_clusters=2, gamma=0.01, eigen_tol=1e-5, assign_labels='discretize', eigen_solver="arpack")
        sc.fit_predict(feat)
        ratio1 = np.sum(np.array(sc.labels_))/float(n_case)
        ratio0 = 1 - ratio1
        if ratio0>best_sc[1] and ratio0>best_sc[2]:
            best_sc = (sc, ratio0, ratio1)
        n += 1
    sc = best_sc[0]
    joblib.dump(sc, open(os.path.join(cur_modelbase_dir, 'cluster.model'), 'wb'))
    class1 = []
    class2 = []
    for idx, v in enumerate(sc.labels_):
        if v == 0:
            class1.append(idx)
        else:
            class2.append(idx)
    datalist0 = [updated_datalist[idx] for idx in class1]
    datalist1 = [updated_datalist[idx] for idx in class2]
    createModel_point(datalist0, config, layerid+'.0')
    createModel_point(datalist1, config, layerid+'.1')

def controler(datalist, config=None):
    createModel_point(datalist, config, '0')
    pass

def get_args():

    parser = argparse.ArgumentParser(description='Personal Segmentation Framework')
    parser.add_argument('-src_data', type=str, default='C:/Users/zhi.qiao/PycharmProjects/pythonProject/dataPreprocess/ctdata_reorg/origindata')
    parser.add_argument('-src_train_datalink', type=str, default='C:/Users/zhi.qiao/PycharmProjects/pythonProject/dataPreprocess/ctseg_data/new_modelData')
    parser.add_argument('-obj_modelbase_dir', type=str, default='C:/Users/zhi.qiao/PycharmProjects/pythonProject/Pytorch-UNet-master/personalized_segmentation/obj_modelbase')
    parser.add_argument('-obj_database_dir', type=str, default='C:/Users/zhi.qiao/PycharmProjects/pythonProject/Pytorch-UNet-master/personalized_segmentation/obj_database')
    parser.add_argument('-epochs', metavar='E', type=int, default=1, help='Number of epochs')
    parser.add_argument('-batch_size', metavar='B', type=int, default=10, help='Batch size')
    parser.add_argument('-learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('-img_scale', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('-n_save_checkpoint', type=int, default=1, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('-device', default='cpu', help='Use mixed precision')

    return parser.parse_args()

def get_init_datalist(datalink):
    import pickle
    datalist_path = os.path.join(datalink, '%s_%s_list.pkl'%('task3','train'))
    datalist = pickle.load(open(datalist_path, 'rb'))
    f_datalist = []
    for row in datalist:
        f_datalist.append((row[0], '', row[1]))
    return f_datalist

if __name__ == '__main__':

    args = get_args()
    f_datalist = get_init_datalist(args.src_train_datalink)
    controler(f_datalist, args)