import pickle
import random

def pairwise_data_dict(path):

    image_data_list = pickle.load(open(path, 'rb'))
    path_list = []
    label_list = []
    info_list = []
    
    l = len(image_data_list)
    ids = []
    for i in range(0, l):
      for j in range(i+1, l):
        ids.append((i, j))

    random.shuffle(ids)
    f_ids = ids[:10000]
    for cid in f_ids:    
      path_list.append((image_data_list[cid[0]][0], image_data_list[cid[1]][0]))
      label_list.append((image_data_list[cid[0]][2], image_data_list[cid[1]][2]))
      info_list.append(1. if image_data_list[cid[0]][2] != image_data_list[cid[1]][2] else 0.)

    return path_list, label_list, info_list




if __name__=="__main__":
    print("main")
    a,b,c = pairwise_data_dict('./data_preprocess/standard_data_multiclass_0325/exp_test_list.pkl')
