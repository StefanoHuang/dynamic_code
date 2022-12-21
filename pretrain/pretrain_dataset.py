import os
from torch.utils.data import Dataset, DataLoader
from utils.sample import Sample, SampleList
import json
from utils.logger import Log
import torch
import numpy as np
from dataset_info.abide_label import get_abide_label
from dataset_info.get_hcp_label import get_hcp_label
from dataset_info.get_mdd_label import get_mdd_label
from torch.nn.utils.rnn import pad_sequence
from classify.dataset_path import dataset_path
logger = Log(__name__).getlog()
import scipy.io as sio
import pickle
class PretrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        _,abide_wrong_index = get_abide_label()
        _,mdd_wrong_index = get_mdd_label(dataset_path('mdd'))
        _,hcp_wrong_index = get_hcp_label(dataset_path('hcp'))
        abide_file = self.get_file_list(dataset_path('abide'),abide_wrong_index)
        mdd_file = self.get_file_list(dataset_path('mdd'),mdd_wrong_index)
        hcp_file = self.get_file_list(dataset_path('hcp'),hcp_wrong_index)
        self.all_anno = abide_file + mdd_file + hcp_file
        #pickle.dump(self.all_anno,open(preprocessed_path,mode='wb'))
        logger.info(f"Pretrain dataset length: {(len(self.all_anno))}")
        self.all_anno = np.array(self.all_anno)
    def get_file_list(self,datadir,wrong):
        subjects_path = os.listdir(datadir)
        subjects_path.sort()
        datapath = []
        feature = []
        for file in subjects_path:
            datapath.append(os.path.join(datadir, file))
        for i in range(len(datapath)):
            subject = datapath[i]
            if i in wrong:
                continue
            print(subject)
            feature.append(subject)
        return feature


    def __len__(self):
        return self.all_anno.shape[0]

    def __getitem__(self, idx):
        anno = self.all_anno[idx]
        return anno

class GraphDataset(Dataset):
    def __init__(self, args, filepath):
        self.args = args
        self.all_feat = self.get_conn(filepath)

    def get_conn(self,datapath):
        feature = []
        for i in range(len(datapath)):
            subject = datapath[i]
            #print(subject)
            mat = sio.loadmat(subject)['DZStruct']
            #print(mat.shape)
            #mat = self.gretna_tranmat(mat)
            #idx = np.triu_indices_from(mat[0], 1)
            #vec_networks = [m[idx] for m in mat]
            #vec_networks = np.array(vec_networks)
            feature += list(mat[0][0])
            '''
            if len(feature) == 10:
                break
            '''
        data_x = np.stack(feature)
        return data_x

    def __len__(self):
        return self.all_feat.shape[0]

    def __getitem__(self, idx):
        anno = self.all_feat[idx]
        return anno

class collater():
    def __init__(self, *params):
        self. params = params

    def __call__(self, data):
        data = list(zip(*data))
        data[1] = np.array(data[1])
        data[0] = [torch.tensor(d).float() for d in data[0]]
        data[0] = pad_sequence(data[0], padding_value=-1e9)
        data[1] = torch.from_numpy(data[1])
        return data    

def BuildDataloader(dataset, batch_size, shuffle, num_workers):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, prefetch_factor=10)
    return dataloader
