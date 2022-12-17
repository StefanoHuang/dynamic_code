import os
from torch.utils.data import Dataset, DataLoader
from utils.sample import Sample, SampleList
import json
from utils.logger import Log
import torch
import numpy as np
from dataset_info.abide_label import get_abide_label
from torch.nn.utils.rnn import pad_sequence
logger = Log(__name__).getlog()
import scipy.io as sio
import pickle
class MyDataset(Dataset):
    def __init__(self, args, dataset_path):
        self.args = args
        if self.args.dataset_name == 'abide':
            self.Y,wrong_index = get_abide_label()
        subjects_path = os.listdir(dataset_path)
        subjects_path.sort()  
        self.all_anno = []          
        #self.all_anno = self.get_conn(dataset_path,wrong_index)
        
        for i in range(len(subjects_path)):
            if i in wrong_index:
                continue
            self.all_anno.append(subjects_path[i])
        self.all_anno = np.array(self.all_anno)
        logger.info(f"{dataset_path} dataset length: {(len(self.all_anno))}")

    def __len__(self):
        return self.all_anno.shape[0]

    def __getitem__(self, idx):
        anno = list(self.all_anno[idx])
        y = self.Y[idx]
        #anno_dict = {"dynamic_feature": anno, "label":y}
        return list(zip(anno,y))


class collater():
    def __init__(self, datadir):
        self. datadir = datadir

    def __call__(self, data):
        data = list(zip(*data))
        data[0] = self.get_conn(data[0])
        data[1] = np.array(data[1])
        data[0] = [torch.tensor(d[:64]).float() for d in data[0]]
        data[0] = pad_sequence(data[0], padding_value=-1e9)
        data[1] = torch.from_numpy(data[1])
        return data    
    def gretna_tranmat(self,mat):
        m = []
        for i in range(len(mat[0][0])):
            m.append(mat[0][0][i])
        m = np.array(m)
        return m
    def get_conn(self,subjects_path):
        datapath = []
        for file in subjects_path:
            datapath.append(os.path.join(self.datadir, file))
        feature = []
        for i in range(len(datapath)):
            subject = datapath[i]
            mat = sio.loadmat(subject)['DZStruct']
            #print(subject)
            #print(mat.shape)
            #mat = self.gretna_tranmat(mat)
            #idx = np.triu_indices_from(mat[0], 1)
            #vec_networks = [m[idx] for m in mat]
            #vec_networks = np.array(vec_networks)
            feature.append(np.stack(list(mat[0][0])))
            '''
            if len(feature) == 10:
                break
            '''
        #data_x = np.array(feature)
        return feature


def BuildDataloader(dataset, batch_size, shuffle, num_workers, collate_fn):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=collate_fn, num_workers=num_workers)
    return dataloader
