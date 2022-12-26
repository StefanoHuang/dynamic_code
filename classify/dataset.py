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
logger = Log(__name__).getlog()
import scipy.io as sio
import pickle
class MyDataset(Dataset):
    def __init__(self, args, dataset_path):
        self.args = args
        if self.args.dataset_name == 'abide':
            self.Y,wrong_index = get_abide_label()
        if self.args.dataset_name == 'mdd':
            self.Y,wrong_index = get_mdd_label(dataset_path)
        if self.args.dataset_name == 'hcp':
            self.Y,wrong_index = get_hcp_label(dataset_path)
        self.all_anno = self.get_conn(dataset_path,wrong_index)
        logger.info(f"{dataset_path} dataset length: {(len(self.all_anno))}")

    def get_conn(self,datadir,wrong):
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
            mat = sio.loadmat(subject)['DZStruct']
            feature.append(np.stack(list(mat[0][0])[:128]))
            '''
            if len(feature) == 10:
                break
            '''
        data_x = np.array(feature)
        return data_x

    def __len__(self):
        return len(self.all_anno)



    def __getitem__(self, idx):
        anno = self.all_anno[idx]
        y = self.Y[idx]
        #anno_dict = {"dynamic_feature": anno, "label":y}
        return list(zip(anno,y))


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

def BuildDataloader(dataset, batch_size, shuffle, num_workers, collate_fn):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=collate_fn, num_workers=num_workers,prefetch_factor=4,drop_last=True)
    return dataloader
