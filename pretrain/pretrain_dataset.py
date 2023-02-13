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
from pretrain.dataset_path import dataset_path
logger = Log(__name__).getlog()
import scipy.io as sio
import pickle
import random
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

class SequenceDataset(Dataset):
    def __init__(self, args, filepath):
        self.args = args
        self.all_feat,self.Y = self.get_conn_ori(filepath)

    def generate_sop(self,inputs):
        mask = np.random.random((1,inputs.shape[0]))
        prob = self.args.mlm_probability
        shuffle_list = np.arange(inputs.shape[2])
        temp_inputs = inputs.transpose(0,2,1)
        np.random.shuffle(shuffle_list)
        inputs = [temp_inputs[i,:,:][shuffle_list].transpose(1,0) if mask[0,i] < prob else temp_inputs[i,:,:].transpose(1,0) for i in range(mask.shape[1]) ]
        labels = [1  if mask[0,i]<prob else 0 for i in range(mask.shape[1])]
        return inputs,labels
    
    def generate_nsp(self,inputs):
        mask = np.random.random((1,inputs.shape[0]))
        length = inputs.shape[1]
        prob = self.args.mlm_probability
        shuffle_list = np.arange(inputs.shape[0])
        np.random.shuffle(shuffle_list)
        inputs = [np.concatenate((inputs[i,:length//2,:],inputs[shuffle_list[i],length//2:,:]),axis=0) if mask[0,i] < prob else inputs[i,:,:] for i in range(mask.shape[1]) ]
        labels = [1  if mask[0,i]<prob else 0 for i in range(mask.shape[1])]
        return inputs,labels
    
    def generate_rp(self,inputs):#predict roi region
        #labels = [i for i in range(inputs.shape[0])]
        newinputs = []
        labels = []
        steps = 8
        for i in range(inputs.shape[0]):
            for j in range((inputs.shape[1])//steps-1):
                newinputs.append(inputs[i,j*steps:(j+1)*steps,:])
                labels.append(i)
        return newinputs,labels
    
    def generate_rm(self,inputs):#random masked
        prob = 0.5
        mask = np.random.random((1,inputs.shape[0]))
        masked_indices = np.random.random((1,inputs.shape[1]))
        newinput = []
        for i in range(mask.shape[1]):
            if mask[0,i] < prob:
                mask_data = np.random.uniform(0,10,size=(inputs.shape[1:]))
                temp = [mask_data[k] if masked_indices[0,k] < self.args.mlm_probability else inputs[i,k,:] for k in range(inputs.shape[1])]
                newinput.append(np.array(temp))
            else:
                newinput.append(inputs[i,:,:])   
        #inputs = [ if mask[0,i] < prob else inputs[i,:,:] for i in range(mask.shape[1]) ]
        labels = [1  if mask[0,i]<prob else 0 for i in range(mask.shape[1])]        
        return newinput,labels
    def generate_nsp_new(self,inputs,replace_input):
        mask = np.random.random((1,inputs.shape[0]))
        length = inputs.shape[1]
        prob = self.args.mlm_probability
        inputs = [np.concatenate((inputs[i,:length//2,:],replace_input[i,length//2:,:]),axis=0) if mask[0,i] < prob else inputs[i,:,:] for i in range(mask.shape[1]) ]
        labels = [1  if mask[0,i]<prob else 0 for i in range(mask.shape[1])]
        return inputs,labels
    def get_conn(self,datapath):
        feature = []
        labels_list = []
        maxlen = 200
        shuffle_list = np.arange(len(datapath))
        np.random.shuffle(shuffle_list)
        for i in range(len(datapath)):
            subject = datapath[i]
            #print(subject)
            mat = sio.loadmat(subject)['DZStruct']
            temp = np.stack(list(mat[0][0])[:maxlen]).transpose(1,0,2)
            replace_subject = datapath[random.choice(shuffle_list)]
            mat_rep = sio.loadmat(replace_subject)['DZStruct']
            temp_rep = np.stack(list(mat_rep[0][0])[:maxlen]).transpose(1,0,2)
            temp,labels = self.generate_nsp_new(temp,temp_rep)
            feature += temp
            labels_list += labels
        feature = [np.pad(feature[i],((0,maxlen-feature[i].shape[0]),(0,0)),'constant',constant_values=(-1e9,-1e9)) if feature[i].shape[0]< maxlen else feature[i] for i in range(len(feature))]
        data_x = np.stack(feature)
        data_y = np.array(labels_list)
        return data_x, data_y

    def get_conn_ori(self,datapath):
        feature = []
        labels_list = []
        maxlen = 200
        for i in range(len(datapath)):
            subject = datapath[i]
            #print(subject)
            mat = sio.loadmat(subject)['DZStruct']
            temp = np.stack(list(mat[0][0])[:maxlen]).transpose(1,0,2)
            temp,labels = self.generate_sop(temp)
            #shuffle_list = np.arange(len(datapath))
            #np.random.shuffle(shuffle_list)
            #temp,labels = self.generate_rp(temp)
            #temp,labels = self.generate_rm(temp)
            feature += temp
            labels_list += labels
        feature = [np.pad(feature[i],((0,maxlen-feature[i].shape[0]),(0,0)),'constant',constant_values=(-1e9,-1e9)) if feature[i].shape[0]< maxlen else feature[i] for i in range(len(feature))]
        data_x = np.stack(feature)
        data_y = np.array(labels_list)
        return data_x, data_y

    def __len__(self):
        return self.all_feat.shape[0]

    def __getitem__(self, idx):
        return self.all_feat[idx],self.Y[idx]


def BuildDataloader(dataset, batch_size, shuffle, num_workers):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, prefetch_factor=4,drop_last=True)
    return dataloader
