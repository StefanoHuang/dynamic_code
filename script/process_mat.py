import os
import sys 
sys.path.append(".")
sys.path.append("..")
import h5py
import scipy.io as sio
from dataset_info.get_mdd_label import get_mdd_label
from dataset_info.get_hcp_label import get_hcp_label
from dataset_info.abide_label import get_abide_label
from classify.dataset_path import dataset_path
import numpy as np
def get_file_list(datadir,wrong):
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
def get_conn(datapath,name):
    #feature = []
    for i in range(len(datapath)):
        subject = datapath[i]
        mat = sio.loadmat(subject)['DZStruct']
        temp = np.stack(list(mat[0][0]))
        filename = subject.split('/')[-1][:-4]
        f = h5py.File(f'/home/ly/hym_code/dynamic_dataset/preprocessed/{name}/{filename}.h5', 'w')
        f['image'] = temp
        f.close()  #关闭文件
        '''
        if temp.shape[0] < 128:
            temp = np.pad(temp,((0,128-temp.shape[0]),(0,0),(0,0)),'constant',constant_values=(-1e9,-1e9))
        '''
        #feature.append(temp)
    #data_x = np.array(feature)
    return


_,mdd_wrong_index = get_abide_label()
mdd_file = get_file_list(dataset_path('abide'),mdd_wrong_index)
print(len(mdd_file))
#conn = get_conn(mdd_file)
get_conn(mdd_file,'abide')
