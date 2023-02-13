import scipy.io as sio
import pickle
import numpy as np
import os
import collections
def get_conn(datadir):
    time_dict = collections.defaultdict(int)
    subjects_path = os.listdir(datadir)
    subjects_path.sort()
    datapath = []
    for file in subjects_path:
        datapath.append(os.path.join(datadir, file))
    for i in range(len(datapath)):
        subject = datapath[i]
        mat = sio.loadmat(subject)['DZStruct']
        #print(subject)
        #feature.append(np.stack(list(mat[0][0])[:128]))
        mat = np.stack(list(mat[0][0]))
        time_dict[mat.shape[0]] += 1
    print(time_dict)
get_conn("/data/abide")
