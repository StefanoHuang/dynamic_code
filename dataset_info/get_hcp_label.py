import os
import re
import scipy.io as sio
import numpy as np
import pandas as pd
import csv
def get_hcp_label(brain_fmri_dir):
    #brain_fmri_dir = '/home/ly/hym_code/extracted/ROISignals_FunImgARglobalCWF'
    subjects_path = os.listdir(brain_fmri_dir)
    subjects_path.sort()
    #print(subjects_path)
    filename = []
    qc = []
    with open("dataset_info/behavior_hcp1200.csv", mode="r") as f:
        reader = csv.reader(f)
        next(reader)
        label = []
        # 逐行获取数据，并输出
        for row in reader:
            filename.append(row[0])
            if row[3] == 'M':
                label.append(0)
            else:
                label.append(1)
            qc.append(row[75])
    wrong = []
    finallabel = []
    for i in range(len(subjects_path)):
        tempidx = subjects_path[i].index('.')
        if subjects_path[i][11:tempidx] not in filename:
            wrong.append(i)
            continue
        idx = filename.index(subjects_path[i][11:tempidx])
        if 'D' in qc[idx] or 'C' in qc[idx]:
            wrong.append(i)
            continue
        else:
            finallabel.append(label[idx])
    return finallabel,wrong

