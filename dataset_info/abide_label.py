import math
import numpy as np
import os
import csv


def get_abide_label():
    qc1 = []
    qc2 = []
    qc3 = []
    filename = []
    label = []
    wrong = []
    files = os.listdir('/home/ly/hym_code/ABIDE_ts')
    files.sort()
    with open("dataset_info/Phenotypic_V1_0b_preprocessed1.csv", mode="r") as f:
        reader = csv.reader(f)
        header = next(reader)

        # 逐行获取数据，并输出
        for row in reader:
            filename.append(row[6])
            label.append(int(row[7])-1)
            qc1.append(row[-11])
            qc2.append(row[-7])
            qc3.append(row[-3])
    finallabel = []
    count = 0
    for i in range(len(files)):
        tempidx = files[i].index('_rois')
        if files[i][:tempidx] not in filename:
            count += 1
            wrong.append(i)
            continue
        idx = filename.index(files[i][:tempidx])
        if qc1[idx] == 'fail' or qc2[idx] == 'fail' or qc3[idx] == 'fail':
            wrong.append(i)
            continue
        else:
            finallabel.append(label[idx])
    finallabel = np.array(finallabel)
    #print(finallabel.shape)
    #print(finallabel)
    #np.save('abide_label.npy',finallabel)
    #np.save('abide_wrong.npy',wrong)
    return finallabel,wrong