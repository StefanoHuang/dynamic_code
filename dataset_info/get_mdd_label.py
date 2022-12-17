import os
import re
import scipy.io as sio
import numpy as np
import pandas as pd

def get_mdd_label(brain_fmri_dir):
    #brain_fmri_dir = '/home/ly/hym_code/extracted/ROISignals_FunImgARglobalCWF'
    subjects_path = os.listdir(brain_fmri_dir)
    subjects_path.sort()
    #print(subjects_path)
    label = []
    for subject in subjects_path:
        #sub.append(subject)
        if subject.split('-')[1] == '1':
            label.append(1)
        if subject.split('-')[1] == '2':
            label.append(0)
    index = select_sub(subjects_path)
    label = np.array(label)
    label = np.delete(label,index)
    return list(label)


def select_sub(subjects_path):
    except_list = []
    with open("dataset_info/Info_not_selected.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            except_list.append(line)

    with open("dataset_info/HAMD_not_selected.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line not in except_list:
                except_list.append(line)

    with open("dataset_info/low_quality.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line not in except_list:
                except_list.append(line)
    
    subjects = []
    sub_index = []
    for subject in subjects_path:
        subjects.append((subject.split('_')[1]).split(('.'))[0])
    #print(subjects)
    for ex in except_list:
        sub_index.append(subjects.index(ex))
    return sub_index
