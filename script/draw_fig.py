from matplotlib import markers, pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
import numpy as np
def read_txt(path,type):
    result = []
    f = open(path,mode='r')               # 返回一个文件对象  
    line = f.readline()
    texture = 'dev_acc' if type == 'acc' else 'dev_loss' # 调用文件的 readline()方法  
    length = 8 if type == 'acc' else 9
    while line:  
        if texture in line:
            idx = line.index(texture)     
            line = line.replace('\n','')        
            result.append(float(line[idx+length:]))
        line = f.readline()   
    f.close()
    return result
plt.cla()
x_major_locator = MultipleLocator(10)
#print(x1)
#y1 = Loss_list
#print(y1)
rp_result = read_txt('/home/scut/hym_code/dynamic_code/_pretrain_repgraph_6layer.txt','acc')
x1 = np.arange(len(rp_result))
plt.title('ACC for replaced ROI prediction task', fontsize=10)
#plt.title('Performance for different numbers of attention heads on ABIDE', fontsize=10)
plt.plot(x1, rp_result, marker = 'o',label=u'acc')
#plt.plot(x1, y_auc, marker = 's',label=u'AUC')
#plt.plot(x1, y_f1,marker = '*',label=u'F1')
plt.xlabel('Epochs', fontsize=10)
#plt.xlabel('numbers of attention heads', fontsize=10)
plt.ylabel('acc', fontsize=10)
plt.grid()
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.legend()
plt.savefig("REPnew.png")
plt.show()