# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:49:48 2018

@author: CCRG
"""
import pandas as pd
"""
test = list(np.ones(100)*0.8337)
test_branch = list(np.ones(100)*0.7321)
data = [acc_list,loss_list,val_acc_list,val_loss_list,branch_acc,branch_loss_list,branch_val_acc,branch_val_loss,test,test_branch]
"""
test = list(np.ones(100)*0.749)
data = [acc_list,loss_list,val_acc_list,val_loss_list,test]

data = list(map(list, zip(*data)))#transpose the list
labels = ['acc','loss', 'val_acc','val_loss', 'test_acc']
df = pd.DataFrame.from_records(data, columns=labels)

df.to_csv('11-1local_06_statics.csv', sep=',')