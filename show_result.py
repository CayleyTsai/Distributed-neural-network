# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:12:11 2018

@author: CCRG
"""

plt.plot(acc_list)
plt.plot(val_acc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()


plt.plot(loss_list)
plt.plot(val_loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='lower left')
plt.show()


plt.plot(branch_acc)
plt.plot(branch_val_acc)
plt.title('branch accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()


plt.plot(branch_loss_list)
plt.plot(branch_val_loss)
plt.title('branch loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='lower left')
plt.show()
