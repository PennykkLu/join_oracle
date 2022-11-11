import numpy as np
import sys
import matplotlib.pyplot as plt

CFN_recall = np.load('/home/hxxiang/jiagu/CFN_topk_recall.npy')
my_recall = np.load('/home/hxxiang/jiagu/my_topk_recall.npy')

fig, axe = plt.subplots(figsize = (10,6))
x = np.linspace(0, len(CFN_recall), num=50);
f = (1/16978) * x
axe.plot(x, f, c='black', label = 'manual')

axe.plot(range(1,len(CFN_recall)+1), CFN_recall, c='g', label = 'CFN recall')
axe.plot(range(1,len(my_recall)+1), my_recall, c='b', label = 'DCRN recall')
axe.set(xlabel = 'Topk', ylabel = 'Recall', title = 'Recall Curve');
axe.legend()

plt.show()



sys.exit()


myseed = 1113
val_acc1_epoch = open('/home/hxxiang/jiagu_model_'+str(myseed)+'/val_acc1_epoch.txt', 'r')
val_acc5_epoch = open('/home/hxxiang/jiagu_model_'+str(myseed)+'/val_acc5_epoch.txt', 'r')
val_loss_epoch = open('/home/hxxiang/jiagu_model_' +str(myseed) + '/val_loss_epoch.txt', 'r')
train_acc1_batch = open('/home/hxxiang/jiagu_model_'+str(myseed)+'/train_acc1_batch.txt', 'r')
train_acc5_batch = open('/home/hxxiang/jiagu_model_'+str(myseed)+'/train_acc5_batch.txt', 'r')
train_loss_batch = open('/home/hxxiang/jiagu_model_'+str(myseed)+'/train_loss_batch.txt', 'r')

val_acc1_epoch = val_acc1_epoch.read().split()
#val_acc5_epoch = val_acc5_epoch.read().split()
val_loss_epoch = val_loss_epoch.read().split()
train_acc1_batch = train_acc1_batch.read().split()
#train_acc5_batch = train_acc5_batch.read().split()
train_loss_batch = train_loss_batch.read().split()

val_acc1_epoch = [float(s) for s in val_acc1_epoch]
#val_acc5_epoch = [float(s) for s in val_acc5_epoch]
val_loss_epoch = [float(s) for s in val_loss_epoch]
train_acc1_batch = [float(s) for s in train_acc1_batch]
#train_acc5_batch = [float(s) for s in train_acc5_batch]
train_loss_batch = [float(s) for s in train_loss_batch]

#只要前300个数据
train_len = len(train_acc1_batch)
val_len = len(val_acc1_epoch)

x_train = range(1,train_len*2+1,2)[:301]
train_acc1_batch = train_acc1_batch[:301]
train_loss_batch = train_loss_batch[:301]

ratio = len(x_train)/train_len
val_acc1_epoch = val_acc1_epoch[:int(val_len*ratio)]
val_loss_epoch = val_loss_epoch[:int(val_len*ratio)]
print('train_befor:'+str(train_len))
print('val_befor:'+str(val_len))
print('train_after:'+str(len(train_acc1_batch)))
print('val_after:'+str(len(val_acc1_epoch)))

bias = len(train_acc1_batch)/len(val_acc1_epoch)
print(bias)
x_val = range(1,len(val_acc1_epoch)*2+1,2)
x_val = [s*bias for s in x_val]



fig, axes = plt.subplots(1,2,figsize = (18,6))
axes[1].plot(x_val, val_acc1_epoch, c='b', label = 'val_acc')
#axes[1].plot(x_val, val_acc5_epoch, c='darkblue', label = 'val_acc5')
axes[1].plot(range(1,len(train_acc1_batch)*2+1,2), train_acc1_batch, c='g', label = 'train_acc')
#axes[1].plot(range(1,len(train_acc5_batch)*2+1,2), train_acc5_batch, c='darkgreen', label = 'train_acc5')
axes[1].set(xlabel = 'Batch', ylabel = 'Acc', title = 'Acc Curve');
axes[1].legend()

axes[0].plot(x_val, val_loss_epoch, c='b', label = 'val_loss')
axes[0].plot(range(1,len(train_loss_batch)*2+1,2), train_loss_batch, c='g', label = 'train_loss')
axes[0].set(xlabel = 'Batch', ylabel = 'Loss', title = 'Loss Curve');
axes[0].legend()
plt.show()