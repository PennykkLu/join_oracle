import numpy as np

emb= np.load('/home/hxxiang/clean_feature.npy')
path = np.load('/home/hxxiang/clean_path.npy')
delepath = '/data/rhji/oracle_cut/exp/crops/jiagu/09731z-097343.jpg'

index1 = np.argwhere(path==delepath)
print(index1)
print('delete:'+ delepath)

print(emb.shape)
print(path.shape)
emb = np.delete(emb,index1[0][0],axis=0)
path = np.delete(path,index1[0][0])

print(emb.shape)
print(emb.shape)
np.save('/home/hxxiang/clean_feature.npy', emb)
np.save('/home/hxxiang/clean_path.npy', path)