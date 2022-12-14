# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:57:01 2017

@author: Biagio Brattoli
"""
import torch
import torch.nn as nn
from torch import cat
import math
import random
import torch.nn.init as init
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append('Utils')
from Utils.Layers import LRN

class Network(nn.Module):

    def __init__(self, classes=1000):
        super(Network, self).__init__()

        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1',nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv.add_module('relu1_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv.add_module('relu2_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('relu3_s1',nn.ReLU(inplace=True))

        self.conv.add_module('conv4_s1',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('relu4_s1',nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('relu5_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1',nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(256*3*3, 1024))
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7',nn.Linear(9*1024,4096))
        self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        self.fc7.add_module('drop7',nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8',nn.Linear(4096, classes))
        
        #self.apply(weights_init)

    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict and 'fc8' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def forward(self, x,train=1):
        if train==1:
            B, T, C, H, W = x.size()
            x = x.transpose(0, 1) #???????????????9?????????????????????????????????batch????????????????????????

            x_list = []
            for i in range(9):
                z = self.conv(x[i])
                z = self.fc6(z.view(B, -1))
                z = z.view([B, 1, -1])
                x_list.append(z)

            x = cat(x_list, 1)
            x = self.fc7(x.view(B, -1))
            x = self.classifier(x)

        elif train==0:
            B, C, H, W = x.size()
            x = self.conv(x)
            #print(x.shape)
            x = self.fc6(x.view(B, -1))
            #print(x.shape)

        else: #train==2 0???2???????????????0??????9??????????????????T???????????????2??????9??????????????????for????????????????????????9??????????????????
            B, T, C, H, W = x.size()
            #x = x.transpose(0, 1)  # ???????????????9?????????????????????????????????batch????????????????????????

            v = torch.zeros(size=(0,1024),dtype=torch.float32,device='cuda')
            for i in range(B):
                z = self.conv(x[i])
                z = self.fc6(z.view(T,-1))
                v = torch.cat((v,z),0)
            return v


        return x


def weights_init(model):
    if type(model) in [nn.Conv2d,nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)



class Network2(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()
        self.linear1 = nn.Linear(1024, 1024, True)
        self.linear2 = nn.Linear(1024, 1024, True)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(1024, 1024, True)
        self.linear4 = nn.Linear(1024, 1024, True)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        '''x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)'''
        return x

class loss2(nn.Module):
    def __init__(self,t=1):
        super(loss2, self).__init__()
        self.t = t
        return
    def forward(self,x,kuai2=5): #x=[batchsize*9,1024]
        T,C=x.size() #T-?????????emb??????
        lossB = 0
        #??????????????????????????????????????????????????????
        simiall = torch.cosine_similarity(x.unsqueeze(1),x.unsqueeze(0),dim=2)
        simiall = torch.exp(simiall/self.t)
        rightpos = [1,3,4,5,7]
        for b in range(int(T/9)):
            for i in range(b*9,b*9+8):
                if i%9 not in rightpos: continue #???????????????5???

                for j in range(b*9,b*9+8):
                    if j % 9 not in rightpos or i==j: continue
                    #???????????????????????????????????????
                    #???????????????????????????
                    '''sumsimi_i = 0
                    for ii in range(T):
                        if (ii%9==i%9 and ii!=i) or ii%9==j%9:
                            #print(ii," ",end='')
                            sumsimi_i+=simiall[i][ii]'''

                    #??????????????????
                    numkey = list(range(T))
                    numkey.remove(i)
                    random.shuffle(numkey)
                    numkey=numkey[:int(T/9)*2-1]
                    sumsimi_i = torch.sum(simiall[i][numkey])
                    lossB = lossB-torch.log(simiall[i][j]/sumsimi_i)

        lossB = lossB / (int(T/9)*kuai2*(kuai2-1)/2)
        return lossB
