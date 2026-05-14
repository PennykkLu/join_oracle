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
    
    def forward(self, x,train=1):  ##train==1,训练完整个CFN,train==0训练到fc6且一张一张训练，train==2训练到fc6且一块一块训练
        if train==1:
            B, T, C, H, W = x.size()  ##Batch_sizt,Tile?,Channel,Height,Width
            x = x.transpose(0, 1) #后面可以把9个位置中相同位置中一个batch的所有图片一起算

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

        elif train == 2: #train==2 0和2的区别是，0不分5个位置（没有T这一维），2分了5个位置，每次for循环把某一张图的5个位置拿来算
            B, T, C, H, W = x.size()
            #x = x.transpose(0, 1)  # 后面可以把9个位置中相同位置中一个batch的所有图片一起算

            v = torch.zeros(size=(0,1024),dtype=torch.float32,device='cuda')
            for i in range(B):
                z = self.conv(x[i])
                z = self.fc6(z.view(T,-1))
                v = torch.cat((v,z),0)
            return v

        else :
            x = x.unsqueeze(0)
            x = self.conv(x)
            #print(x.shape)
            x = self.fc6(x.flatten())

        return x


def weights_init(model):
    if type(model) in [nn.Conv2d,nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)



class Network2(nn.Module):
    '''def __init__(self):
        super(Network2, self).__init__()
        self.linear1 = nn.Linear(1024, 2048, True)
        self.linear2 = nn.Linear(2048, 1024, True)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(1024, 1024, True)
        self.linear4 = nn.Linear(1024, 1024, True)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x'''

    def __init__(self):
        super(Network2, self).__init__()
        self.linear1 = nn.Linear(1024, 1024, True)
        self.linear2 = nn.Linear(1024, 1024, True)
        self.linear3 = nn.Linear(1024, 1024, True)
        self.linear4 = nn.Linear(1024, 1024, True)
        self.linear5 = nn.Linear(1024, 1024, True)
        self.linear6 = nn.Linear(1024, 1024, True)
        self.relu = nn.ReLU()
        self.relu1 = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.linear6(x)
        #x = self.relu1(x)
        return x

class loss2(nn.Module):
    def __init__(self,t=1):
        super(loss2, self).__init__()
        self.t = t
        return
    def forward(self,x,kuai2=5): #x=[batchsize*9,1024]
        T,C=x.size() #T-所有的emb个数
        lossB = 0
        #能不能先把所有两两之间的相似度算出来
        simiall = torch.cosine_similarity(x.unsqueeze(1),x.unsqueeze(0),dim=2)
        simiall = torch.exp(simiall/self.t)
        for b in range(int(T/5)):  ## b为表示整张图
            for i in range(b*5,b*5+5):   ##i为小块特征
                for j in range(b*5,b*5+5):
                    if i==j: continue
                    #在此计算相似度之和（分母）
                    #对应位置作为负样本
                    '''sumsimi_i = 0
                    for ii in range(T):
                        if (ii%9==i%9 and ii!=i) or ii%9==j%9:
                            #print(ii," ",end='')
                            sumsimi_i+=simiall[i][ii]'''

                    #随机采负样本
                    numkey = list(range(T))
                    for k in range(b*5,b*5+5):
                        numkey.remove(k)
                    random.shuffle(numkey)
                    numkey=numkey[:int(T/5)*2-2]
                    sumsimi_i = torch.sum(simiall[i][numkey])
                    lossB = lossB-torch.log(simiall[i][j]/(sumsimi_i+simiall[i][j]))

        lossB = lossB / (int(T/5)*kuai2*(kuai2-1))


        return lossB
