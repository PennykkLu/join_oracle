# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: Biagio Brattoli
"""
import os, sys, numpy as np
import argparse
from time import time
from tqdm import tqdm
import math

import tensorflow # needs to call tensorflow before torch, otherwise crush
sys.path.append('Utils')


import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from PIL import Image
sys.path.append('Dataset')
from JigsawNetwork import Network,Network2,loss2
import torchvision.transforms as transforms
from Utils.TrainingUtils import adjust_learning_rate, compute_accuracy
from sklearn.metrics.pairwise import cosine_similarity
import shutil

parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet')
parser.add_argument('--data', type=str, help='Path to Imagenet folder')
parser.add_argument('--model', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=600, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--epochs', default=20000, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=71, type=int, help='batch size')
parser.add_argument('--kuai', default=3, type=int, help='kuai * kuai')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=6, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, No training')
args = parser.parse_args()

#from ImageDataLoader import DataLoader
from Dataset.JigsawImageLoader import mydataset,testdataset,manudataset

best_recall=0.0
best_recall2 = 0.0
random.seed(2022)

def main():


    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    print('Process number: %d'%(os.getpid()))
    
    ## DataLoader initialize ILSVRC2012_train_processed
    datapath = '/data/tszhe/jiaguwen/pic/all'
    manupath = '/data/tszhe/jiaguwen/pic/cut5'
    #cp_path = '/data/tszhe/jiaguwen/pic/cp'
    cp_path='/data/rhji/oracle_cut/exp/crops/jiagu'
    fileslist = os.listdir(manupath)
    files_len = int(len(fileslist)/5)
    numkey = list(range(files_len))
    random.shuffle(numkey)
    train_len = int(files_len*0.8)
    test_len = files_len - train_len
    train_key = numkey[:train_len]
    test_key = numkey[train_len:]


    train_data = manudataset(data_path=manupath,
                           kuai=3,
                           key=train_key) #这个叫dataloader的其实是Dataset的子类
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=args.batch, #包含batch*5张图片
                                            shuffle=True,
                                            pin_memory = True,
                                            num_workers=args.cores)
    


    val_data1 = manudataset(data_path=manupath,
                         kuai=3)

    val_loader1 = torch.utils.data.DataLoader(dataset=val_data1,
                                              batch_size=val_data1.__len__(),
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.cores)
    val_data2 = manudataset(data_path=manupath,
                           kuai=3,
                           key=test_key) #这个叫dataloader的其实是Dataset的子类
    val_loader2 = torch.utils.data.DataLoader(dataset=val_data2,
                                              batch_size=val_data2.__len__(),
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.cores)
    cp_data = manudataset(data_path=cp_path,
                            kuai=3)

    cp_loader = torch.utils.data.DataLoader(dataset=cp_data,
                                              batch_size=int(cp_data.__len__()/5),
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.cores)

    # for i, (cpimages, cpnames) in enumerate(cp_loader):
    #     cpimages = Variable(cpimages)
    #     if args.gpu is not None:
    #         cpimages = cpimages.cuda()


    # Network initialize
    net = Network(args.classes).cuda()
##    net.load_state_dict(torch.load("/home/hxxiang/jiagu_model_1113/model_save80.55554962158203_epoch310"))
    net.load_state_dict(torch.load("/home/tszhe/CodePj/jiagu/JigsawPuzzle/model_save_600_83.33332824707031"))
##    net.load_state_dict(torch.load("/home/rhji/jiagu/JigsawPuzzlePytorch-3/model_save_net_0.867231638418079"))
    net.zero_grad()
    '''acc = test(net, val_loader, 0)
    test1(net, val_loader1, val_data1.datalen,val_data1)'''
    net2 = Network2().cuda()
 ##   net2.load_state_dict(torch.load("/home/rhji/jiagu/JigsawPuzzlePytorch-3/model_save_net1_0.4166666666666667"))
    net2.zero_grad()

    net.eval()
    net2.eval()
    save_topk(net, net2, val_loader2, val_data2.datalen,cp_loader)
##    save_emb(net, net2, val_loader1, cp_loader)
 #   image_to_find = '/data/tszhe/jiaguwen/pic/cut5/100_3.jpg'
 #   find_pic(image_to_find,net,net2,val_loader1, int(val_data1.datalen / 5),cp_loader)
 #   find_lowest_pic(net, val_loader1, int(val_data1.datalen / 5))
 #   test1(net, net2, val_loader1, int(val_data1.datalen / 5),cp_loader)
    test2(net, net2, val_loader2, val_data2.datalen,cp_loader)


def find_lowest_pic(net,val_loader,item_num):
    print('find_lowest_pic')
    net.eval()
    with torch.no_grad():
        for i, (images,names) in enumerate(val_loader):  # 这个for其实没有意义，一次会把整个batch都取完。
            images = Variable(images)
            if args.gpu is not None:
                images = images.cuda()
            outputs = net(images, 0)
##            outputs = net2(outputs)
            outputs = outputs.cpu().numpy()
            simi = cosine_similarity(outputs)

            lowest_pos_num = np.zeros(5,dtype=int)
            for j in range(item_num):
                lowest_sim = 1
                randpicpos = 2
                picnum = j*5+randpicpos
                for bias in range(-2,3):
                    cossim = simi[picnum][picnum+bias]
                    if cossim < lowest_sim:
                        lowest_sim = cossim
                        lowest_pos = 2 + bias
                lowest_pos_num[lowest_pos] += 1
            #    print("pic:{:d},lowest_pos:{:d}".format(j, lowest_pos))
            for j in range(5):
                print("lowest_pos:{:d},sum:{:d},".format(j,lowest_pos_num[j]))


def test1(net,net2,val_loader,item_num,cp_loader): #这个item_num是有多少个整图，不要乘9
    global best_recall
    print('Evaluating network.......')
    print('Dataset: 170*5 + 1w')
    accuracy = []
    net.eval()
    net2.eval()
    with torch.no_grad():
        for i, (images,names) in enumerate(val_loader):  # 这个for其实没有意义，一次会把整个batch都取完。
            images = Variable(images)
            if args.gpu is not None:
                images = images.cuda()
            outputs = net(images, 0)
            outputs = net2(outputs)
            outputs = outputs.cpu().numpy()

            for j, (cpimages, cpnames) in enumerate(cp_loader):
                cpimages = Variable(cpimages)
                if args.gpu is not None:
                    cpimages = cpimages.cuda()
                cpoutputs = net(cpimages, 0)
                cpoutputs = net2(cpoutputs)
                cpoutputs = cpoutputs.cpu().numpy()
                outputs = np.append(outputs, cpoutputs, axis=0)

            simi = cosine_similarity(outputs)


            fpall = np.zeros(5,dtype=int)
            fnall = np.zeros(5,dtype=int)
            topn = (10, 20)
            ##topn = (10,20,50,200,500,1000,2000,3000,4000,5000,6000,7000,8000)
            tpall = np.zeros(len(topn),dtype=int)
            for j in range(item_num):

                #randpicpos = random.randint(0,args.kuai * args.kuai-1) #中间的块编号为4
                randpicpos = 2
                picnum = j*5+randpicpos
                idx = np.argsort(simi[picnum]).astype(np.int32)[::-1]
                delidx = np.where(idx == picnum)
                idx = np.delete(idx, delidx)
                #print(val_data.files[j]+','+str(randpicpos)+': ',end="")
                for ii,n in enumerate(topn):
                    tp = fp = fn = tn = 0
                    for k in range(n):
                        '''if ii == 1:
                            print(val_data.files[int(idx[k].item()/9)] + ',' + str(idx[k].item()%9)+' ',end="")'''
                        if int(idx[k]/5)==j:
                            tp = tp + 1
                    #print()
                    tpall[ii]+=tp
            print()
            tpall = tpall/item_num
            for j in range(len(topn)):
                precesion = tpall[j] / topn[j]
                recall = tpall[j] / 4
                f1 = 2.0*precesion*recall/(precesion+recall)
                print("top{:d} : precision:{:04f} recall:{:04f} f1:{:04f}".format(topn[j],precesion, recall, f1))
                if j==2 and recall>best_recall:
                    best_recall = recall
            print("best recall:",best_recall)



    net.train()
    net2.train()


def test2(net,net2,val_loader,item_num,cp_loader): #这个item_num是有多少张完整的图（不乘9，也不包括后面的残片）
    global best_recall2
    print('Evaluating test2.......')
    print('Dataset: 170*5*0.2 + 1w')

    accuracy = []
    net.eval()
    net2.eval()
    with torch.no_grad():
        for i, (images,names) in enumerate(val_loader):  # 这个for其实没有意义，一次会把整个batch都取完。
            images = Variable(images)
            if args.gpu is not None:
                images = images.cuda()
            outputs = net(images, 2)
##            outputs = net2(outputs)
            outputs = outputs.cpu().numpy()

            for j, (cpimages, cpnames) in enumerate(cp_loader):
                cpimages = Variable(cpimages)
                if args.gpu is not None:
                    cpimages = cpimages.cuda()
                cpoutputs = net(cpimages, 0)
##                cpoutputs = net2(cpoutputs)
                cpoutputs = cpoutputs.cpu().numpy()
                outputs = np.append(outputs, cpoutputs, axis=0)

            simi = cosine_similarity(outputs)


            fpall = np.zeros(5,dtype=int)
            fnall = np.zeros(5,dtype=int)
##            topn = (10,20)
            topn = (10, 20, 50, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000)
            tpall = np.zeros(len(topn), dtype=int)
            for j in range(item_num):

                #randpicpos = random.randint(0,args.kuai * args.kuai-1) #中间的块编号为4
                randpicpos = 2
                picnum = j*5+randpicpos
                #_, idx = simi[picnum].sort(descending=True)
                idx = np.argsort(simi[picnum]).astype(np.int32)[::-1]
                delidx = np.where(idx==picnum)
                idx = np.delete(idx,delidx)
                #print(val_data.files[j]+','+str(randpicpos)+': ',end="")
                for ii,n in enumerate(topn):
                    tp = fp = fn = tn = 0
                    for k in range(n):
                        '''if ii == 1:
                            print(val_data.files[int(idx[k].item()/9)] + ',' + str(idx[k].item()%9)+' ',end="")'''
                        if int(idx[k]/5)==j:
                            tp = tp + 1
                    #print()
                    tpall[ii]+=tp
                    if ii==5 and tp>4:
                        print(j,'error')
            print()
            tpall = tpall/item_num
            for j in range(len(topn)):
                precesion = tpall[j] / topn[j]
                recall = tpall[j] / 4
                f1 = 2.0*precesion*recall/(precesion+recall)
                print("top{:d} : precision:{:04f} recall:{:04f} f1:{:04f}".format(topn[j],precesion, recall, f1))
                if j==2 and recall>best_recall2:
                    best_recall2 = recall
            print("best recall2:",best_recall2)



    net.train()
    net2.train()


def save_emb(net,net2,val_loader,cp_loader):
    print('Saving feature.......')
    net.eval()
    net2.eval()
    with torch.no_grad():
        for i, (images, names) in enumerate(val_loader):  # 这个for其实没有意义，一次会把整个batch都取完。
            images = Variable(images)
            if args.gpu is not None:
                images = images.cuda()
            outputs = net(images, 0)
            outputs = net2(outputs)
            outputs = outputs.cpu().numpy()

            for j, (cpimages, cpnames) in enumerate(cp_loader):
                cpimages = Variable(cpimages)
                if args.gpu is not None:
                    cpimages = cpimages.cuda()
                cpoutputs = net(cpimages, 0)
                cpoutputs = net2(cpoutputs)
                cpoutputs = cpoutputs.cpu().numpy()
                outputs = np.append(outputs, cpoutputs, axis=0)
                names = np.append(names, cpnames)

            np.save('/home/hxxiang/feature.npy', outputs)
            np.save('/home/hxxiang/path.npy', names)
    print('saving end')


def save_topk(net,net2,val_loader,item_num,cp_loader):
    print('Dataset: 170*5*0.2 + 1w')

    net.eval()
    net2.eval()
    with torch.no_grad():
        for i, (images, names) in enumerate(val_loader):  # 这个for其实没有意义，一次会把整个batch都取完。
            images = Variable(images)
            if args.gpu is not None:
                images = images.cuda()
            outputs = net(images, 2)
 ##           outputs = net2(outputs)
            outputs = outputs.cpu().numpy()

            for j, (cpimages, cpnames) in enumerate(cp_loader):
                cpimages = Variable(cpimages)
                if args.gpu is not None:
                    cpimages = cpimages.cuda()
                cpoutputs = net(cpimages, 0)
 ##               cpoutputs = net2(cpoutputs)
                cpoutputs = cpoutputs.cpu().numpy()
                outputs = np.append(outputs, cpoutputs, axis=0)

            simi = cosine_similarity(outputs)

            tpall = np.zeros(6000, dtype=int)
            for j in range(item_num):

                # randpicpos = random.randint(0,args.kuai * args.kuai-1) #中间的块编号为4
                randpicpos = 2
                picnum = j * 5 + randpicpos
                # _, idx = simi[picnum].sort(descending=True)
                idx = np.argsort(simi[picnum]).astype(np.int32)[::-1]
                delidx = np.where(idx == picnum)
                idx = np.delete(idx, delidx)
                # print(val_data.files[j]+','+str(randpicpos)+': ',end="")
                tp = 0
                for k in range(6000):
                    '''if ii == 1:
                        print(val_data.files[int(idx[k].item()/9)] + ',' + str(idx[k].item()%9)+' ',end="")'''
                    if int(idx[k] / 5) == j:
                        tp = tp + 1
                    tpall[k] += tp

            print()
            tpall = tpall / item_num
            recall = tpall / 4
            np.save('CFN_topk_recall_2022.npy', recall)

    net.train()
    net2.train()


if __name__ == "__main__":
    main()
