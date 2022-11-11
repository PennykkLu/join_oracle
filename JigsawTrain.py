# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: Biagio Brattoli
"""
import os, sys, numpy as np
import argparse
from time import time
from tqdm import tqdm

import tensorflow # needs to call tensorflow before torch, otherwise crush
sys.path.append('Utils')


import torch
import torch.nn as nn
from torch.autograd import Variable
import random

sys.path.append('Dataset')
from JigsawNetwork import Network

from Utils.TrainingUtils import adjust_learning_rate, compute_accuracy
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet')
parser.add_argument('--data', type=str, help='Path to Imagenet folder')
parser.add_argument('--model', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=600, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--epochs', default=20000, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=150, type=int, help='batch size')
parser.add_argument('--kuai', default=3, type=int, help='kuai * kuai')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=6, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, No training')
args = parser.parse_args()

#from ImageDataLoader import DataLoader
from Dataset.JigsawImageLoader import mydataset,testdataset,manudataset

myseed = random.randint(1,5000)
print('seed:'+str(myseed))
random.seed(myseed)
os.makedirs('/home/hxxiang/jiagu_model_'+str(myseed)+'/')

def main():
    '''if args.gpu is not None:
        print(('Using GPU %d'%args.gpu))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        str(args.gpu)
    else:
        print('CPU mode')'''

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    print('Process number: %d'%(os.getpid()))
    
    ## DataLoader initialize ILSVRC2012_train_processed
    datapath = '/data/tszhe/jiaguwen/pic/all'
    manupath = '/data/tszhe/jiaguwen/pic/cut5'
    fileslist = os.listdir(datapath)     ##返回指定的文件夹包含的文件或文件夹的名字的列表
    fileslist.sort()                   ##所有数据集
    files_len = len(fileslist)
    numkey = list(range(files_len))      ##key
    random.shuffle(numkey)
    train_len = int(files_len*0.8)
    test_len = files_len - train_len
    train_key = numkey[:train_len]
    test_key = numkey[train_len:]




    train_data = mydataset(data_path=datapath,
                           filelist = fileslist,
                           train_key=train_key,
                            classes=args.classes,
                           kuai=3,
                           train=1) #这个叫dataloader的其实是Dataset的子类
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=args.batch,
                                            shuffle=True,
                                            pin_memory = True,
                                            num_workers=args.cores)
    

    val_data = mydataset(data_path=datapath,
                           filelist=fileslist,
                           train_key = test_key,
                            classes=args.classes,
                            kuai=3,
                            train=1)
    val_data1 = manudataset(data_path=manupath,
                         kuai=3)
    val_loader1 = torch.utils.data.DataLoader(dataset=val_data1,
                                            batch_size=val_data1.__len__(),
                                            shuffle=False,
                                            pin_memory = True,
                                            num_workers=args.cores)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=val_data.__len__(),
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.cores)
    
    iter_per_epoch = train_data.datalen/args.batch
    print('Images: train %d, validation %d'%(train_data.datalen,val_data.datalen))
    
    # Network initialize
    net = Network(args.classes).cuda()
##    net.load_state_dict(torch.load("/home/tszhe/CodePj/jiagu/JigsawPuzzle/model_save_600_83.33332824707031"))
    net.zero_grad()
##    acc = test(net, val_loader, 0)
##    test1(net, val_loader1, int(val_data1.datalen/5), val_data1)
##    sys.exit()
    ############## Load from checkpoint if exists, otherwise from model ###############
    '''if os.path.exists(args.checkpoint):
        files = [f for f in os.listdir(args.checkpoint) if 'pth' in f]
        if len(files)>0:
            files.sort()
            #print files
            ckp = files[-1]
            net.load_state_dict(torch.load(args.checkpoint+'/'+ckp))
            args.iter_start = int(ckp.split(".")[-3].split("_")[-1])
            print('Starting from: ',ckp)
        else:
            if args.model is not None:
                net.load(args.model)
    else:
        if args.model is not None:
            net.load(args.model)'''

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay = 5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    #test(net, val_loader, val_data.datalen)
    
    ############## TESTING ###############
    if args.evaluate:
        #test(net,criterion,None,val_loader,0)
        return
    
    ############## TRAINING ###############
    print(('Start training: lr %f, batch size %d, classes %d'%(args.lr,args.batch,args.classes)))
    print(('Checkpoint: '+args.checkpoint))
    
    # Train the Model
    batch_time, net_time = [], []
    best_acc = 0.0
    steps = args.iter_start
    lr = args.lr
    loss = 10

    val_acc1_epoch = open('/home/hxxiang/jiagu_model_'+str(myseed)+'/val_acc1_epoch.txt', 'a')
    val_acc5_epoch = open('/home/hxxiang/jiagu_model_'+str(myseed)+'/val_acc5_epoch.txt', 'a')
    val_loss_epoch = open('/home/hxxiang/jiagu_model_' +str(myseed) + '/val_loss_epoch.txt', 'a')
    train_acc1_batch = open('/home/hxxiang/jiagu_model_'+str(myseed)+'/train_acc1_batch.txt', 'a')
    train_acc5_batch = open('/home/hxxiang/jiagu_model_'+str(myseed)+'/train_acc5_batch.txt', 'a')
    train_loss_batch = open('/home/hxxiang/jiagu_model_'+str(myseed)+'/train_loss_batch.txt', 'a')


    for epoch in range(int(args.iter_start/iter_per_epoch),args.epochs):
        ##每训练5次完整数据记录一次拼图任务
        if epoch%5==0:
            #test1(net,val_loader1,val_data1.datalen,val_data1)
            acc1, acc5, val_loss = test(net, val_loader,steps)
            val_acc1_epoch.write(str(acc1)+' ')
            val_acc5_epoch.write(str(acc5)+' ')
            val_loss_epoch.write(str(val_loss)+' ')
            if acc1>best_acc:
                best_acc = acc1
                torch.save(net.state_dict(), '/home/hxxiang/jiagu_model_'+str(myseed)+'/model_save' + str(best_acc) +'_epoch'+str(epoch))
                #acc = test(net, val_loader, steps)
        #lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)
        #lr = args.lr
        ##设置学习率
        if loss<1:
            lr = lr / 5
        if loss<0.5:
            lr = lr/5
        if loss<0.1:
            lr =lr/5
        
        end = time()
        for i, (images, labels, original) in enumerate(train_loader):
            batch_time.append(time()-end)
            if len(batch_time)>100:
                del batch_time[0]
            
            images = Variable(images)
            labels = Variable(labels) #里面只有一个标号，代表是用permutations中的第几条序列进行打乱的。
            if args.gpu is not None:
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            t = time()
            outputs = net(images,1) #一个classes数量的向量，未经过softmax
            net_time.append(time()-t)
            if len(net_time)>100:
                del net_time[0]
            
            prec1, prec5 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5)) #topk里面放想看第几顺位的准确度
            acc = prec1.item()
            acc1_batch = prec1.item()
            acc5_batch = prec5.item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = float(loss.cpu().data.numpy())

            ##每两个batch检查一次
            if steps%2==0:
                print(('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Loss: % 1.3f, Accuracy % 2.2f%%' %(
                            epoch+1, args.epochs, steps, 
                            np.mean(batch_time), np.mean(net_time),
                            lr, loss,acc)))
                train_acc1_batch.write(str(acc1_batch)+' ')
                train_acc5_batch.write(str(acc5_batch)+' ')
                train_loss_batch.write(str(loss)+' ')

            steps += 1

            end = time()

        if os.path.exists(args.checkpoint+'/stop.txt'):
            # break without using CTRL+C
            break

def test1(net,val_loader,item_num,val_data): #这个item_num是有多少个整图，不要乘9， val_lader1其实是人工分块的
    print('Evaluating network.......')
    accuracy = []
    net.eval()
    with torch.no_grad():
        for i, (images,_) in enumerate(val_loader): #这个for其实没有意义，一次会把整个batch都取完。就是为了计算它们之间的simi
            images = Variable(images)
            if args.gpu is not None:
                images = images.cuda()

            # Forward + Backward + Optimize
            outputs = net(images,0)
            outputs = outputs.cpu().numpy()
            simi = cosine_similarity(outputs)
            simi = torch.from_numpy(simi)
        ##求出特征

        tpall = np.zeros(3,dtype=int)
        fpall = np.zeros(3,dtype=int)
        fnall = np.zeros(3,dtype=int)
        topn = (20, 50, 100)
        for j in range(item_num):  ##j为第几张完整图

            # randpicpos = random.randint(0,args.kuai * args.kuai-1) #中间的块编号为4
            randpicpos = 2
            picnum = j * 5 + randpicpos
            _, idx = simi[picnum].sort(descending=True)  ##由第j张图片的中心位置找最近的特征
            # print(val_data.files[j]+','+str(randpicpos)+': ',end="")
            for ii, n in enumerate(topn):  ##前20张有多少找对了，前50张有多少找对了，前100张有多少找对了
                tp = fp = fn = tn = 0
                for k in range(n):

                    k = k + 1  # 最相近的肯定是自己，pos[1]才是别人的最相近的
                    '''if ii == 1:
                        print(val_data.files[int(idx[k].item()/9)] + ',' + str(idx[k].item()%9)+' ',end="")'''
                    if int(idx[k].item() / 5) == j:
                        tp = tp + 1
                # print()
                tpall[ii] += tp
        print()
        tpall = tpall / item_num
        for j in range(len(topn)):
            precesion = tpall[j] / topn[j]
            recall = tpall[j] / 4
            f1 = 2.0 * precesion * recall / (precesion + recall)
            print("top{:d} : precision:{:04f} recall:{:04f} f1:{:04f}".format(topn[j], precesion, recall, f1))


    net.train()


def test(net,val_loader,steps):
    print('Evaluating network.......')
    criterion = nn.CrossEntropyLoss()
    accuracy1 = []
    accuracy5 = []
    net.eval()
    for i, (images, labels, _) in enumerate(val_loader):
        images = Variable(images)
        if args.gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images,1)
        outputs = outputs.cpu().data

        prec1, prec5 = compute_accuracy(outputs, labels, topk=(1, 5))
        accuracy1.append(prec1.item())
        accuracy5.append(prec5.item())
        val_loss = criterion(outputs, labels)
        val_loss = float(val_loss.cpu().data.numpy())

    print('TESTING: %d), Accuracy %.2f%%' %(steps,np.mean(accuracy1)))
    net.train()
    return np.mean(accuracy1), np.mean(accuracy5),val_loss

if __name__ == "__main__":
    main()
