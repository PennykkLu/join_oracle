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

sys.path.append('Dataset')
from JigsawNetwork import Network,Network2,loss2

from Utils.TrainingUtils import adjust_learning_rate, compute_accuracy
from sklearn.metrics.pairwise import cosine_similarity

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
    cp_path = '/data/tszhe/jiaguwen/pic/cp'
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
                                            batch_size=cp_data.__len__(),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=args.cores)
    
    iter_per_epoch = train_data.datalen/args.batch
    #print('Images: train %d, validation %d'%(train_data.datalen,val_data.datalen))
    
    # Network initialize
    net = Network(args.classes).cuda()
    net.load_state_dict(torch.load("/home/tszhe/CodePj/jiagu/JigsawPuzzle/model_save_600_83.33332824707031"))
    net.zero_grad()

    net2 = Network2().cuda()


    #criterion = nn.CrossEntropyLoss()
    criterion = loss2(0.5)
    #optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay = 5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=args.lr, weight_decay=0.0)
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
    t = 1

    #取出cp中的图片和名字
    for i, (cpimages, cpnames) in enumerate(cp_loader):  # 这个for其实没有意义，一次会把整个batch都取完。
        cpimages = Variable(cpimages)
        if args.gpu is not None:
            cpimages = cpimages.cuda()



    for epoch in range(int(args.iter_start/iter_per_epoch),args.epochs):
        if epoch%5==0:
            test1(net,net2,val_loader1,int(val_data1.datalen/5),cpimages)
            test2(net, net2, val_loader2, val_data2.datalen,cpimages)
            #test2(net, net2, val_loader3, train_data.datalen)
            #test1(net, net2, val_loader1, int(val_data1.datalen / 5))
            '''acc = test(net, val_loader,steps)
            if acc>best_acc:
                best_acc = acc
                torch.save(net.state_dict(), '/home/tszhe/CodePj/jiagu/JigsawPuzzle/model_save' + str(best_acc))'''
        #lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)
        #lr = args.lr
        '''if loss<1:
            lr = lr / 5
        if loss<0.5:
            lr = lr/5'''
        if loss<3.5:
            lr =1e-4
        
        end = time()
        net.train()
        net2.train()
        for i, (images,names) in enumerate(train_loader):
            batch_time.append(time()-end)
            if len(batch_time)>100:
                del batch_time[0]
            
            images = Variable(images)
            if args.gpu is not None:
                images = images.cuda()
            net.zero_grad()
            net2.zero_grad()
            # Forward + Backward + Optimize
            outputs = net(images, 2)
            outputs=net2(outputs)

 ##           cpoutputs = net(cpimages,0)
 ##           cpoutputs = net2(cpoutputs)


            loss = criterion(outputs)
            loss.backward()
            optimizer.step()
            optimizer2.step()
            loss = float(loss.cpu().data.numpy())

            if steps%1==0:
                print(('[%2d/%2d] %5d) , LR %.5f, Loss: % 1.3f' %(
                            epoch+1, args.epochs, steps,
                            lr, loss)))

            steps += 1

            end = time()

        if os.path.exists(args.checkpoint+'/stop.txt'):
            # break without using CTRL+C
            break

def test1(net,net2,val_loader,item_num,cpimages): #这个item_num是有多少个整图，不要乘9
    global best_recall
    print('Evaluating network.......')
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

            cpoutputs = net(cpimages, 0)
            cpoutputs = net2(cpoutputs)
            cpoutputs = cpoutputs.cpu().numpy()

            outputs = np.append(outputs, cpoutputs, axis=0)


            simi = cosine_similarity(outputs)


            tpall = np.zeros(3,dtype=int)
            fpall = np.zeros(3,dtype=int)
            fnall = np.zeros(3,dtype=int)
            topn = (20,50,100)
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

def test2(net,net2,val_loader,item_num,cpimages): #这个item_num是有多少张完整的图（不乘9，也不包括后面的残片）
    global best_recall2
    print('Evaluating test2.......')
    accuracy = []
    net.eval()
    net2.eval()
    with torch.no_grad():
        for i, (images,names) in enumerate(val_loader):  # 这个for其实没有意义，一次会把整个batch都取完。
            images = Variable(images)
            if args.gpu is not None:
                images = images.cuda()
            outputs = net(images, 2)
            outputs = net2(outputs)
            outputs = outputs.cpu().numpy()

            cpoutputs = net(cpimages, 0)
            cpoutputs = net2(cpoutputs)
            cpoutputs = cpoutputs.cpu().numpy()

            outputs = np.append(outputs, cpoutputs, axis=0)

            simi = cosine_similarity(outputs)


            tpall = np.zeros(6,dtype=int)
            fpall = np.zeros(6,dtype=int)
            fnall = np.zeros(6,dtype=int)
            topn = (4,10,20,50,100,222)
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

def test(net,val_loader,steps):
    print('Evaluating network.......')
    accuracy = []
    net.eval()
    for i, (images, labels, _) in enumerate(val_loader):
        images = Variable(images)
        if args.gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images,1)
        outputs = outputs.cpu().data

        prec1, prec5 = compute_accuracy(outputs, labels, topk=(1, 5))
        accuracy.append(prec1.item())

    print('TESTING: %d), Accuracy %.2f%%' %(steps,np.mean(accuracy)))
    net.train()
    return np.mean(accuracy)

if __name__ == "__main__":
    main()
