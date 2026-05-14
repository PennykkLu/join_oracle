# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: Biagio Brattoli
"""
import os, sys, numpy as np
import argparse
from time import time
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
from torchvision import utils as vutils

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
parser.add_argument('--classes', default=1000, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--epochs', default=2000, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=150, type=int, help='batch size')
parser.add_argument('--kuai', default=3, type=int, help='kuai * kuai')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--cores', default=6, type=int, help='number of CPU core for loading')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set, No training')
args = parser.parse_args()

#from ImageDataLoader import DataLoader
from Dataset.JigsawImageLoader import mydataset,testdataset


def main():
    '''if args.gpu is not None:
        print(('Using GPU %d'%args.gpu))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        str(args.gpu)
    else:
        print('CPU mode')'''

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    print('Process number: %d'%(os.getpid()))
    
    ## DataLoader initialize ILSVRC2012_train_processed
    datapath = '/data/tszhe/jiaguwen/pic/all'
    fileslist = os.listdir(datapath)
    files_len = len(fileslist)
    numkey = list(range(files_len))
    random.shuffle(numkey)
    train_len = int(files_len*0.8)
    test_len = files_len - train_len
    train_key = numkey[:train_len]
    test_key = numkey[train_len:]




    train_data = mydataset(data_path=datapath,
                           filelist = fileslist,
                           train_key=train_key,
                            classes=args.classes,
                           kuai=3) #这个叫dataloader的其实是Dataset的子类
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=args.batch,
                                            shuffle=True,
                                            pin_memory = True,
                                            num_workers=args.cores)
    

    val_data = mydataset(data_path=datapath,
                           filelist=fileslist,
                           train_key = test_key,
                            classes=args.classes,
                            kuai=3)
    val_data1 = testdataset(data_path=datapath,
                         filelist=fileslist,
                         test_key=test_key,
                         classes=args.classes,
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
    #net.load_state_dict(torch.load("/home/tszhe/CodePj/jiagu/JigsawPuzzle/model_save"))
    ############## Load from checkpoint if exists, otherwise from model ###############
    if os.path.exists(args.checkpoint):
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
            net.load(args.model)

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay = 5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    #test(net, val_loader, val_data.datalen)
    '''logger = Logger(args.checkpoint+'/train')
    logger_test = Logger(args.checkpoint+'/test') #可有可无？'''
    
    ############## TESTING ###############
    if args.evaluate:
        test1(net, val_loader1, val_data1.datalen, val_data1)
        acc = test(net, val_loader, 0)
        sys.exit(0)
        return

    ############## TRAINING ###############
    print(('Start training: lr %f, batch size %d, classes %d'%(args.lr,args.batch,args.classes)))
    print(('Checkpoint: '+args.checkpoint))
    
    # Train the Model
    batch_time, net_time = [], []
    best_acc = 0.0
    steps = args.iter_start
    lr = args.lr
    for epoch in range(int(args.iter_start/iter_per_epoch),args.epochs):
        if epoch%5==0:
            test1(net,val_loader1,val_data1.datalen,val_data1)
            acc = test(net, val_loader,steps)
            if acc>best_acc:
                best_acc = acc
                torch.save(net.state_dict(), '/home/tszhe/CodePj/jiagu/JigsawPuzzle/model_save' + str(best_acc))
        #lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)
        #lr = args.lr
        if epoch % 40 == 0 and epoch>0:
            lr = lr/5
        
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

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = float(loss.cpu().data.numpy())

            if steps%20==0:
                print(('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Loss: % 1.3f, Accuracy % 2.2f%%' %(
                            epoch+1, args.epochs, steps, 
                            np.mean(batch_time), np.mean(net_time),
                            lr, loss,acc)))

                
                original = [im[0] for im in original]
                imgs = np.zeros([9,75,75,3])
                for ti, img in enumerate(original):
                    img = img.numpy()
                    imgs[ti] = np.stack([(im-im.min())/(im.max()-im.min()) 
                                         for im in img],axis=2)
                
                #logger.image_summary('input', imgs, steps)

            steps += 1

            '''if steps%1000==0:
                filename = '%s/jps_%03i_%06d.pth.tar'%(args.checkpoint,epoch,steps)
                net.save(filename)
                print('Saved: '+args.checkpoint)'''
            
            end = time()

        if os.path.exists(args.checkpoint+'/stop.txt'):
            # break without using CTRL+C
            break

def test1(net,val_loader,item_num,val_data): #这个item_num是有多少个整图，不要乘9
    print('Evaluating network.......')
    accuracy = []
    net.eval()
    with torch.no_grad():
        for i, images in enumerate(val_loader): #这个for其实没有意义，一次会把整个batch都取完。就是为了计算它们之间的simi
            images = Variable(images)
            if args.gpu is not None:
                images = images.cuda()

            # Forward + Backward + Optimize
            outputs = net(images,0)
            outputs = outputs.cpu().numpy()
            simi = cosine_similarity(outputs)
            simi = torch.from_numpy(simi)


        tpall = np.zeros(3,dtype=int)
        fpall = np.zeros(3,dtype=int)
        fnall = np.zeros(3,dtype=int)
        topn = (5,10,20)
        for j in range(item_num):

            randpicpos = random.randint(0,args.kuai * args.kuai-1) #中间的块编号为4
            #randpicpos = 4
            picnum = j*args.kuai*args.kuai+randpicpos
            _, idx = simi[picnum].sort(descending=True)

            '''for ii in range(10):
                vutils.save_image(val_data.kuai[idx[k].item()], "/home/tszhe/CodePj/jiagu/JigsawPuzzle/pic_save/"+str(ii+1)+'.jpg')
            vutils.save_image(val_data.kuai[idx[k].item()],'/home/tszhe/CodePj/jiagu/JigsawPuzzle/pic_save/0.jpg')
            break'''
            for ii,n in enumerate(topn):
                tp = fp = fn = tn = 0
                for k in range(n):
                    k = k+1 #最相近的肯定是自己，pos[1]才是别人的最相近的
                    if int(idx[k].item()/(args.kuai*args.kuai))==j:
                        tp = tp + 1
                    else:
                        fp = fp + 1
                fn = args.kuai*args.kuai-1-tp
                tpall[ii]+=tp
                fpall[ii]+=fp
                fnall[ii]+=fn

        tpall = tpall/item_num
        fpall = fpall/item_num
        fnall = fnall/item_num
        for j in range(len(topn)):
            precesion = 1.0*tpall[j]/(tpall[j]+fpall[j])
            recall = 1.0*tpall[j]/(tpall[j]+fnall[j])
            f1 = 2.0*precesion*recall/(precesion+recall)
            print("top{:d} : precision:{:04f} recall:{:04f} f1:{:04f}".format(topn[j],precesion, recall, f1))


    net.train()


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
