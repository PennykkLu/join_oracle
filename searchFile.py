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

def find_pic(image_path):


    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('Process number: %d' % (os.getpid()))
    # Network initialize
    net = Network(args.classes).cuda()
    ##    net.load_state_dict(torch.load("/home/tszhe/CodePj/jiagu/JigsawPuzzle/model_save_600_83.33332824707031"))
    net.load_state_dict(torch.load("/home/rhji/jiagu/JigsawPuzzlePytorch-3/model_save_net_0.867231638418079"))
    net.zero_grad()
    '''acc = test(net, val_loader, 0)
    test1(net, val_loader1, val_data1.datalen,val_data1)'''
    net2 = Network2().cuda()
    net2.load_state_dict(torch.load("/home/rhji/jiagu/JigsawPuzzlePytorch-3/model_save_net1_0.4166666666666667"))
    net2.zero_grad()

    ##读取图片并处理
    image_transformer = transforms.Compose([  # 图片已经裁减过了，直接读取就好
        transforms.Resize((75, 75), Image.BILINEAR),
        transforms.ToTensor()])

    image_to_find = Image.open(image_path).convert('RGB')
    if image_to_find.size[0] != 75:
        image_to_find = image_transformer(image_to_find)


    ##寻找最大匹配
    print('finding best match... ...')

    net.eval()
    net2.eval()
    with torch.no_grad():
        find_images = Variable(image_to_find)
        if args.gpu is not None:
            find_images = find_images.cuda()
        find_outputs = net(find_images, 3)
        find_outputs = net2(find_outputs)
        find_outputs = find_outputs.cpu().numpy()

    emb= np.load('/home/hxxiang/clean_feature.npy')
    path = np.load('/home/hxxiang/clean_path.npy')
    find_outputs = find_outputs.reshape((1,-1))
    simi = cosine_similarity(emb, find_outputs)
    simi_path = zip(simi, path)
    sorted_simi_path = sorted(simi_path, key=lambda x: x[0], reverse=True)

    net.train()
    net2.train()

    return sorted_simi_path