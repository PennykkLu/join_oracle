# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017

@author: Biagio Brattoli
"""
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import random
from torchvision import utils as vutils


class mydataset(data.Dataset):  ## classes:随机序列数， train==1，则裁64x64再放大成75x75并标准化，否则裁75x75, daluan==1则打乱顺序
    def __init__(self, data_path, filelist, train_key,classes=1000,kuai=3,train=1,daluan=1):
        self.data_path = data_path
        self.files = filelist #这个不是按名称顺序，不过也是固定的，没有随机性
        self.datalen = len(train_key)
        self.kuai = kuai
        self.daluan=daluan
        self.permutations = self.__retrive_permutations(classes) #已经生成好了一堆0-8的随机序列，从中取指定个数（classes）的序列
        self.__image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),  #双线性插值
            transforms.CenterCrop(255)])
        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            #transforms.Lambda(rgb_jittering), #里面放自己写的函数即可
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])
        self.__train_transform = transforms.Compose([
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor(),
        ])

        #能不能提前把图片装里面

        self.datapic = []  ##里面每个元素都是被分割成3x3的图片
        # 生成残片

        for i in train_key:  # 对于每张图片
            tiles = [None]*9
            framename = os.path.join(self.data_path, self.files[i])
            img = Image.open(framename).convert('RGB')
            '''if np.random.rand() < 0.30:
                img = img.convert('LA').convert('RGB')''' #有30%概率变成灰度图，这里没用，因为样本本身全是灰度图

            if img.size[0] != 255:
                img = self.__image_transformer(img)

            # 要把一张255*255的图片切3*3块
            s = float(img.size[0]) / self.kuai
            a = s / 2     ##块长的一半，用于计算块坐标

            for n in range(self.kuai * self.kuai):
                i = int(n / self.kuai)
                j = n % self.kuai
                c = [a * i * 2 + a, a * j * 2 + a]  # 那一块的中心坐标
                c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)  # 用中心坐标算左上右下坐标
                tile = img.crop(c.tolist())  # 这个出来是85*85的 论文中是75*75的？
                if train==1:
                    tile = self.__augment_tile(tile)
                    # Normalize the patches indipendently to avoid low level features shortcut
                    # 不知道标准化对测试有没有影响 先保留
                    m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
                    s[s == 0] = 1
                    norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
                    tile = norm(tile)
                else:
                    tile = self.__train_transform(tile)
                tiles[n] = tile
            self.datapic.append(tiles)
            '''for ii in range(9):
                vutils.save_image(tiles[ii], os.path.join(data_path, '0-{:d}.jpg'.format(ii)))
            img.save(os.path.join(data_path, '00.jpg'.format(ii)))'''



    def __getitem__(self, index):  ##daluan==1:返回打乱图片、permutation序号、未打乱图片
        if self.daluan ==1:
            tiles = self.datapic[index]
            order = random.randint(0,len(self.permutations)-1) #从序列里随机选一个
            #order = index % 1000
            data = [tiles[self.permutations[order][t]] for t in range(self.kuai*self.kuai)] #按这个序列把图片打乱
            data = torch.stack(data, 0) #把上面得到的几个tensor捏到一起
            return data, int(order), tiles
        else:
            tiles = self.datapic[index]
            data = [tiles[t] for t in range(self.kuai * self.kuai)]  # 按这个序列把图片打乱
            data = torch.stack(data, 0)  # 把上面得到的几个tensor捏到一起
            return data

    def __len__(self):
        return self.datalen

    def __dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))

        return file_names, labels

    def __retrive_permutations(self, classes):
        all_perm = np.load('/home/tszhe/CodePj/jiagu/JigsawPuzzle/permutations_1000.npy')
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        all_perm = all_perm[:classes]

        return all_perm


def rgb_jittering(im): #对每个残片进行抖动
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += random.randint(-2, 2)  #每个像素有+-2范围的抖动
    im[im > 255] = 255
    im[im < 0] = 0 #处理一下抖出边界的值
    return im.astype('uint8')


class testdataset(data.Dataset):
    def __init__(self, data_path, filelist, test_key, classes=1000,kuai=3):
        self.data_path = data_path
        self.files = filelist
        self.datalen = len(test_key)
        self.kuai = kuai
        self.__image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),  #双线性插值
            transforms.CenterCrop(255)])
        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.Lambda(rgb_jittering), #里面放自己写的函数即可
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std =[0.229, 0.224, 0.225])
        ])
        self.__test_transform = transforms.Compose([
            transforms.Resize((75,75),Image.BILINEAR),
            transforms.ToTensor(),
        ])
        self.tiles = []
        #生成残片

        for i in test_key: #对于每张图片
            framename = os.path.join(self.data_path, self.files[i])
            img = Image.open(framename).convert('RGB')

            if img.size[0] != 255:
                img = self.__image_transformer(img)

            # 要把一张255*255的图片切3*3块
            s = float(img.size[0]) / self.kuai
            a = s / 2

            for n in range(self.kuai*self.kuai):
                i = n / self.kuai
                j = n % self.kuai
                c = [a * i * 2 + a, a * j * 2 + a]  # 那一块的中心坐标
                c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)  # 用中心坐标算四个角坐标
                tile = img.crop(c.tolist())  # 这个出来是85*85的 论文中是75*75的？
                tile = self.__test_transform(tile)
                #tile = self.__augment_tile(tile)
                # Normalize the patches indipendently to avoid low level features shortcut
                # 不知道标准化对测试有没有影响 先保留
                '''m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
                s[s == 0] = 1
                norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
                tile = norm(tile)'''
                self.tiles.append(tile)



    def __getitem__(self, index):
        return self.tiles[index]

    def __len__(self): #这个len指的是有多少个tile，是乘过9的
        return self.datalen*self.kuai*self.kuai

    def __dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))

        return file_names, labels

    def __retrive_permutations(self, classes):
        all_perm = np.load('permutations_1000.npy')
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        all_perm = all_perm[:classes]

        return all_perm


class manudataset(data.Dataset):  ##key为对整张图片的选择索引
    def __init__(self, data_path, kuai=3,key=None):
        if key==None:
            self.key = None
            self.data_path = data_path
            self.files = os.listdir(data_path)
            self.files.sort()
            self.datalen = len(self.files)
            self.kuai = kuai
            self.__image_transformer = transforms.Compose([ #图片已经裁减过了，直接读取就好
                transforms.Resize((75, 75), Image.BILINEAR),
                transforms.ToTensor()])

            self.pic = []  ##每张图片都是75x75的小块
            self.picname = []
            #生成残片

            for i in range(self.datalen): #对于每张图片
                framename = os.path.join(self.data_path, self.files[i])
                img = Image.open(framename).convert('RGB')

                if img.size[0] != 75:
                    img = self.__image_transformer(img)

                self.pic.append(img)
                self.picname.append(framename)
        else:
            self.key=key
            self.data_path = data_path
            self.files = os.listdir(data_path)
            self.files.sort()
            self.datalen = len(key)
            self.kuai = kuai
            self.full_tiles_num = int(len(key)/5)
            self.__image_transformer = transforms.Compose([  # 图片已经裁减过了，直接读取就好
                transforms.Resize((75, 75), Image.BILINEAR),
                transforms.ToTensor()])

            self.pic = []
            self.picname = []
            # 生成残片

            for i in key:  # 对于每张图片
                tiles = [None] * 5
                for j in range(5):
                    framename = os.path.join(self.data_path, self.files[i*5+j])
                    #print(self.files[i*5+j])
                    img = Image.open(framename).convert('RGB')

                    if img.size[0] != 75:
                        img = self.__image_transformer(img)
                    tiles[j]=img
                self.pic.append(tiles)
                self.picname.append(self.files[i*5])

            '''cp_files = os.listdir(cppath)
            cp_files.sort()
            cp_datalen = len(cp_files)
            for i in range(cp_datalen):
                framename = os.path.join(cppath, cp_files[i])
                img = Image.open(framename).convert('RGB')

                if img.size[0] != 75:
                    img = self.__image_transformer(img)

                self.pic.append(img)
                self.picname.append(self.files[i])'''



    def __getitem__(self, index):
        if self.key==None:
            return self.pic[index], self.picname[index]
        else:
            tiles = self.pic[index]
            data = [tiles[t] for t in range(5)]  # 按这个序列把图片打乱
            data = torch.stack(data, 0)  # 把上面得到的几个tensor捏到一起
            return data,self.picname[index]


    def __len__(self):
        return self.datalen

    def __dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))

        return file_names, labels

