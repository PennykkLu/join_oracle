# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:53:30 2017

@author: bbrattol
"""

def adjust_learning_rate(optimizer, epoch, init_lr=0.1, step=30, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (decay ** (epoch // step))
    print('Learning Rate %f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  #每个样本最有可能被划分的前5类
    pred = pred.t() #pred[i]为这些图的第i可能性的分类
    correct = pred.eq(target.view(1, -1).expand_as(pred))  #把他复制几份堆在一起

    res = []
    for k in topk: #把每一种topi都与真实值做比较，算一个准确度
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

