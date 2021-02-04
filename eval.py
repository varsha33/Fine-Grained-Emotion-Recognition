import os
import time
import shutil
import time
import json
import random
import numpy as np
from easydict import EasyDict as edict
import argparse
from sklearn.metrics import classification_report,f1_score
import pickle

## torch packages
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

## for visulisation
import matplotlib.pyplot as plt

## custom
from select_model_input import select_model,select_input
import dataset
from label_dict import ed_label_dict,ed_emo_dict,class_names,class_indices


def accuracy_topk(output, target, topk=(3,)):
    """ Taken fromhttps://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840/2
    Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(pred,target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0].item()

def get_pred_softmax(logits):
    softmax_layer = nn.Softmax(dim=1)
    return softmax_layer(logits)

def eval_model(model, val_iter, loss_fn,log_dict):

    confusion = log_dict.param.confusion
    per_class = log_dict.param.per_class
    y_true = []
    y_pred = []
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_epoch_acc3 = 0

    eval_batch_size = 1

    if confusion:
        conf_matrix = torch.zeros(log_dict.param.output_size, log_dict.param.output_size)
    if per_class:
           class_correct = list(0. for i in range(log_dict.param.output_size))
           class_total = list(0. for i in range(log_dict.param.output_size))

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            model = model.cuda()
            text, attn,target = select_input(batch,log_dict.param)
            target = torch.autograd.Variable(target).long()

            if torch.cuda.is_available():
                text = [text[0].cuda(),text[1].cuda(),text[2].cuda(),text[3].cuda()]

                attn = attn.cuda()
                target = target.cuda()

            prediction = model(text,attn)

            correct = np.squeeze(torch.max(prediction, 1)[1].eq(target.view_as(torch.max(prediction, 1)[1])))
            pred_ind = torch.max(prediction, 1)[1].view(target.size()).data

            if confusion:
                for t, p in zip(target.data, pred_ind):
                        conf_matrix[t.long(), p.long()] += 1
            if per_class:
                label = target[0]
                class_correct[label] += correct.item()
                class_total[label] += 1

            loss = loss_fn(prediction, target)

            num_corrects = (pred_ind == target.data).sum()
            y_true.extend(target.data.cpu().tolist())
            y_pred.extend(pred_ind.cpu().tolist())

            acc = 100.0 * num_corrects/eval_batch_size
            acc3 = accuracy_topk(prediction, target, topk=(3,))
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
            total_epoch_acc3 += acc3

        if confusion:
            import seaborn as sns
            sns.heatmap(conf_matrix, annot=True,xticklabels=list(ed_label_dict.keys()),yticklabels=list(ed_label_dict.keys()),cmap='Blues')

            plt.show()
        if per_class:
            for i in range(log_dict.param.output_size):
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                ed_emo_dict[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))

    f1_score_e = f1_score(y_true, y_pred, labels=class_indices,average='macro')
    f1_score_w = f1_score(y_true, y_pred, labels=class_indices,average='weighted')
    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter),f1_score_e,f1_score_w,total_epoch_acc3/len(val_iter)

