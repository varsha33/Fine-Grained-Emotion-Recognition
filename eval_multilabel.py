import os
import time
import shutil
import time
import json
import random
import numpy as np
from easydict import EasyDict as edict
import argparse
from sklearn.metrics import classification_report,f1_score,precision_recall_fscore_support,average_precision_score,accuracy_score
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
from label_dict import emo_label_map,label_emo_map,class_names,class_indices,goemotions_label_dict,goemotions_emo_dict,semeval_emo_dict,semeval_label_dict


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

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

def eval_model(model, val_iter, loss_fn,config,arch_name,save_home,mode="train"):

    confusion = config.confusion
    per_class = config.per_class
    y_true = []
    y_score = []
    y_pred = []
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_epoch_acc3 = 0

    sigmoid_layer = nn.Sigmoid()
    threshold = 0.2 ## taken from the original paper
    eval_batch_size = 1

    if confusion:
        conf_matrix = torch.zeros(config.output_size, config.output_size)
    if per_class:
           class_correct = list(0. for i in range(config.output_size))
           class_total = list(0. for i in range(config.output_size))

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):

            text, attn, target = select_input(batch,config,arch_name)

            if torch.cuda.is_available():
                text = [text[0].cuda(),text[1].cuda(),text[2].cuda(),text[3].cuda()]

                model = model.cuda()
                attn = attn.cuda()
                target = target.cuda()


            prediction = model(text,attn)

            loss = loss_fn(prediction, target)

            pred_ind = sigmoid_layer(prediction).detach().cpu().tolist()[0]

            y_score.append(pred_ind)
            y_pred.append([0 if p <threshold else 1 for p in pred_ind])
            y_true.append(target.detach().cpu().tolist()[0])
            total_epoch_loss += loss.item()

        os.makedirs(save_home,exist_ok=True)
        results = {}
        p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average="macro")

        results["precision"] = p
        results["recall"] = r
        results["f1"] = f1

        y_true = np.array([np.array(i) for i in y_true])
        y_pred = np.array([np.array(i) for i in y_pred])

        # print(y_true[:5,:],y_pred[:5,:])
        if config.dataset == "goemotions":
            for i in range(27):
                emotion = goemotions_emo_dict[i]
                emotion_true = y_true[:, i]
                emotion_pred = y_pred[:, i]
                # print(emotion_true,emotion_pred)
                results[emotion + "_accuracy"] = accuracy_score(emotion_true, emotion_pred)
                results[emotion + "_precision"], results[emotion + "_recall"], results[emotion + "_f1"], _ = precision_recall_fscore_support(
                        emotion_true, emotion_pred, average="binary")
        elif config.dataset == "semeval":
            for i in range(11):
                emotion = semeval_emo_dict[i]
                emotion_true = y_true[:, i]
                emotion_pred = y_pred[:, i]
                # print(emotion_true,emotion_pred)
                results[emotion + "_accuracy"] = accuracy_score(emotion_true, emotion_pred)
                results[emotion + "_precision"], results[emotion + "_recall"], results[emotion + "_f1"], _ = precision_recall_fscore_support(
                        emotion_true, emotion_pred, average="binary")

    return total_epoch_loss/len(val_iter),results
