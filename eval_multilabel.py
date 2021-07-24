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
from label_dict import ed_label_dict,ed_emo_dict,class_names,class_indices,goemotions_label_dict,goemotions_emo_dict,semeval_emo_dict,semeval_label_dict


def eval_model(model, val_iter, loss_fn,log_dict,save_home):

    y_true = []
    y_score = []
    y_pred = []
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_epoch_acc3 = 0

    sigmoid_layer = nn.Sigmoid()
    threshold = 0.3 ## taken from the original paper
    eval_batch_size = 1

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):

            text, attn, target = select_input(batch,log_dict.param)

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
        if log_dict.param.dataset == "goemotions":
            for i in range(28):
                emotion = goemotions_emo_dict[i]
                emotion_true = y_true[:, i]
                emotion_pred = y_pred[:, i]
                # print(emotion_true,emotion_pred)
                results[emotion + "_accuracy"] = accuracy_score(emotion_true, emotion_pred)
                results[emotion + "_precision"], results[emotion + "_recall"], results[emotion + "_f1"], _ = precision_recall_fscore_support(
                        emotion_true, emotion_pred, average="binary")
        elif log_dict.param.dataset == "semeval":
            for i in range(11):
                emotion = semeval_emo_dict[i]
                emotion_true = y_true[:, i]
                emotion_pred = y_pred[:, i]
                # print(emotion_true,emotion_pred)
                results[emotion + "_accuracy"] = accuracy_score(emotion_true, emotion_pred)
                results[emotion + "_precision"], results[emotion + "_recall"], results[emotion + "_f1"], _ = precision_recall_fscore_support(
                        emotion_true, emotion_pred, average="binary")

    return total_epoch_loss/len(val_iter),results
