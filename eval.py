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
from label_dict import emo_label_map,label_emo_map,class_names,class_indices

# from xai_emo_rec import explain_model
# from comparison import do_comparison

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = False

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

def eval_model(model, val_iter, loss_fn,config,mode="train",explain=False):

    confusion = config.confusion
    per_class = config.per_class
    y_true = []
    y_pred = []
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_epoch_acc3 = 0

    eval_batch_size = 1

    if confusion:
        conf_matrix = torch.zeros(config.output_size, config.output_size)
    if per_class:
           class_correct = list(0. for i in range(config.output_size))
           class_total = list(0. for i in range(config.output_size))

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            model = model.cuda()
            text, attn,target = select_input(batch,config)
            target = torch.autograd.Variable(target).long()

            if torch.cuda.is_available():
                if config.arch_name=="a_bert":
                    text = [text[0].cuda(),text[1].cuda()]
                    attn = attn.cuda()
                elif config.arch_name == "va_bert":
                    text = [text[0].cuda(),text[1].cuda(),text[2].cuda()]
                    attn = attn.cuda()
                elif config.arch_name == "vad_bert" or config.arch_name=="kea_bert":
                    text = [text[0].cuda(),text[1].cuda(),text[2].cuda(),text[3].cuda()]
                    attn = attn.cuda()
                else:
                    text = text.cuda()
                    attn = attn.cuda()
                target = target.cuda()

            prediction = model(text,attn)

            correct = np.squeeze(torch.max(prediction, 1)[1].eq(target.view_as(torch.max(prediction, 1)[1])))
            pred_ind = torch.max(prediction, 1)[1].view(target.size()).data

            if mode == "explain":
                pred_softmax = get_pred_softmax(prediction)
                explain_model(model,text,target.data,batch["utterance_data_str"],pred_ind,pred_softmax) ## use jupyter-notebook while doing explainations
            else:
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
            sns.heatmap(conf_matrix, annot=True,xticklabels=list(emo_label_map.keys()),yticklabels=list(emo_label_map.keys()))
            plt.show()
        if per_class:
            for i in range(config.output_size):
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                label_emo_map[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))

    if mode != "explain":

        f1_score_e = f1_score(y_true, y_pred, labels=class_indices,average='macro')
        f1_score_w = f1_score(y_true, y_pred, labels=class_indices,average='weighted')
        return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter),f1_score_e,f1_score_w,total_epoch_acc3/len(val_iter)



def load_model(resume,model,optimizer):


    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()
    optimizer.load_state_dict(checkpoint['optimizer']) ## during retrain TODO

    return model,optimizer,start_epoch



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Enter eval details')

    parser.add_argument('-r','--resume_path',type=str,help='Input resume path')
    parser.add_argument('-m','--mode',default='eval',type=str,help='Input resume path')
    parser.add_argument('-n','--rem_epoch',default=10,type=int,help='How much more epochs to run')
    parser.add_argument('-p','--patience',default=10,type=int,help='Early stopping patience')
    parser.add_argument('-c','--compare_model',default="",type=str,help='Provide resume path of the comparing model')

    args = parser.parse_args()

    resume_path = args.resume_path
    mode = args.mode
    rem_epoch = args.rem_epoch
    patience = args.patience
    comp_resume_path = args.compare_model
    ## Load the resume model parameters
    log_path = resume_path.replace("model_best.pth.tar","log.json")
    with open(log_path,'r') as f:
        log = json.load(f)
    f.close()

    if mode == "compare":
        do_comparison(resume_path,comp_resume_path)
        exit()

    ## Initialising parameters
    learning_rate = log["param"]["learning_rate"]
    batch_size = 1 ## batch_size=1 for testing and validation
    input_type = log["param"]["input_type"]
    arch_name = log["param"]["arch_name"]
    hidden_size = log["param"]["hidden_size"]
    embedding_length = log["param"]["embedding_length"]
    output_size = log["param"]["output_size"]
    tokenizer = log["param"]["tokenizer"]
    embedding_type = log["param"]["embedding_type"]

    ## Loading data
    print('Loading dataset')
    start_time = time.time()
    vocab_size, word_embeddings,train_iter, valid_iter ,test_iter= dataset.get_dataloader(batch_size,tokenizer,embedding_type,arch_name)
    finish_time = time.time()
    print('Finished loading. Time taken:{:06.3f} sec'.format(finish_time-start_time))

    eval_config = edict(log["param"])
    eval_config.param = log["param"]
    eval_config.resume_path = resume_path
    eval_config.batch_size = 1  ## batch_size=1 for testing and validation

    if mode == "explain":
        model = select_model(eval_config,vocab_size,word_embeddings,grad_check=False)
    else:
        model = select_model(eval_config,vocab_size,word_embeddings)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,eval_config.step_size, gamma=0.5)

    model,optimizer,start_epoch = load_model(resume_path,model,optimizer)

    if mode == "retrain": ## retrain from checkpoint TODO
        from train import train_model
        eval_config.patience = patience
        eval_config.nepoch = rem_epoch
        eval_config.confusion = False
        eval_config.per_class = True
        eval_config.start_epoch = start_epoch

        data  = (train_iter,valid_iter,test_iter)
        model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        writer = SummaryWriter('./runs/'+input_type+"/"+arch_name+"/")
        save_home = "./save/"+input_type+"/"+arch_name+"/"+model_run_time

        train_model(eval_config,data,model,loss_fn,optimizer,lr_scheduler,writer,save_home)

    elif mode == "eval":

        print(f'Train Acc: {log["train_acc"]:.3f}%, Valid Acc: {log["valid_acc"]:.3f}%, Test Acc: {log["test_acc"]:.3f}%')

        eval_config.confusion = True
        eval_config.per_class = True

        ## testing

        test_loss, test_acc,test_f1_score,test_f1_score_w,test_top3_acc= eval_model(model, test_iter,loss_fn,eval_config,mode)

        print(f'Top3 Acc: {test_top3_acc:.3f}%, F1 Score: {test_f1_score:.3f}, F1 Score W: {test_f1_score_w:.3f}')


    elif mode == "explain":

        print(f'Train Acc: {log["train_acc"]:.3f}%, Valid Acc: {log["valid_acc"]:.3f}%, Test Acc: {log["test_acc"]:.3f}%')

        eval_config.confusion = False
        eval_config.per_class = False

        ## explaining
        eval_model(model, test_iter,loss_fn,eval_config,mode,explain=True)
