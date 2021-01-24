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

# from xai_emo_rec import explain_model
# from comparison import do_comparison

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
    threshold = 0.3 ## taken from the original paper
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
            if config.dataset == "ed":
                target = torch.autograd.Variable(target).long()

            if torch.cuda.is_available():
                if arch_name =="electra" or arch_name == "bert":
                    text = text.cuda()
                else:
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
        raw_result_dict = {"y_true":y_true,"y_score":y_score}
        f = open(save_home+"/raw_result.pkl",'wb')
        pickle.dump(raw_result_dict,f)
        f.close()

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






def load_model(resume,model,optimizer):


    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    for i,v in checkpoint["state_dict"].items():
        print(i,v.size())
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
        model = select_model(eval_config,arch_name,vocab_size,word_embeddings)

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

        test_loss, test_acc,test_f1_score,test_f1_score_w,test_top3_acc= eval_model(model, test_iter,loss_fn,eval_config,arch_name,mode)
        print(test_acc)
        print(f'Top3 Acc: {test_top3_acc:.3f}%, F1 Score: {test_f1_score:.3f}, F1 Score W: {test_f1_score_w:.3f}')


    elif mode == "explain":

        print(f'Train Acc: {log["train_acc"]:.3f}%, Valid Acc: {log["valid_acc"]:.3f}%, Test Acc: {log["test_acc"]:.3f}%')

        eval_config.confusion = False
        eval_config.per_class = False

        ## explaining
        eval_model(model, test_iter,loss_fn,eval_config,arch_name,mode,explain=True)
