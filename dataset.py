## Taken and modified from https://github.com/HLTCHKUST/MoEL.git

import os
import sys
import numpy as np
import random
import pandas as pd
import pickle

## torch packages
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import BertTokenizer,AutoTokenizer

## custom
from embedding import get_glove_embedding

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = False

class ED_dataset(Dataset):

    def __init__(self,data):
        
        self.data = data
        

    def __getitem__(self, index):

    
        item = {}
        
        item["utterance_data"] = torch.LongTensor(self.data["utterance_data"][index])
        item["arousal_utterance"] = torch.Tensor(self.data["arousal_utterance"][index])
        item["speaker_data"] = torch.LongTensor(self.data["speaker_data"][index])
        item["listener_data"] = torch.LongTensor(self.data["listener_data"][index])
        item["speaker_idata"] = torch.LongTensor(self.data["speaker_idata"][index])
        item["listener_idata"] = torch.LongTensor(self.data["listener_idata"][index])
        item["emotion"] = self.data["emotion"][index]
        item["utterance_data_list"] = self.data["utterance_data_list"][index]
        item["utterance_data_str"] = self.data["utterance_data_str"][index]

        return item
    
    def __len__(self):
        return len(self.data["emotion"])



def collate_fn(data):

    def merge(sequences,N=None,arousal=False):
        lengths = [len(seq) for seq in sequences]
        if N == None:
            N = max(lengths)

        
        if arousal:
            padded_seqs = torch.zeros(len(sequences),N) ## padding index 0.5 (neutral)
            padded_seqs = torch.add(padded_seqs, 0.5)
        else:
            padded_seqs = torch.zeros(len(sequences),N).long() ## padding index 0

        for i, seq in enumerate(sequences):
            if not torch.is_tensor(seq):
                seq = torch.LongTensor(seq)
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 

    def merge_utterance_level(utterances,uN=None,N=None):

        u_lengths = [len(utterance) for utterance in utterances]
        if uN == None:
            uN = max(u_lengths)
        padded_utterances = torch.zeros(len(utterances),uN,N).long()

        for i,utterance in enumerate(utterances):
            end = len(utterance)
            padded_utterances[i,:end,:] = merge(utterance,N)[0]  #36 is the number I found out by taking the max of all the sequences
        
        return padded_utterances,u_lengths

    data.sort(key=lambda x: len(x["utterance_data"]), reverse=True) ## sort by source seq
   
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]


    ## input
    
    u_list_batch,u_list_lengths = merge_utterance_level(item_info["utterance_data_list"],8,153) #153:number found
    
    input_batch, input_lengths = merge(item_info['utterance_data'],N=512)
    ainput_batch, ainput_lengths = merge(item_info['arousal_utterance'],N=512,arousal=True)
    sinput_batch, sinput_lengths = merge(item_info['speaker_data'])
    linput_batch, linput_lengths = merge(item_info['listener_data'])
    si_input_batch, si_input_lengths = merge(item_info['speaker_idata'])
    li_input_batch,li_input_lengths = merge(item_info['listener_idata'])
    

    d = {}
    d["utterance_data_list"] = u_list_batch
    d["utterance_data"] = input_batch
    d["arousal_utterance"] = ainput_batch
    d["speaker_data"] = sinput_batch
    d["listener_data"] = linput_batch
    d["speaker_idata"] = si_input_batch
    d["listener_idata"] = li_input_batch
    d["emotion"] = item_info["emotion"]
    d["utterance_data_str"] = item_info['utterance_data_str']
    return d 


def get_dataloader(batch_size,tokenizer,embedding_type,arch_name):

    if embedding_type == "bert":

        if tokenizer == "bert":
            with open('./.preprocessed_data/dataset_preproc.p', "rb") as f:
                [data_train,data_test, data_valid] = pickle.load(f)
            f.close()

        elif tokenizer == "distil_bert":
            with open('./preprocessed_data/distil_dataset_preproc.p', "rb") as f:
                [data_train,data_test, data_valid] = pickle.load(f)
            f.close()

        dataset = ED_dataset(data_train)
        train_iter  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True,collate_fn=collate_fn,num_workers=0)
        
        dataset = ED_dataset(data_test)
        test_iter  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True,collate_fn=collate_fn,num_workers=0)
           
        dataset = ED_dataset(data_valid)
        valid_iter  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True,collate_fn=collate_fn,num_workers=0)
        
        return None, None, train_iter, valid_iter, test_iter  #vocab size and embedding size is not required for this

    elif embedding_type == "glove":

        return get_glove_embedding()

