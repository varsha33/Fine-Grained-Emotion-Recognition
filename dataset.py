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
torch.set_printoptions(threshold=1000)
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
        item["turnwise_data"] = self.data["turnwise_data"][index]
        item["utterance_data_str"] = self.data["utterance_data_str"][index]
        item["sep_data"] = torch.LongTensor(self.data["sep_data"][index])
        item["arousal_sep"] = torch.Tensor(self.data["arousal_sep"][index])
        item["valence_sep"] = torch.Tensor(self.data["valence_sep"][index])

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
            attention_mask = torch.zeros(len(sequences),N).long()
        else:
            padded_seqs = torch.zeros(len(sequences),N).long() ## padding index 0
            attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            if not torch.is_tensor(seq):
                seq = torch.LongTensor(seq)
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    def merge_utterance_level(utterances,uN=None,N=None):

        u_lengths = [len(utterance) for utterance in utterances]
        if uN == None:
            uN = max(u_lengths)
        padded_utterances = torch.zeros(len(utterances),uN,N).long()
        utterance_attn_mask = torch.zeros(len(utterances),uN,N).long()

        for i,utterance in enumerate(utterances):
            end = len(utterance)
            process_i = merge(utterance,N) ## here utterance is list of utterances
            padded_utterances[i,:end,:] = process_i[0]
            utterance_attn_mask[i,:end,:] = process_i[1]

        return padded_utterances,utterance_attn_mask,u_lengths

    data.sort(key=lambda x: len(x["utterance_data"]), reverse=True) ## sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]


    ## input

    u_list_batch,u_list_attn,u_list_lengths = merge_utterance_level(item_info["utterance_data_list"],8,153) #153:number found
    tu_list_batch,tu_list_attn,tu_list_lengths = merge_utterance_level(item_info["turnwise_data"],4,306) #153:number found

    input_batch,input_attn, input_lengths = merge(item_info['utterance_data'])
    ainput_batch, ainput_attn,ainput_lengths = merge(item_info['arousal_utterance'],arousal=True)
    sinput_batch,sinput_attn, sinput_lengths = merge(item_info['speaker_data'])
    linput_batch, linput_attn,linput_lengths = merge(item_info['listener_data'])
    si_input_batch,_, si_input_lengths = merge(item_info['speaker_idata'])
    li_input_batch,_,li_input_lengths = merge(item_info['listener_idata'])
    sep_input_batch,sep_input_attn,sep_input_lengths = merge(item_info["sep_data"])
    asep_input_batch,asep_input_attn,asep_input_lengths = merge(item_info["arousal_sep"],N=512,arousal=True)
    vsep_input_batch,vsep_input_attn,vsep_input_lengths = merge(item_info["valence_sep"],N=512,arousal=True)

    d = {}
    d["utterance_data_list"] = u_list_batch
    d["utterance_data_list_attn"] = u_list_attn
    d["sep_data"] = sep_input_batch
    d["sep_data_attn"] =sep_input_attn
    d["arousal_sep"] = asep_input_batch
    d["valence_sep"] = vsep_input_batch
    d["arousal_sep_attn"] =asep_input_attn
    d["turnwise_data"] = tu_list_batch
    d["turnwise_data_attn"] = tu_list_attn
    d["utterance_data"] = input_batch
    d["utterance_data_attn"] = input_attn
    d["arousal_utterance"] = ainput_batch
    d["speaker_data"] = sinput_batch
    d["speaker_data_attn"] = sinput_attn
    d["listener_data"] = linput_batch
    d["listener_data_attn"] = linput_attn
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

