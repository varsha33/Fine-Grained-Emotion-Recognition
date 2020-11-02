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


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)




class ED_dataset(Dataset):

    def __init__(self,data):

        self.data = data


    def __getitem__(self, index):


        item = {}

        item["utterance_data_str"] = self.data["utterance_data_str"][index]
        item["utterance_data"] = torch.LongTensor(self.data["utterance_data"][index])

        item["arousal_data"] = torch.Tensor(self.data["arousal_data"][index])
        item["valence_data"] = torch.Tensor(self.data["valence_data"][index])
        item["dom_data"] = torch.Tensor(self.data["dom_data"][index])


        item["speaker_data"] = torch.LongTensor(self.data["speaker_data"][index])
        item["listener_data"] = torch.LongTensor(self.data["listener_data"][index])
        item["speaker_idata"] = torch.LongTensor(self.data["speaker_idata"][index])
        item["listener_idata"] = torch.LongTensor(self.data["listener_idata"][index])


        item["utterance_data_list"] = self.data["utterance_data_list"][index]
        item["turn_data"] = self.data["turn_data"][index]

        item["glove_data"] = torch.LongTensor(self.data["glove_data"][index])

        item["emotion"] = self.data["emotion"][index]

        return item

    def __len__(self):
        return len(self.data["emotion"])



def collate_fn(data):

    def merge(sequences,N=None,lexicon=False):
        lengths = [len(seq) for seq in sequences]
        if N == None:
            N = max(lengths)
        if lexicon:
            padded_seqs = torch.zeros(len(sequences),N) ## padding index 0, but float
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

    u_list_batch,u_list_attn_mask,u_list_lengths = merge_utterance_level(item_info["utterance_data_list"],8,153) #153,8 :max number found
    tu_list_batch,tu_list_attn_mask,tu_list_lengths = merge_utterance_level(item_info["turn_data"],4,306) #153*2,4 :max number found

    input_batch,input_attn_mask, input_lengths = merge(item_info['utterance_data'])

    ainput_batch,_,ainput_lengths = merge(item_info['arousal_data'],N=512,lexicon=True)
    vinput_batch,_,vinput_lengths = merge(item_info['valence_data'],N=512,lexicon=True)
    dinput_batch,_,dinput_lengths = merge(item_info['dom_data'],N=512,lexicon=True)

    sinput_batch, sinput_attn_mask, sinput_lengths = merge(item_info['speaker_data'])
    linput_batch, linput_attn_mask, linput_lengths = merge(item_info['listener_data'])

    si_input_batch, _ , si_input_lengths = merge(item_info['speaker_idata'])
    li_input_batch, _ ,li_input_lengths = merge(item_info['listener_idata'])

    glove_input_batch,_,glove_input_lengths = merge(item_info["glove_data"],N=512)

    d = {}
    d["utterance_data_list"] = u_list_batch
    d["utterance_data_list_attn_mask"] = u_list_attn_mask


    d["arousal_data"] = ainput_batch
    d["valence_data"] = vinput_batch
    d["dom_data"] = dinput_batch

    d["turn_data"] = tu_list_batch
    d["turn_data_attn_mask"] = tu_list_attn_mask

    d["utterance_data"] = input_batch
    d["utterance_data_attn_mask"] = input_attn_mask

    d["speaker_data"] = sinput_batch
    d["speaker_data_attn_mask"] = sinput_attn_mask
    d["listener_data"] = linput_batch
    d["listener_data_attn_mask"] = linput_attn_mask

    d["speaker_idata"] = si_input_batch
    d["listener_idata"] = li_input_batch

    ##TODO
    # d["speaker_idata_attn_mask"] = si_input_attn_mask
    # d["listener_idata_attn_mask"] = li_input_attn_mask

    d["emotion"] = item_info["emotion"]
    d["utterance_data_str"] = item_info['utterance_data_str']

    d["glove_data"] = glove_input_batch

    return d


def get_dataloader(batch_size,tokenizer,embedding_type,arch_name):

    if embedding_type == "bert" or embedding_type == "glove+bert":

        if tokenizer == "bert":
            with open('./.preprocessed_data/dataset_preproc.p', "rb") as f:
                [data_train, data_valid, data_test, vocab_size, word_embeddings] = pickle.load(f)
            f.close()

        dataset = ED_dataset(data_train)
        train_iter  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True,collate_fn=collate_fn,num_workers=0)

        # For validation and testing batch_size is 1
        dataset = ED_dataset(data_valid)
        valid_iter  = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False,collate_fn=collate_fn,num_workers=0)

        dataset = ED_dataset(data_test)
        test_iter  = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False,collate_fn=collate_fn,num_workers=0)

        return vocab_size,word_embeddings, train_iter, valid_iter, test_iter

