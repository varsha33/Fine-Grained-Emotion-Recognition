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
# from embedding import get_glove_embedding


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
        item["emotion"] = self.data["emotion"][index]

        return item

    def __len__(self):
        return len(self.data["emotion"])


class GoEmo_dataset(Dataset):

    def __init__(self,data):

        self.data = data


    def __getitem__(self, index):


        item = {}
        ## converting to ED dataset terminology for ease
        item["utterance_data_str"] = self.data["cause"][index]
        item["utterance_data"] = torch.LongTensor(self.data["tokenized_cause"][index])

        item["arousal_data"] = torch.Tensor(self.data["arousal_data"][index])
        item["valence_data"] = torch.Tensor(self.data["valence_data"][index])
        item["dom_data"] = torch.Tensor(self.data["dom_data"][index])
        item["emotion"] = torch.Tensor(self.data["emotion"][index])

        return item

    def __len__(self):
        return len(self.data["emotion"])

class SemEval_dataset(Dataset):

    def __init__(self,data):

        self.data = data


    def __getitem__(self, index):


        item = {}
        ## converting to ED dataset terminology for ease
        item["utterance_data_str"] = self.data["cause"][index]
        item["utterance_data"] = torch.LongTensor(self.data["tokenized_cause"][index])

        item["arousal_data"] = torch.Tensor(self.data["arousal_data"][index])
        item["valence_data"] = torch.Tensor(self.data["valence_data"][index])
        item["dom_data"] = torch.Tensor(self.data["dom_data"][index])
        item["emotion"] = torch.Tensor(self.data["emotion"][index])

        return item

    def __len__(self):
        return len(self.data["emotion"])


def collate_fn(data):

    def merge(sequences,N=None,lexicon=False):
        lengths = [len(seq) for seq in sequences]
        if N == None:
            N = max(lengths)
            if N  > 512 : # no conversation goes beyond 512.
                N=512
        if lexicon:
            padded_seqs = torch.zeros(len(sequences),N) ## padding index 0, but float
        else:
            padded_seqs = torch.zeros(len(sequences),N).long() ## padding index 0

        attention_mask = torch.zeros(len(sequences),N).long()
        for i, seq in enumerate(sequences):
            if not torch.is_tensor(seq):
                seq = torch.LongTensor(seq)
            if lengths[i] < 512:
                end = lengths[i]
            else:
                end = 512
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths



    data.sort(key=lambda x: len(x["utterance_data"]), reverse=True) ## sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]


    ## input
    input_batch,input_attn_mask, input_lengths = merge(item_info['utterance_data'])

    ainput_batch,_,ainput_lengths = merge(item_info['arousal_data'],N=512,lexicon=True) # not used here, 600 just to make up to xlnet tokenization
    vinput_batch,_,vinput_lengths = merge(item_info['valence_data'],N=512,lexicon=True) # not used here
    dinput_batch,_,dinput_lengths = merge(item_info['dom_data'],N=512,lexicon=True) # not used here


    d = {}


    d["arousal_data"] = ainput_batch
    d["valence_data"] = vinput_batch
    d["dom_data"] = dinput_batch

    d["utterance_data"] = input_batch
    d["utterance_data_attn_mask"] = input_attn_mask


    d["emotion"] = item_info["emotion"]
    d["utterance_data_str"] = item_info['utterance_data_str']

    return d

def collate_fn_bilstm(data):

    def merge(sequences,N=None,lexicon=False):
        lengths = [len(seq) for seq in sequences]
        if N == None:
            N = max(lengths)
            if N  > 512 : # no conversation goes beyond 512.
                N=512
        if lexicon:
            padded_seqs = torch.zeros(len(sequences),N) ## padding index 0, but float
        else:
            padded_seqs = torch.zeros(len(sequences),N).long() ## padding index 0

        attention_mask = torch.zeros(len(sequences),N).long()
        for i, seq in enumerate(sequences):
            if not torch.is_tensor(seq):
                seq = torch.LongTensor(seq)
            if lengths[i] < 512:
                end = lengths[i]
            else:
                end = 512
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i,:end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths



    data.sort(key=lambda x: len(x["utterance_data"]), reverse=True) ## sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]


    ## input
    input_batch,input_attn_mask, input_lengths = merge(item_info['utterance_data'])

    ainput_batch,_,ainput_lengths = merge(item_info['arousal_data'],lexicon=True) # not used here, 600 just to make up to xlnet tokenization
    vinput_batch,_,vinput_lengths = merge(item_info['valence_data'],lexicon=True) # not used here
    dinput_batch,_,dinput_lengths = merge(item_info['dom_data'],lexicon=True) # not used here


    d = {}


    d["arousal_data"] = ainput_batch
    d["valence_data"] = vinput_batch
    d["dom_data"] = dinput_batch

    d["utterance_data"] = input_batch
    d["utterance_data_attn_mask"] = input_attn_mask


    d["emotion"] = item_info["emotion"]
    d["utterance_data_str"] = item_info['utterance_data_str']

    return d


def get_dataloader(batch_size,tokenizer,dataset,arch_name):

    if dataset == "ed":

        with open('./.preprocessed_data/mid_dataset_preproc.p', "rb") as f:
            [data_train, data_valid, data_test, vocab_size, word_embeddings] = pickle.load(f)
        f.close()


        if "bilstm" in arch_name:
            collate = collate_fn_bilstm
        else:
            collate = collate_fn


        dataset = ED_dataset(data_train)
        train_iter  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True,collate_fn=collate,num_workers=0)

        # For validation and testing batch_size is 1
        dataset = ED_dataset(data_valid)
        valid_iter  = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False,collate_fn=collate,num_workers=0)

        dataset = ED_dataset(data_test)
        test_iter  = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False,collate_fn=collate,num_workers=0)

        return vocab_size,word_embeddings, train_iter, valid_iter, test_iter

    if dataset == "goemotions":


        with open('./.preprocessed_data/goemotions_preprocessed_bert.pkl', "rb") as f:
            data_dict = pickle.load(f)
        f.close()

        if "bilstm" in arch_name:
            collate = collate_fn_bilstm
        else:
            collate = collate_fn


        dataset = GoEmo_dataset(data_dict["train"])
        train_iter  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True,collate_fn=collate,num_workers=0)

        # For validation and testing batch_size is 1
        dataset = GoEmo_dataset(data_dict["valid"])
        valid_iter  = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False,collate_fn=collate,num_workers=0)

        dataset = GoEmo_dataset(data_dict["test"])
        test_iter  = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False,collate_fn=collate,num_workers=0)


        return 0,0, train_iter, valid_iter, test_iter

    if dataset == "semeval":

        with open('./.preprocessed_data/semeval_preprocessed_bert.pkl', "rb") as f:
            data_dict = pickle.load(f)
        f.close()

        if "bilstm" in arch_name:
            collate = collate_fn_bilstm
        else:
            collate = collate_fn

        dataset = SemEval_dataset(data_dict["train"])
        train_iter  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True,collate_fn=collate,num_workers=0)

        # For validation and testing batch_size is 1
        dataset = SemEval_dataset(data_dict["valid"])
        valid_iter  = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False,collate_fn=collate,num_workers=0)

        dataset = SemEval_dataset(data_dict["test"])
        test_iter  = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False,collate_fn=collate,num_workers=0)


        return 0,0, train_iter, valid_iter, test_iter
