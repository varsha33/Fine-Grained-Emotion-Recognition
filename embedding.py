import os
import sys
import numpy as np
import pandas as pd
import pickle
import random

## torch packages
import torch
from torch.nn import functional as F
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from torch.utils.data import Dataset
from transformers import BertTokenizer,AutoTokenizer

## custom
import config

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False



def get_glove_embedding():

    
    if config.tokenizer == "whitespace":
        tokenize = lambda x: x.split()
    elif config.tokenizer == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        tokenize = lambda x: tokenizer.tokenize(x)

    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=False, batch_first=True, fix_length=config.max_seq_len)
    LABEL = data.LabelField(use_vocab=False)


    fields = [('speaker_utterance', TEXT),('emotion', LABEL),('utterance_data', TEXT),('prompt', TEXT)]

    train_data, valid_data, test_data = data.TabularDataset.splits(path = './.data/empathetic_dialogues',train = 'train.csv',validation = 'valid.csv',test = 'test.csv',format = 'csv',fields = fields,skip_header = True)

    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=200))

    word_embeddings = TEXT.vocab.vectors

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=config.batch_size, sort_key=lambda x: len(x.utterance_data), repeat=False, shuffle=True)

    vocab_size = len(TEXT.vocab)

    return  vocab_size, word_embeddings, train_iter, valid_iter, test_iter