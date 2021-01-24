import os
import sys
import numpy as np
import pandas as pd
import pickle
import random
import numpy as np

## torch packages
import torch
from torch.nn import functional as F
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from torch.utils.data import Dataset
from transformers import BertTokenizer,AutoTokenizer

## custom
# import config


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
TEXT = data.Field(sequential=True, tokenize=tokenizer.tokenize, lower=True, include_lengths=False, batch_first=True,init_token='<sos>',eos_token='<eos>')
fields = {'utterance_data':('utterance', TEXT)}
train_data, valid_data, test_data = data.TabularDataset.splits(path = '/content/gdrive/My Drive/emotion_recognition/.data/empathetic_dialogues',train = 'train.csv',validation = 'valid.csv',test = 'test.csv',format = 'csv',fields = fields,skip_header = False)

TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=200))


def get_glove_embedding():
    return len(TEXT.vocab),TEXT.vocab.vectors

def get_glove_vec(utterance):

    preprocessed_text = TEXT.preprocess(utterance)

    return [TEXT.vocab[x] for x in preprocessed_text]
