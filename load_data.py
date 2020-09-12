# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import config
from transformers import BertTokenizer,AutoTokenizer
import random


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

def load_dataset_glove(test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    
    if config.tokenizer == "whitespace":
        tokenize = lambda x: x.split()
    elif config.tokenizer == "wordpiece+punkt":
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        tokenize = lambda x: tokenizer.tokenize(x)

    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=False, batch_first=True, fix_length=config.max_len)
    LABEL = data.LabelField()

    if config.input_type == "speaker":
        fields = {'speaker_utterance':('utterance', TEXT),'emotion':('emotion', LABEL)}
    elif config.input_type == "speaker+listener": 
        fields = {'utterance_data':('utterance', TEXT),'emotion':('emotion', LABEL)}
    elif config.input_type == "prompt":
        fields = {'prompt':('utterance', TEXT),'emotion':('emotion', LABEL)}


    train_data, valid_data, test_data = data.TabularDataset.splits(path = './.data/empathetic_dialogues',train = 'train.csv',validation = 'valid.csv',test = 'test.csv',format = 'csv',fields = fields,skip_header = False)

    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=200))
    
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=config.batch_size, sort_key=lambda x: len(x.utterance), repeat=False, shuffle=True)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter


def load_dataset_bert(test_sen=None):

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    LABEL = data.LabelField(use_vocab=False)
    TEXT = data.Field(sequential=True,use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=config.max_len, pad_token=PAD_INDEX, unk_token=UNK_INDEX) # if use_vocab is false then the tokenized output should be numerical which is why we use encode
    

    fields = [('speaker_data', TEXT),('emotion', LABEL),('utterance_data', TEXT),('prompt', TEXT)] 

 
    train_data, valid_data, test_data = data.TabularDataset.splits(path = './.data/empathetic_dialogues',train = 'train.csv',validation = 'valid.csv',test = 'test.csv',format = 'csv',fields = fields,skip_header=True)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=config.batch_size, sort_key=lambda x: len(x.utterance_data), repeat=False, shuffle=True)

    return train_iter, valid_iter, test_iter
    
