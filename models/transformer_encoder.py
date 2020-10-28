
import numpy as np
import random

## torch packages
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights,nhead=2,nlayers=1,dropout=0.5):

        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer


        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.weights = weights
        self.input_mask = None
        self.dropout = dropout
        self.nhead = nhead
        self.nlayers = nlayers
        self.pos_encoder = PositionalEncoding(embedding_length,dropout)
        encoder_layers = TransformerEncoderLayer(embedding_length, nhead, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
        self.label = nn.Linear(embedding_length, output_size)

        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_sentences,attn_mask):

        input_sentences = input_sentences.transpose(0,1) # as seq_len is the first dimension expected (seq_len,batch_size,embedding_len)

        if self.input_mask is None or self.input_mask.size(0) != len(input_sentences):
            device = input_sentences.device
            mask = self._generate_square_subsequent_mask(len(input_sentences)).to(device)
            self.input_mask = mask


        input = self.word_embeddings(input_sentences) * math.sqrt(self.embedding_length)

        input = self.pos_encoder(input)
        # (seq_len, batch_size,embedding_len) , mask: (seq_len, embedding_len)
        output = self.transformer_encoder(input, self.input_mask)
        # (seq_len, batch_size,embedding_len)
        output = output.transpose(0,1)
        # (batch_size,seq_len,embedding_len)
        output = output.permute(0, 2, 1) # y.size() = (batch_size, embedding_len, seq_len)
        output = F.max_pool1d(output, output.size()[2]) # y.size() = (batch_size, embedding_len, 1)
        output = output.squeeze(2) # (batch_size, embedding_len)

        # output = output[:,0,:] # only the first output

        return self.label(output)
