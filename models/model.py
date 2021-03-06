import numpy as np
import random
import math

## torch packages
import torch
import torch.nn as nn
from torch.utils import checkpoint

from transformers import ElectraForSequenceClassification,BertForSequenceClassification
from torch.autograd import Variable
from torch.nn import functional as F
import time


class KEA_BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size):
        super(KEA_BERT, self).__init__()


        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size)

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.a = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec
        self.v = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec
        self.d = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec

        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_size,384)
        self.label = nn.Linear(384,output_size)

    def attention_net(self,input_matrix, final_output):

        hidden = final_output

        attn_weights = torch.bmm(input_matrix, hidden.unsqueeze(2)).squeeze(2)

        soft_attn_weights = F.softmax(attn_weights, 1)

        new_hidden_state = torch.bmm(input_matrix.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state


    def forward(self,text,attn_mask):

        input = self.encoder(text[0],attn_mask,output_hidden_states=True,return_dict=True).hidden_states[-1]

        cls_input = input[:,0,:]


        arousal_encoder = F.relu(self.a(text[1]))
        valence_encoder = F.relu(self.v(text[2]))
        dom_encoder = F.relu(self.d(text[3]))

        input = torch.cat((input,arousal_encoder.unsqueeze(1),valence_encoder.unsqueeze(1),dom_encoder.unsqueeze(1)),dim=1)
        # input = self.layernorm(input)
        output = self.attention_net(input,cls_input)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        logits = self.label(output)

        return logits



class KEA_ELECTRA(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size):
        super(KEA_ELECTRA, self).__init__()


        options_name = "google/electra-base-discriminator"
        self.encoder = ElectraForSequenceClassification.from_pretrained(options_name,num_labels=output_size)

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.a = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec
        self.v = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec
        self.d = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec

        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_size,384)
        self.label = nn.Linear(384,output_size)

    def attention_net(self,input_matrix, final_output):

        hidden = final_output

        attn_weights = torch.bmm(input_matrix, hidden.unsqueeze(2)).squeeze(2)

        soft_attn_weights = F.softmax(attn_weights, 1)

        new_hidden_state = torch.bmm(input_matrix.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state


    def forward(self,text,attn_mask):

        input = self.encoder(text[0],attn_mask,output_hidden_states=True,return_dict=True).hidden_states[-1]

        cls_input = input[:,0,:]


        arousal_encoder = F.relu(self.a(text[1]))
        valence_encoder = F.relu(self.v(text[2]))
        dom_encoder = F.relu(self.d(text[3]))

        input = torch.cat((input,arousal_encoder.unsqueeze(1),valence_encoder.unsqueeze(1),dom_encoder.unsqueeze(1)),dim=1)
        output = self.attention_net(input,cls_input)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        logits = self.label(output)

        return logits


class KEA_Electra_Word_level(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size):
        super(KEA_Electra_Word_level, self).__init__()


        options_name = "google/electra-base-discriminator"
        self.encoder = ElectraForSequenceClassification.from_pretrained(options_name,num_labels=output_size)

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.bilstm = nn.LSTM(hidden_size+3,int(hidden_size/2), dropout=0.2, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_size,384)
        self.label = nn.Linear(384,output_size)

    def attention_net(self,input_matrix, final_output):

        hidden = final_output

        attn_weights = torch.bmm(input_matrix, hidden.unsqueeze(2)).squeeze(2)

        soft_attn_weights = F.softmax(attn_weights, 1)

        new_hidden_state = torch.bmm(input_matrix.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state


    def forward(self,text,attn_mask):

        input = self.encoder(text[0],attn_mask,output_hidden_states=True,return_dict=True).hidden_states[-1]

        cls_input = input[:,0,:]
        # print(input.size())
        # print(text[1][:,:input.size()[1]].size())
        seq_len = input.size()[1]
        word_query = torch.cat((input,text[1][:,:seq_len].unsqueeze(2),text[2][:,:seq_len].unsqueeze(2),text[3][:,:seq_len].unsqueeze(2)),dim=2)


        h_0 = Variable(torch.zeros(2, input.size()[0],int(self.hidden_size/2)).cuda())
        c_0 = Variable(torch.zeros(2, input.size()[0],int(self.hidden_size/2)).cuda())


        word_query = word_query.permute(1, 0, 2)

        output, (h_n, c_n) = self.bilstm(word_query, (h_0, c_0))

        output = output.permute(1, 0, 2)
        output = self.attention_net(output,cls_input)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        logits = self.label(output)

        return logits

class KEA_Bert_Word_level(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size):
        super(KEA_Bert_Word_level, self).__init__()


        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size)

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.bilstm = nn.LSTM(hidden_size+3,int(hidden_size/2), dropout=0.2, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_size,384)
        self.label = nn.Linear(384,output_size)

    def attention_net(self,input_matrix, final_output):

        hidden = final_output

        attn_weights = torch.bmm(input_matrix, hidden.unsqueeze(2)).squeeze(2)

        soft_attn_weights = F.softmax(attn_weights, 1)

        new_hidden_state = torch.bmm(input_matrix.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state


    def forward(self,text,attn_mask):

        input = self.encoder(text[0],attn_mask,output_hidden_states=True,return_dict=True).hidden_states[-1]

        cls_input = input[:,0,:]
        seq_len = input.size()[1]

        word_query = torch.cat((input,text[1][:,:seq_len].unsqueeze(2),text[2][:,:seq_len].unsqueeze(2),text[3][:,:seq_len].unsqueeze(2)),dim=2)


        h_0 = Variable(torch.zeros(2, input.size()[0],int(self.hidden_size/2)).cuda())
        c_0 = Variable(torch.zeros(2, input.size()[0],int(self.hidden_size/2)).cuda())

        word_query = word_query.permute(1, 0, 2)

        output, (h_n, c_n) = self.bilstm(word_query, (h_0, c_0))

        output = output.permute(1, 0, 2)
        output = self.attention_net(output,cls_input)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        logits = self.label(output)

        return logits






