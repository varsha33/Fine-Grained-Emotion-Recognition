import numpy as np
import random
import math

## torch packages
import torch
import torch.nn as nn
from torch.utils import checkpoint

from transformers import ElectraForSequenceClassification,BertForSequenceClassification

from transformers.modeling_bert import BertModel,BertEmbeddings,BertPooler,BertEncoder

from torch.autograd import Variable
from torch.nn import functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

import time


class BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size)


    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text,attn_mask,output_hidden_states=True,return_dict=True) # no labels provided, output attention and output hidden states = False

        return text_fea.logits


class KEA_BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
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


class Electra_BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(Electra_BERT, self).__init__()

        options_name = "google/electra-base-discriminator"
        self.encoder = ElectraForSequenceClassification.from_pretrained(options_name,num_labels=output_size)


    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text,attn_mask,output_hidden_states=True,return_dict=True) # no labels provided, output attention and output hidden states = False

        return text_fea.logits


class KEA_Electra(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(KEA_Electra, self).__init__()


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
        # input = self.layernorm(input)
        output = self.attention_net(input,cls_input)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        logits = self.label(output)

        return logits

class Knowledge_baseline(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(Knowledge_baseline, self).__init__()


        options_name = "google/electra-base-discriminator"
        self.encoder = ElectraForSequenceClassification.from_pretrained(options_name,num_labels=output_size)

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.a = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec
        self.v = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec
        self.d = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size*4,hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2,384)
        self.label = nn.Linear(384,output_size)



    def forward(self,text,attn_mask):

        input = self.encoder(text[0],attn_mask,output_hidden_states=True,return_dict=True).hidden_states[-1]

        cls_input = input[:,0,:]


        arousal_encoder = F.relu(self.a(text[1]))
        valence_encoder = F.relu(self.v(text[2]))
        dom_encoder = F.relu(self.d(text[3]))


        input = torch.cat((cls_input,arousal_encoder,valence_encoder,dom_encoder),dim=1)
        # input = self.layernorm(input)

        output = F.relu(self.fc1(input))
        output = self.dropout(output)
        output = F.relu(self.fc2(output))

        logits = self.label(output)

        return logits


class KBERT_BiLSTM(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(KBERT_BiLSTM, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size)


        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm_hidden_size = 384
        self.bilstm = nn.LSTM(hidden_size+3,self.lstm_hidden_size, dropout=0.2, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_size,384)
        self.label = nn.Linear(384,output_size)
        self.fc_layer = nn.Linear(2*384, 256)
        self.fcdropout = nn.Dropout(p=0.2)
        self.label = nn.Linear(256, output_size)


    def forward(self,text,attn_mask):

        input = self.encoder(text[0],attn_mask,output_hidden_states=True,return_dict=True).hidden_states[-1]

        VAD = torch.cat((text[1].unsqueeze(2),text[2].unsqueeze(2),text[3].unsqueeze(2)),dim=2)


        input = torch.cat((input,VAD),dim=2)
        h_0 = Variable(torch.zeros(2, input.size()[0], self.lstm_hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, input.size()[0], self.lstm_hidden_size).cuda())

        input = input.permute(1, 0, 2)

        output, (h_n, c_n) = self.bilstm(input, (h_0, c_0))

        fc_out = F.relu(self.fc_layer(h_n.view(-1,384*2)))
        fc_out = self.fcdropout(fc_out)
        logits = self.label(fc_out)


        return logits


class KBERT_BiLSTMwSA(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(KBERT_BiLSTMwSA, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size)


        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm_hidden_size = 384
        self.bilstm = nn.LSTM(hidden_size+3,self.lstm_hidden_size, dropout=0.2, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

        self.W_s1 = nn.Linear(2*self.lstm_hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30*2*self.lstm_hidden_size, 2000)
        self.label = nn.Linear(2000, output_size)


        # self.fc_layer = nn.Linear(2*384, 256)
        # self.fcdropout = nn.Dropout(p=0.2)
        # self.label = nn.Linear(256, output_size)

    def attention_net(self, lstm_output):

        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
        encoding of the input sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
        the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
        connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
        pos & neg.

        Arguments
        ---------

        lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
        ---------

        Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
                  attention to different parts of the input sentence.

        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                      attn_weight_matrix.size() = (batch_size, 30, num_seq)

        """
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self,text,attn_mask):

        input = self.encoder(text[0],attn_mask,output_hidden_states=True,return_dict=True).hidden_states[-1]

        VAD = torch.cat((text[1].unsqueeze(2),text[2].unsqueeze(2),text[3].unsqueeze(2)),dim=2)


        input = torch.cat((input,VAD),dim=2)
        h_0 = Variable(torch.zeros(2, input.size()[0], self.lstm_hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, input.size()[0], self.lstm_hidden_size).cuda())

        input = input.permute(1, 0, 2)

        output, (h_n, c_n) = self.bilstm(input, (h_0, c_0))

        # fc_out = F.relu(self.fc_layer(h_n.view(-1,384*2)))
        # fc_out = self.fcdropout(fc_out)
        # logits = self.label(fc_out)


        output = output.permute(1, 0, 2)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        # h_n.size() = (1, batch_size, hidden_size)
        # c_n.size() = (1, batch_size, hidden_size)
        attn_weight_matrix = self.attention_net(output)
        # attn_weight_matrix.size() = (batch_size, r, num_seq)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)
        # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
        # self.dropout = nn.Dropout(p=0.3)
        logits = self.label(fc_out)
        # logits.size() = (batch_size, output_size)

        return logits
