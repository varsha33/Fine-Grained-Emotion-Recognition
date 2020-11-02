import numpy as np
import random
import math

## torch packages
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification,BertModel,AlbertModel,DistilBertModel,ElectraModel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.RCNN import RCNN

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
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)
        self.label = nn.Linear(hidden_size,output_size)

    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text,attn_mask,output_hidden_states=True,return_dict=True) # no labels provided, output attention and output hidden states = False
        input = text_fea.hidden_states[-1][:,0,:]
        output = self.label(input)
        return output

class _BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(_BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text,attn_mask,output_hidden_states=True,return_dict=True) # no labels provided, output attention and output hidden states = False

        return text_fea


class a_BERT(nn.Module):

    def __init__(self,resume_path,bert_base_model,batch_size,output_size,hidden_size,grad_check,freeze):
        super(a_BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

        self.pooler = nn.Linear(hidden_size,hidden_size)
        self.pooler_dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(512,hidden_size)
        self.fc2 = nn.Linear(2*hidden_size,384)
        self.layernorm = nn.LayerNorm(2*hidden_size)
        self.label = nn.Linear(384,output_size)

    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text[0],attn_mask,output_hidden_states=True,return_dict=True) # no labels provided, output attention and
        # print(text[1].cpu().numpy())
        output = torch.tanh(self.pooler(text_fea.hidden_states[-1][:,0,:]))
        output = self.pooler_dropout(output)
        arousal_encoder = F.relu(self.fc1(text[1]))
        output = torch.cat((output,arousal_encoder),dim=1)
        output = self.layernorm(output)
        output = torch.tanh(self.fc2(output))
        logits = self.label(output)
        return logits


class va_BERT(nn.Module):

    def __init__(self,resume_path,bert_base_model,batch_size,output_size,hidden_size,grad_check,freeze):
        super(va_BERT, self).__init__()

        if freeze:
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            bert_base_model.load_state_dict(checkpoint['state_dict'])
            for param in bert_base_model.parameters():
                param.require_grad = False

            self.encoder = bert_base_model
        else:
            self.encoder = bert_base_model


        self.pooler = nn.Linear(hidden_size,hidden_size)
        self.pooler_dropout = nn.Dropout(0.1)
        self.a = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec
        self.v = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec


        self.layernorm = nn.LayerNorm(3*hidden_size)
        self.fc1 = nn.Linear(3*hidden_size,hidden_size)
        self.fc1_layernorm = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size,384)
        self.label = nn.Linear(384,output_size)


    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text[0],attn_mask) # no labels provided, output attention and
        output = torch.tanh(self.pooler(text_fea.hidden_states[-1][:,0,:]))
        output = self.pooler_dropout(output)

        arousal_encoder = F.relu(self.a(text[1]))
        valence_encoder = F.relu(self.v(text[2]))

        output = torch.stack([output,arousal_encoder,valence_encoder],dim=0)
        # print(output.size())
        output = self.layernorm(output)
        output = torch.tanh(self.fc1(output))
        output = self.fc1_layernorm(output)
        output = torch.tanh(self.fc2(output))
        logits = self.label(output)

        return logits

class KEA_BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(KEA_BERT, self).__init__()


        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.a = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec
        self.v = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec
        self.d = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec

        self.layernorm = nn.LayerNorm(hidden_size)


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

        input = torch.cat((input,arousal_encoder.unsqueeze(1),valence_encoder.unsqueeze(1)),dim=1)
        input = self.layernorm(input)

        output = self.attention_net(input,cls_input)
        output = F.relu(self.fc1(output))
        logits = self.label(output)

        return logits



class vad_BERT(nn.Module):

    def __init__(self,resume_path,bert_base_model,batch_size,output_size,hidden_size,grad_check,freeze):
        super(vad_BERT, self).__init__()

        if freeze:
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            bert_base_model.load_state_dict(checkpoint['state_dict'])
            for param in bert_base_model.parameters():
                param.require_grad = False

            self.encoder = bert_base_model
        else:
            self.encoder = bert_base_model


        self.pooler = nn.Linear(hidden_size,hidden_size)
        self.pooler_dropout = nn.Dropout(0.1)
        self.a = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec
        self.v = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec
        self.d = nn.Linear(512,hidden_size) #512 is the size of lexicon_vec

        self.layernorm = nn.LayerNorm(4*hidden_size)
        self.fc1 = nn.Linear(4*hidden_size,2*hidden_size)
        self.fc1_layernorm = nn.LayerNorm(2*hidden_size)
        self.fc2 = nn.Linear(2*hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,384)
        self.label = nn.Linear(384,output_size)




    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text[0],attn_mask) # no labels provided, output attention and
        output = torch.tanh(self.pooler(text_fea.hidden_states[-1][:,0,:]))
        output = self.pooler_dropout(output)

        arousal_encoder = F.relu(self.a(text[1]))
        valence_encoder = F.relu(self.v(text[2]))
        dom_encoder = F.relu(self.d(text[3]))

        output = torch.cat((output,arousal_encoder,valence_encoder,dom_encoder),dim=1)
        output = self.layernorm(output)
        output = F.relu(self.fc1(output))
        output = self.fc1_layernorm(output)
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        logits = self.label(output)

        return logits

