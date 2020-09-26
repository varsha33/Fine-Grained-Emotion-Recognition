import numpy as np

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification,BertModel,DistilBertForSequenceClassification
from torch.autograd import Variable
from torch.nn import functional as F



class BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size):
        super(BERT, self).__init__()
 
        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=True)

    def forward(self, text): #here text is utterance based on the input type specified

        text_fea = self.encoder(text,return_dict=True) # no labels provided, output attention and output hidden states = False
        
        return text_fea.logits



class Speaker_Listener_BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size):
        super(Speaker_Listener_BERT, self).__init__()
        
        options_name = "bert-base-uncased"
        self.sencoder = BertModel.from_pretrained(options_name,gradient_checkpointing=True)
        self.lencoder = BertModel.from_pretrained(options_name,gradient_checkpointing=True)
        self.label = nn.Linear(2*hidden_size,output_size)


    def forward(self,text): #here text is a dict of speaker and listener utterance

        speaker_text_fea = self.sencoder(text[0],return_dict=True) # no labels provided, output attention and output hidden states = False
        listener_text_fea = self.lencoder(text[1],return_dict=True) # no labels provided, output attention and output hidden states = False
        
        stacked_hidden = torch.cat((speaker_text_fea.pooler_output, listener_text_fea.pooler_output), dim=1)
        stacked_hidden = self.label(stacked_hidden)

        return stacked_hidden



class Hierarchial_BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size):
        super(Hierarchial_BERT, self).__init__()
            
        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,gradient_checkpointing=True)
        self.attention = nn.Linear(hidden_size,8)
        self.label = nn.Linear(6144,output_size)

        # print(self.speaker_encoder.config)

    def attention_net(self,turn_input):

        attn_weight_matrix = self.attention(turn_input)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self,text): #here text is utterance_data_list

        sentence_encoder_list = []
        for i in text:
            sentence_encoder_list.append(self.encoder(i,return_dict=True,output_hidden_states=True).hidden_states[-1][:,0,:])
        
        sentence_encoded = torch.stack(sentence_encoder_list)
        turn_encoded = torch.flatten(turn_encoded, start_dim=1)
        logits = self.label(turn_encoded)

        return logits

class Hierarchial_BERT_wLSTM(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,lstm_hidden_size=256):
        super(Hierarchial_BERT_wLSTM, self).__init__()
            
        options_name = "bert-base-uncased"
        self.batch_size = batch_size
        self.lstm_hidden_size = lstm_hidden_size
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,gradient_checkpointing=True)
        self.attention = nn.Linear(hidden_size,8)
        self.lstm =  nn.LSTM(hidden_size,lstm_hidden_size)
        self.label = nn.Linear(lstm_hidden_size,output_size)

    def attention_net(self, lstm_output, final_state):
        
        hidden = final_state.squeeze(0)

        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        
        soft_attn_weights = F.softmax(attn_weights, 1)
        
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        
        return new_hidden_state

    def forward(self,text): #here text is utterance_data_list

        sentence_encoder_list = []
        for i in text:
            sentence_encoder_list.append(self.encoder(i,return_dict=True,output_hidden_states=True).hidden_states[-1][:,0,:])
        
        sentence_encoded = torch.stack(sentence_encoder_list) # (batch_size,utterance_length (8),hidden_size(768))
        sentence_encoded = sentence_encoded.permute(1, 0, 2) # input.size() = (utterance_length (8),batch_size,hidden_size(768))
        
        h_0 = Variable(torch.zeros(1,self.batch_size,self.lstm_hidden_size).cuda())
        c_0 = Variable(torch.zeros(1,self.batch_size,self.lstm_hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(sentence_encoded, (h_0, c_0))
        output = output.permute(1, 0, 2) # output.size() = (batch_size,utterance_length (8),lstm_hidden_size(256))

        attn_output = self.attention_net(output, final_hidden_state) # attn_output.size() = (batch_size,lstm_hidden_size(256))
        logits = self.label(attn_output)

        return logits


class BERT_MTL(nn.Module):

    def __init__(self,batch_size, output_size, hidden_size):
        super(BERT_MTL, self).__init__()
        
        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,hidden_size=hidden_size,num_labels=output_size,gradient_checkpointing=True)
        # print(superelf.encoder.config)
        self.fc1 = nn.Linear(hidden_size,128) # 128 heuristically chosen for the valence head
        self.dropout = nn.Dropout(0.4) 
        self.label = nn.Linear(128,2) # for valence head


    def forward(self, text):
        text_fea = self.encoder(text,output_attentions=True,output_hidden_states=True,return_dict=True) # no labels provided, output attention and output hidden states = False
        last_hidden_state =  text_fea.hidden_states[-1]
        last_hidden_state_first_token = last_hidden_state[:,0,:]
    
        valence_logits = torch.tanh(self.fc1(last_hidden_state_first_token))
        valence_logits = self.dropout(valence_logits)
        valence_logits = self.label(valence_logits)
    
        return text_fea.logits,valence_logits
