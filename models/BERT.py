import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification,BertModel,DistilBertForSequenceClassification
from torch.autograd import Variable
from torch.nn import functional as F

class BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

    def forward(self, text): #here text is utterance based on the input type specified

        text_fea = self.encoder(text,output_hidden_states=True,return_dict=True) # no labels provided, output attention and output hidden states = False

        return text_fea

class simple_BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(simple_BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

    def forward(self, text): #here text is utterance based on the input type specified

        text_fea = self.encoder(text,return_dict=True) # no labels provided, output attention and output hidden states = False

        return text_fea.logits

class Arousal_BERT(nn.Module):

    def __init__(self,resume_path,bert_base_model,batch_size,output_size,hidden_size,grad_check,freeze=True,lstm_hidden_size=256):
        super(Arousal_BERT, self).__init__()

        if freeze:
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            bert_base_model.load_state_dict(checkpoint['state_dict'])
            for param in bert_base_model.parameters():
                param.require_grad = False

            self.encoder = bert_base_model
        else:
            ## if not using frozen-bert
            self.encoder = bert_base_model

        self.batch_size = batch_size
        self.arousal_encoder = nn.Linear(512,256)
        self.fc1 = nn.Linear(768,256)
        self.fc2 = nn.Linear(256+256,128)
        self.label = nn.Linear(128,output_size)
    def forward(self, text):

        text_fea = self.encoder(text[0])
        input1 = torch.tanh(self.fc1(text_fea.hidden_states[-1][:,0,:]))
        input2 = torch.tanh(self.arousal_encoder(text[1]))
        input = torch.cat((input1,input2),1)
        input = F.relu(self.fc2(input))
        logits = self.label(input)
        return logits

class BERT_RCNN(nn.Module):

    def __init__(self,resume_path,bert_base_model,batch_size,output_size,hidden_size,grad_check,freeze=False,lstm_hidden_size=128):
        super(BERT_RCNN, self).__init__()


        if freeze:
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            bert_base_model.load_state_dict(checkpoint['state_dict'])
            for param in bert_base_model.parameters():
                param.require_grad = False

            self.bert_base_model = bert_base_model
        else:
            options_name = "bert-base-uncased"
            self.bert_base_model = BertModel.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = lstm_hidden_size

        self.dropout = 0.8
        self.lstm = nn.LSTM(hidden_size,lstm_hidden_size,num_layers=1, dropout=self.dropout, bidirectional=True)
        self.W2 = nn.Linear(2*lstm_hidden_size+hidden_size, lstm_hidden_size)
        self.label = nn.Linear(lstm_hidden_size, output_size)


    def forward(self,text):

        input = self.bert_base_model(text).hidden_states[-1]
        ## Uncommnet the four lines to take maxpool of the last 4 states and change [-1] to [-4:] in the above line
        # input = torch.stack(input) ## input.size() = (4,batch_size,num_sequences,hidden_size) because last 4 hidden layers
        # input = input.permute(1, 2, 3,0)
        # input = self.maxpool(input)  ## input.size() = (batch_size,num_sequences,hidden_size,1)
        # input = input.squeeze() ## input.size() = (batch_size,num_sequences,hidden_size)

        input = input.permute(1,0,2)

        h_0 = Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))

        final_encoding = torch.cat((output, input), 2).permute(1, 0, 2)

        y = self.W2(final_encoding) # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1) # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2]) # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)
        logits = self.label(y)

        return logits

class Speaker_Listener_BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(Speaker_Listener_BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.sencoder = BertModel.from_pretrained(options_name,gradient_checkpointing=grad_check) ## Speaker BERT
        self.lencoder = BertModel.from_pretrained(options_name,gradient_checkpointing=grad_check) ## Listener BERT
        self.label = nn.Linear(2*hidden_size,output_size)


    def forward(self,text): #here text is a dict of speaker and listener utterance

        speaker_text_fea = self.sencoder(text[0],return_dict=True) # no labels provided, output attention and output hidden states = False
        listener_text_fea = self.lencoder(text[1],return_dict=True) # no labels provided, output attention and output hidden states = False

        stacked_hidden = torch.cat((speaker_text_fea.pooler_output, listener_text_fea.pooler_output), dim=1)
        stacked_hidden = self.label(stacked_hidden)

        return stacked_hidden

class Hierarchial_BERT_Deep(nn.Module):

    def __init__(self,resume_path,bert_base_model,batch_size,output_size,hidden_size,grad_check,freeze=False,lstm_hidden_size=256):
        super(Hierarchial_BERT, self).__init__()

        if freeze:
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            bert_base_model.load_state_dict(checkpoint['state_dict'])
            for param in bert_base_model.parameters():
                param.require_grad = False
            self.encoder = bert_base_model
        else:
            self.encoder = bert_base_model


        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = lstm_hidden_size
        self.hidden_size = hidden_size


        self.dropout = 0.8
        self.lstm = nn.LSTM(hidden_size,lstm_hidden_size,num_layers=1, bidirectional=True)
        self.fc1 = nn.Linear(2*lstm_hidden_size,128) ## 256 heuristically chosen
        # self.fc2 = nn.Linear(256,128) ## 128 heuristically chosen
        self.label = nn.Linear(128, output_size)


    def forward(self,text): #here text is utterance_data_list

        sentence_encoder_list = []
        for i in text:
            sentence_encoder_list.append(self.encoder(i).hidden_states[-1])

        sentence_encoded = torch.stack(sentence_encoder_list)


        # input = sentence_encoded.permute(1,0,2)

        # h_0 = Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_size).cuda())
        # c_0 = Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_size).cuda())

        # output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))

        # output = torch.cat((final_hidden_state[0,:,:],final_hidden_state[1,:,:]),1)

        # output = F.relu(self.fc1(output))
        # # output = F.relu(self.fc2(output))
        # logits = self.label(output)
        return logits


class Hierarchial_BERT(nn.Module):

    def __init__(self,resume_path,bert_base_model,batch_size,output_size,hidden_size,grad_check,freeze=False,lstm_hidden_size=256):
        super(Hierarchial_BERT, self).__init__()

        if freeze:
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            bert_base_model.load_state_dict(checkpoint['state_dict'])
            for param in bert_base_model.parameters():
                param.require_grad = False
            self.encoder = bert_base_model
        else:
            self.encoder = bert_base_model


        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = lstm_hidden_size
        self.hidden_size = hidden_size


        self.dropout = 0.8
        self.lstm = nn.LSTM(hidden_size,lstm_hidden_size,num_layers=1, bidirectional=True)
        self.fc1 = nn.Linear(2*lstm_hidden_size,128) ## 256 heuristically chosen
        # self.fc2 = nn.Linear(256,128) ## 128 heuristically chosen
        self.label = nn.Linear(128, output_size)


    def forward(self,text): #here text is utterance_data_list

        sentence_encoder_list = []
        for i in text:
            sentence_encoder_list.append(self.encoder(i).hidden_states[-1][:,0,:])

        sentence_encoded = torch.stack(sentence_encoder_list)


        input = sentence_encoded.permute(1,0,2)

        h_0 = Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))

        output = torch.cat((final_hidden_state[0,:,:],final_hidden_state[1,:,:]),1)

        output = F.relu(self.fc1(output))
        # output = F.relu(self.fc2(output))
        logits = self.label(output)
        return logits


class Hierarchial_BERT_SL(nn.Module):
    def __init__(self,resume_path,bert_base_model,batch_size,output_size,hidden_size,grad_check,freeze=False):

        super(Hierarchial_BERT_SL,self).__init__()
        if freeze:
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            bert_base_model.load_state_dict(checkpoint['state_dict'])
            for param in bert_base_model.parameters():
                param.require_grad = False
            self.utterance_encoder = bert_base_model
        else:
            options_name = "bert-base-uncased"
            self.utterance_encoder = BertModel.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

        self.fc1 = nn.Linear(hidden_size,128) ## 128 heuristically chosen
        self.label = nn.Linear(2*128,output_size)

    def forward(self,text):
        speaker_encoded,listener_encoded  = [],[]
        for i in text:
            output = torch.mean(self.utterance_encoder(i,return_dict=True).last_hidden_state[:,-4:,:],1)
            speaker_encoded.append(output[0::2,:])
            listener_encoded.append(output[1::2,:])

        listener_encoded = torch.mean(torch.stack(listener_encoded),1)
        speaker_encoded =  torch.mean(torch.stack(speaker_encoded),1)
        output = torch.cat((F.relu(self.fc1(listener_encoded)),F.relu(self.fc1(speaker_encoded))),1)
        logits = self.label(output)
        return logits

class BERT_MTL(nn.Module):

    def __init__(self,batch_size, output_size, hidden_size,grad_check):
        super(BERT_MTL, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,hidden_size=hidden_size,num_labels=output_size,gradient_checkpointing=grad_check)
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
