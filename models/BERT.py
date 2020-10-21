import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification,BertModel,AlbertForSequenceClassification,DistilBertForSequenceClassification
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text,attn_mask,output_hidden_states=True,return_dict=True) # no labels provided, output attention and output hidden states = False

        return text_fea.logits

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
        output = self.pooler(text_fea.hidden_states[-1][:,0,:])
        output = self.pooler_dropout(output)
        arousal_encoder = self.fc1(text[1])
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
        self.a1 = nn.Linear(512,512) #512 is the size of lexicon_vec
        self.a1_layernorm = nn.LayerNorm(512)

        self.layernorm = nn.LayerNorm(512+512+hidden_size)
        self.fc1 = nn.Linear(512+512+hidden_size,hidden_size)
        self.fc1_layernorm = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size,384)
        self.label = nn.Linear(384,output_size)

    def lexicon_encoder(self,lexicon_vec):
        output = torch.tanh(self.a1(lexicon_vec))
        output = self.a1_layernorm(output)
        return output

    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text[0],attn_mask) # no labels provided, output attention and
        output =    self.pooler(text_fea.hidden_states[-1][:,0,:])
        output = self.pooler_dropout(output)

        arousal_encoder = self.lexicon_encoder(text[1])
        valence_encoder = self.lexicon_encoder(text[2])

        output = torch.cat((output,arousal_encoder,valence_encoder),dim=1)
        output = self.layernorm(output)
        output = torch.tanh(self.fc1(output))
        output = self.fc1_layernorm(output)
        output = torch.tanh(self.fc2(output))
        logits = self.label(output)
        return logits

class BERT_RCNN(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check,lstm_hidden_size=128):
        super(BERT_RCNN, self).__init__()


        options_name = "bert-base-uncased"
        self.encoder = BertModel.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = lstm_hidden_size

        self.dropout = nn.Dropout(0.1)
        self.pooler = nn.Linear(hidden_size,hidden_size)
        self.lstm = nn.LSTM(hidden_size,lstm_hidden_size,num_layers=1, bidirectional=True)
        self.W2 = nn.Linear(2*lstm_hidden_size+hidden_size, lstm_hidden_size)
        self.label = nn.Linear(lstm_hidden_size, output_size)


    def forward(self,text,attn_mask):

        input = self.encoder(text,attn_mask,output_hidden_states=True,return_dict=True).hidden_states[-1]
        # print(input.size())
        input = self.pooler(input)
        input = self.dropout(input)

        input = input.permute(1,0,2)

        h_0 = Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, self.batch_size, self.lstm_hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))

        final_encoding = torch.cat((input,output), 2).permute(1, 0, 2)

        y = self.W2(final_encoding) # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1).contiguous() # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2]) # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)

        logits = self.label(y)


        return logits

class SL_BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(SL_BERT, self).__init__()


        options_name = "bert-base-uncased"
        self.encoder = BertModel.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

        self.batch_size = batch_size
        self.output_size = output_size


        self.dropout = nn.Dropout(0.1)
        self.spooler = nn.Linear(hidden_size,hidden_size)
        self.lpooler = nn.Linear(hidden_size,hidden_size)

        self.sfc1 = nn.Linear(hidden_size,256)
        self.lfc1 = nn.Linear(hidden_size,256)


        self.fc2 = nn.Linear(2*256,256)
        self.label = nn.Linear(256,output_size)
        self.attn = nn.MultiheadAttention(2*hidden_size,4)

    def forward(self,text,attn_mask):

        sinput = self.encoder(text[0],attn_mask[0],output_hidden_states=True,return_dict=True)

        linput = self.encoder(text[0],attn_mask[0],output_hidden_states=True,return_dict=True)

        sinput = self.spooler(sinput.hidden_states[-1][:,0,:])
        sinput = self.dropout(sinput)
        sinput = torch.tanh(self.sfc1(sinput))


        linput = self.lpooler(linput.hidden_states[-1][:,0,:])
        linput = self.dropout(linput)
        linput = torch.tanh(self.lfc1(linput))

        sl_input = torch.cat((sinput,linput),dim=1)
        sl_input = F.relu(self.fc2(sl_input))
        logits = self.label(sl_input)

        return logits

