import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification,BertModel,AlbertForSequenceClassification,DistilBertForSequenceClassification
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
np.set_printoptions(threshold=10000)
class sep_BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(sep_BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text,attn_mask,output_hidden_states=True,return_dict=True) # no labels provided, output attention and output hidden states = False

        return text_fea.logits


class _sep_BERT(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(_sep_BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text,attn_mask,output_hidden_states=True,return_dict=True) # no labels provided, output attention and output hidden states = False

        return text_fea

class sep_BERT_LSTM(nn.Module):

    def __init__(self,batch_size,output_size,hidden_size,grad_check):
        super(sep_BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)
        self.lstm = nn.LSTM(hidden_size,384)

    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text,attn_mask,output_hidden_states=True,return_dict=True) # no labels provided, output attention and output hidden states = False


        return text_fea
class asep_BERT(nn.Module):

    def __init__(self,resume_path,bert_base_model,batch_size,output_size,hidden_size,grad_check,freeze):
        super(asep_BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

        self.pooler = nn.Linear(hidden_size,hidden_size)
        self.pooler_dropout = nn.Dropout(0.1)


        self.fc1 = nn.Linear(512,512)
        self.fc2 = nn.Linear(512+768,384)
        self.label = nn.Linear(384,output_size)

    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text[0],attn_mask,output_hidden_states=True,return_dict=True) # no labels provided, output attention and
        # print(text[1].cpu().numpy())
        output =    self.pooler(text_fea.hidden_states[-1][:,0,:])
        output = self.pooler_dropout(output)
        arousal_encoder = self.fc1(text[1])
        output = torch.cat((output,arousal_encoder),dim=1)
        output = torch.tanh(self.fc2(output))
        logits = self.label(output)
        return logits


class asep_appa_BERT(nn.Module):

    def __init__(self,resume_path,bert_base_model,batch_size,output_size,hidden_size,grad_check,freeze):
        super(asep_BERT, self).__init__()

        if freeze:
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            bert_base_model.load_state_dict(checkpoint['state_dict'])
            for param in bert_base_model.parameters():
                param.require_grad = False

            self.encoder = bert_base_model
        else:
            self.encoder = bert_base_model
            # options_name = "bert-base-uncased"
            # self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

        self.pooler = nn.Linear(hidden_size,hidden_size)
        self.pooler_dropout = nn.Dropout(0.1)

        self.a1 = nn.Linear(512,512)
        self.a1_layernorm = nn.LayerNorm(512)
        self.a2 = nn.Linear(512,512)
        self.a2_layernorm = nn.LayerNorm(512)
        self.a3 = nn.Linear(512,1024)
        self.a3_layernorm = nn.LayerNorm(1024)
        self.a4 = nn.Linear(1024,1024)
        self.a4_layernorm = nn.LayerNorm(1024)
        self.a5 = nn.Linear(1024,1024)
        self.a5_layernorm = nn.LayerNorm(1024)
        self.a6 = nn.Linear(1024,2048)
        self.a6_layernorm = nn.LayerNorm(2048)

        self.layernorm = nn.LayerNorm(2048+768)
        self.fc1 = nn.Linear(2048+768,1024)
        self.fc1_layernorm = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024,384)
        self.label = nn.Linear(384,output_size)

    def arousal_encoder(self,arousal_vec):
        output = torch.tanh(self.a1(arousal_vec))
        output = self.a1_layernorm(output)
        output = torch.tanh(self.a2(output))
        output = self.a2_layernorm(output)
        output = torch.tanh(self.a3(output))
        output = self.a3_layernorm(output)
        output = torch.tanh(self.a4(output))
        output = self.a4_layernorm(output)
        output = torch.tanh(self.a5(output))
        output = self.a5_layernorm(output)
        output = torch.tanh(self.a6(output))
        output = self.a6_layernorm(output)
        return output

    def forward(self, text,attn_mask): #here text is utterance based on the input type specified

        text_fea = self.encoder(text[0],attn_mask) # no labels provided, output attention and
        output =    self.pooler(text_fea.hidden_states[-1][:,0,:])
        output = self.pooler_dropout(output)
        arousal_encoder = self.arousal_encoder(text[1])
        output = torch.cat((output,arousal_encoder),dim=1)
        output = self.layernorm(output)
        output = torch.tanh(self.fc1(output))
        output = self.fc1_layernorm(output)
        output = torch.tanh(self.fc2(output))
        logits = self.label(output)
        return logits


class vasep_BERT(nn.Module):

    def __init__(self,resume_path,bert_base_model,batch_size,output_size,hidden_size,grad_check,freeze):
        super(vasep_BERT, self).__init__()

        if freeze:
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            bert_base_model.load_state_dict(checkpoint['state_dict'])
            for param in bert_base_model.parameters():
                param.require_grad = False

            self.encoder = bert_base_model
        else:
            self.encoder = bert_base_model
            # options_name = "bert-base-uncased"
            # self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=output_size,gradient_checkpointing=grad_check)

        self.pooler = nn.Linear(hidden_size,hidden_size)
        self.pooler_dropout = nn.Dropout(0.1)
        self.a1 = nn.Linear(512,512)
        self.a1_layernorm = nn.LayerNorm(512)
        self.a2 = nn.Linear(512,512)
        self.a2_layernorm = nn.LayerNorm(512)
        self.a3 = nn.Linear(512,1024)
        self.a3_layernorm = nn.LayerNorm(1024)
        self.a4 = nn.Linear(1024,1024)
        self.a4_layernorm = nn.LayerNorm(1024)
        self.a5 = nn.Linear(1024,1024)
        self.a5_layernorm = nn.LayerNorm(1024)
        self.a6 = nn.Linear(1024,2048)
        self.a6_layernorm = nn.LayerNorm(2048)

        self.layernorm = nn.LayerNorm(512+512+768)
        self.fc1 = nn.Linear(512+512+768,768)
        self.fc1_layernorm = nn.LayerNorm(768)
        self.fc2 = nn.Linear(768,384)
        self.label = nn.Linear(384,output_size)

    def lexicon_encoder(self,lexicon_vec):
        output = torch.tanh(self.a1(lexicon_vec))
        output = self.a1_layernorm(output)
        # output = torch.tanh(self.a2(output))
        # output = self.a2_layernorm(output)
        # output = torch.tanh(self.a3(output))
        # output = self.a3_layernorm(output)
        # output = torch.tanh(self.a4(output))
        # output = self.a4_layernorm(output)
        # output = torch.tanh(self.a5(output))
        # output = self.a5_layernorm(output)
        # output = torch.tanh(self.a6(output))
        # output = self.a6_layernorm(output)
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
