import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        
        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels=32,gradient_checkpointing=True)
        # print(superelf.encoder.config)
    def forward(self, text):
       	text_fea = self.encoder(text)
       	
        return text_fea[0]