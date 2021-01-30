import torch

## custom-
from models.model import ELECTRA,KEA_ELECTRA, KEA_Electra_Word_level,KEA_Bert_Word_level
from transformers.configuration_bert import BertConfig as config

def select_model(config,arch_name,vocab_size=None,word_embeddings=None,grad_check=True):

    batch_size = config.batch_size
    hidden_size = config.hidden_size
    embedding_length = config.embedding_length
    # arch_name = config.arch_name
    output_size = config.output_size
    bert_resume_path = ""
    # freeze = config.freeze

    if arch_name == "bert":
        model = BERT(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "electra":
        model = ELECTRA(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "kea_electra":
        model = KEA_ELECTRA(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "kea_bert":
        model = KEA_BERT(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "kea_electra_word":
        model = KEA_Electra_Word_level(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "kea_bert_word":
        model = KEA_Bert_Word_level(batch_size,output_size,hidden_size,grad_check)
    return model



def select_input(batch,config,arch_name):
    # arch_name = config.arch_name
    input_type = config.input_type
    embedding_type = config.embedding_type
    dataset = config.dataset

    if dataset == "ed":

        if  arch_name == "electra" or arch_name == "bert":
            text = batch["utterance_data"]
            attn = batch["utterance_data_attn_mask"]

        else:
            text = [batch["utterance_data"],batch["arousal_data"],batch["valence_data"],batch["dom_data"]]
            attn = batch["utterance_data_attn_mask"]

        target = batch["emotion"]
        target = torch.Tensor(target)

    elif dataset == "goemotions" or dataset == "semeval":


        if  arch_name == "electra" or arch_name == "bert":
            text = batch["utterance_data"]
            attn = batch["utterance_data_attn_mask"]

        else:
            text = [batch["utterance_data"],batch["arousal_data"],batch["valence_data"],batch["dom_data"]]
            attn = batch["utterance_data_attn_mask"]

        target = batch["emotion"]
        target = torch.stack(target)

    return text,attn,target

