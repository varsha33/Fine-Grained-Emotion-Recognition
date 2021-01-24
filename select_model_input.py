import torch

## custom-
from models.model import Electra_BERT,BERT,KEA_BERT,KEA_Electra,KBERT_BiLSTM,KBERT_BiLSTMwSA, Knowledge_baseline
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
        model = Electra_BERT(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "kea_electra":
        model = KEA_Electra(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "kea_bert":
        model = KEA_BERT(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "kbase":
        model = KBERT_BiLSTM(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "kbert_bilstm":
        model = KBERT_BiLSTM(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "kbert_bilstmwsa":
        model = KBERT_BiLSTMwSA(batch_size,output_size,hidden_size,grad_check)
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

