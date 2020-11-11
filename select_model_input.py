import torch

## custom-
from models.model import Electra_BERT,BERT,KEA_BERT,KEA_Electra


def select_model(config,arch_name,vocab_size=None,word_embeddings=None,grad_check=True):

    batch_size = config.batch_size
    hidden_size = config.hidden_size
    embedding_length = config.embedding_length
    # arch_name = config.arch_name
    output_size = config.output_size
    bert_resume_path = config.resume_path
    # freeze = config.freeze

    if arch_name == "bert":
        model = BERT(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "electra":
        model = Electra_BERT(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "kea_electra":
        model = KEA_Electra(batch_size,output_size,hidden_size,grad_check)
    if arch_name == "kea_bert":
        model = KEA_BERT(batch_size,output_size,hidden_size,grad_check)
    return model



def select_input(batch,config,arch_name):
    # arch_name = config.arch_name
    input_type = config.input_type
    embedding_type = config.embedding_type


    if  arch_name == "electra" or arch_name == "bert":

        if input_type == "speaker+listener":
            text = batch["utterance_data"]
            attn = batch["utterance_data_attn_mask"]
        elif input_type == "speaker":
            text = batch["speaker_data"]
            attn = batch["speaker_data_attn_mask"]
        elif input_type == "listener":
            text = batch["listener_data"]
            attn = batch["listener_data_attn_mask"]
    if arch_name == "kea_bert" or arch_name == "kea_electra":
        text = [batch["utterance_data"],batch["arousal_data"],batch["valence_data"],batch["dom_data"]]
        attn = batch["utterance_data_attn_mask"]

    target = batch["emotion"]
    target = torch.Tensor(target)

    return text,attn,target

