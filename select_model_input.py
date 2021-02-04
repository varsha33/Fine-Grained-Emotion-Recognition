import torch

## custom-
from models.model import KEA_ELECTRA, KEA_Electra_Word_level,KEA_Bert_Word_level,KEA_BERT


## config here is log_dict.param
def select_model(config):

    batch_size = config.batch_size
    hidden_size = config.hidden_size
    output_size = config.output_size
    arch_name = config.arch_name

    if arch_name == "kea_electra":
        model = KEA_ELECTRA(batch_size,output_size,hidden_size)
    if arch_name == "kea_bert":
        model = KEA_BERT(batch_size,output_size,hidden_size)
    if arch_name == "kea_electra_word":
        model = KEA_Electra_Word_level(batch_size,output_size,hidden_size)
    if arch_name == "kea_bert_word":
        model = KEA_Bert_Word_level(batch_size,output_size,hidden_size)
    return model



def select_input(batch,config):

    dataset = config.dataset
    arch_name = config.arch_name

    text = [batch["utterance_data"],batch["arousal_data"],batch["valence_data"],batch["dom_data"]]
    attn = batch["utterance_data_attn_mask"]

    if dataset == "ed": ##single-label, the output label is numerical

        target = batch["emotion"]
        target = torch.Tensor(target)

    elif dataset == "goemotions" or dataset == "semeval": ##multi-label, the output label is one-hot encoded

        target = batch["emotion"]
        target = torch.stack(target)

    return text,attn,target

