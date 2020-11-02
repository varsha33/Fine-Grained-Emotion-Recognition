import torch

## custom

from models.LSTM_Attn import AttentionModel
from models.selfAttention import SelfAttention
from models.transformer_encoder import TransformerModel
from models.RCNN import RCNN
from models.BERT import _BERT, BERT,a_BERT,va_BERT,vad_BERT,KEA_BERT


def select_model(config,arch_name,vocab_size=None,word_embeddings=None,grad_check=True):

    batch_size = config.batch_size
    hidden_size = config.hidden_size
    embedding_length = config.embedding_length
    # arch_name = config.arch_name
    output_size = config.output_size
    bert_resume_path = config.resume_path
    freeze = config.freeze

    if arch_name == "lstm":
        model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "selfattn":
        model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "rcnn":
        model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "lstm+attn":
        model = AttentionModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "transformer":
        model = TransformerModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "bert":
        model = BERT(batch_size,output_size,hidden_size,grad_check)
    elif arch_name == "a_bert":
        bert_model = _BERT(batch_size,output_size,hidden_size,grad_check)
        model = a_BERT(bert_resume_path,bert_model,batch_size,output_size,hidden_size,grad_check,freeze)
    elif arch_name == "va_bert":
        bert_model = _BERT(batch_size,output_size,hidden_size,grad_check)
        model = va_BERT(bert_resume_path,bert_model,batch_size,output_size,hidden_size,grad_check,freeze)
    elif arch_name == "kea_bert":
        model = KEA_BERT(batch_size,output_size,hidden_size,grad_check)
    elif arch_name == "vad_bert":
        bert_model = _BERT(batch_size,output_size,hidden_size,grad_check)
        model = vad_BERT(bert_resume_path,bert_model,batch_size,output_size,hidden_size,grad_check,freeze)

    return model



def select_input(batch,config,arch_name):
    # arch_name = config.arch_name
    input_type = config.input_type
    embedding_type = config.embedding_type

    if embedding_type == "bert" or embedding_type == "glove+bert":

        if arch_name == "bert":

            if input_type == "speaker+listener":
                text = batch["utterance_data"]
                attn = batch["utterance_data_attn_mask"]
            elif input_type == "speaker":
                text = batch["speaker_data"]
                attn = batch["speaker_data_attn_mask"]
            elif input_type == "listener":
                text = batch["listener_data"]
                attn = batch["listener_data_attn_mask"]


        if arch_name == "a_bert":
            text = [batch["utterance_data"],batch["arousal_data"]]
            attn = batch["utterance_data_attn_mask"]

        if arch_name == "va_bert":
            text = [batch["utterance_data"],batch["arousal_data"],batch["valence_data"]]
            attn = batch["utterance_data_attn_mask"]

        if arch_name == "vad_bert" or arch_name == "kea_bert":
            text = [batch["utterance_data"],batch["arousal_data"],batch["valence_data"],batch["dom_data"]]
            attn = batch["utterance_data_attn_mask"]

        target = batch["emotion"]
        target = torch.Tensor(target)


    elif embedding_type == "glove":

        text = batch["glove_data"]
        attn = batch["utterance_data_attn_mask"] ##dummy here

        target = batch["emotion"]
        target = torch.Tensor(target)


    return text,attn,target

