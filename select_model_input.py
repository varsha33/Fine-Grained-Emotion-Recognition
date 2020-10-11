import torch

## custom
from models.BERT import BERT,simple_BERT,BERT_RCNN,Speaker_Listener_BERT,Hierarchial_BERT_SL,Arousal_BERT,Hierarchial_BERT,Hierarchial_Deep_BERT
from models.LSTM import LSTMClassifier
from models.LSTM_Attn import AttentionModel
from models.selfAttention import SelfAttention
from models.transformer_encoder import TransformerModel
from models.RCNN import RCNN
from models.CNN import CNN
from models.RNN import RNN
from models.RCNN_attn import RCNN_attn
from models.sep_BERT import sep_BERT,asep_BERT,vasep_BERT,_sep_BERT


def select_model(config,vocab_size=None,word_embeddings=None,grad_check=True):

    batch_size = config.batch_size
    hidden_size = config.hidden_size
    embedding_length = config.embedding_length
    arch_name = config.arch_name
    output_size = config.output_size
    bert_resume_path = config.resume_path
    freeze = config.freeze

    if arch_name == "lstm":
        model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "selfattn":
        model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "rnn":
        model = RNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "rcnn":
        model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "lstm+attn":
        model = AttentionModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "cnn":
        model = CNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "transformer":
        model = TransformerModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "rcnn_attn":
        model = RCNN_attn(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "bert":
        model = simple_BERT(batch_size,output_size,hidden_size,grad_check)
    elif arch_name == "sl_bert":
        model = Speaker_Listener_BERT(batch_size,output_size,hidden_size,grad_check)
    elif arch_name == "h_bert":
        bert_model = BERT(batch_size,output_size,hidden_size,grad_check)
        model = Hierarchial_BERT(bert_resume_path,bert_model,batch_size,output_size,hidden_size,grad_check)
    elif arch_name == "hd_bert":
        bert_model = BERT(batch_size,output_size,hidden_size,grad_check)
        model = Hierarchial_Deep_BERT(bert_resume_path,bert_model,batch_size,output_size,hidden_size,grad_check)
    elif arch_name == "bert_rcnn":
        bert_model = BERT(batch_size,output_size,hidden_size,grad_check)
        model = BERT_RCNN(bert_resume_path,bert_model, batch_size,output_size,hidden_size,grad_check)
    elif arch_name == "h_bert_sl":
        bert_model = BERT(batch_size,output_size,hidden_size,grad_check)
        model = Hierarchial_BERT_SL(bert_resume_path,bert_model,batch_size,output_size,hidden_size,grad_check)
    elif arch_name == "a_bert":
        bert_model = BERT(batch_size,output_size,hidden_size,grad_check)
        model = Arousal_BERT(bert_resume_path,bert_model, batch_size,output_size,hidden_size,grad_check)
    elif arch_name == "bertplus_rcnn":
        bert_model = BERT(batch_size,output_size,hidden_size,grad_check)
        model = BERTplus_RCNN(bert_resume_path,bert_model, batch_size,output_size,hidden_size,grad_check)
    elif arch_name == "sep_bert":
        model = sep_BERT(batch_size,output_size,hidden_size,grad_check)
    elif arch_name == "asep_bert":
        bert_model = _sep_BERT(batch_size,output_size,hidden_size,grad_check)
        model = asep_BERT(bert_resume_path,bert_model,batch_size,output_size,hidden_size,grad_check,freeze)
    elif arch_name == "vasep_bert":
        bert_model = _sep_BERT(batch_size,output_size,hidden_size,grad_check)
        model = vasep_BERT(bert_resume_path,bert_model,batch_size,output_size,hidden_size,grad_check,freeze)

    return model



def select_input(batch,config):
    arch_name = config.arch_name
    input_type = config.input_type
    embedding_type = config.embedding_type

    if embedding_type == "glove": #for models until BERT


        if input_type == "speaker+listener":
            text = batch.utterance_data
        elif input_type == "speaker":
            text = batch.speaker_utterance
        elif input_type == "prompt":
            text = batch.prompt

        target = batch.emotion

    elif embedding_type == "bert":

        if arch_name == "bert":
            if input_type == "speaker+listener":
                text = batch["utterance_data"]
                attn = batch["utterance_data_attn"]
            elif input_type == "speaker":
                text = batch["speaker_data"]
                attn = batch["speaker_data_attn"]
            elif input_type == "listener":
                text = batch["listener_data"]
                attn = batch["listener_data_attn"]
            elif input_type == "prompt":
                text = batch["prompt"]

        if arch_name == "sl_bert":
            text = [batch["speaker_idata"],batch["listener_idata"]]

        if arch_name == "h_bert" or arch_name == "h_bert_sl":
            text = batch["utterance_data_list"]
            attn = batch["utterance_data_list_attn"]

        if arch_name == "bert_rcnn" or arch_name == "bertplus_rcnn":
            text = batch["utterance_data"]
            attn = batch["utterance_data_attn"]

        if arch_name == "a_bert":
            text = [batch["utterance_data"],batch["arousal_utterance"]]

        if arch_name == "hd_bert":
            text = batch["turnwise_data"]
            attn = batch["turnwise_data_attn"]

        if arch_name == "sep_bert":
            text = batch["sep_data"]
            attn = batch["sep_data_attn"]

        if arch_name == "asep_bert":
            text = [batch["sep_data"],batch["arousal_sep"]]
            attn = batch["sep_data_attn"]

        if arch_name == "vasep_bert":
            text = [batch["sep_data"],batch["arousal_sep"],batch["valence_sep"]]
            attn = batch["sep_data_attn"]

        target = batch["emotion"]
        target = torch.Tensor(target)


    return text,attn,target

