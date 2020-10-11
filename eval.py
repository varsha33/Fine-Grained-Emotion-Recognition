import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models.LSTM import LSTMClassifier
from models.LSTM_Attn import AttentionModel
from models.selfAttention import SelfAttention
from models.transformer_encoder import TransformerModel
from models.RCNN import RCNN
from models.CNN import CNN
from models.RNN import RNN
from models.RCNN_attn import RCNN_attn
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import shutil
import time
import json
import config
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()


def eval_model(model, val_iter,confusion=False,per_class=False):
    total_epoch_loss = 0
    total_epoch_acc = 0
    y_true = []
    y_pred = []
    if confusion:
        conf_matrix = torch.zeros(output_size, output_size)
    if per_class:
           class_correct = list(0. for i in range(output_size))
           class_total = list(0. for i in range(output_size))

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.utterance[0]

            if (text.size()[0] is not config.batch_size):
                continue
            target = batch.emotion
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            if confusion:
                for t, p in zip(target.data, torch.max(prediction, 1)[1].view(target.size()).data):
                        conf_matrix[t.long(), p.long()] += 1

            correct = np.squeeze(torch.max(prediction, 1)[1].eq(target.view_as(torch.max(prediction, 1)[1])))
            if per_class:
                for i in range(batch_size):
                    label = target[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            pred_ind = torch.max(prediction, 1)[1].view(target.size()).data
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

            y_true.extend(target.data.cpu().tolist())
            y_pred.extend(pred_ind.cpu().tolist())

        if confusion:
            import seaborn as sns
            sns.heatmap(conf_matrix, annot=True,xticklabels=list(config.emo_label_map.keys()),yticklabels=list(config.emo_label_map.keys()))
            plt.show()
        if per_class:
            for i in range(output_size):
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                config.label_emo_map[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))

        f1_score_e = f1_score(y_true, y_pred, average='macro')
    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter),f1_score_e

def load_model(resume,model,optimizer):

    # print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()
    optimizer.load_state_dict(checkpoint['optimizer'])
    test_loss, test_acc = eval_model(model, test_iter,confusion=config.confusion,per_class=config.per_class)



if __name__ == '__main__':

    log_path = config.resume.replace("model_best.pth","log.json")

    with open(log_path,'r') as f:
        log = json.load(f)
    f.close()

    learning_rate = log["param"]["learning_rate"]
    batch_size = log["param"]["batch_size"]
    hidden_size = log["param"]["hidden_size"]
    embedding_length = log["param"]["embedding_length"]
    arch_name = log["param"]["arch_name"]
    output_size = 32


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate)
    load_model(config.resume,model,optimizer)
    print(f'Train Acc: {log["train_acc"]:.3f}%, Valid Acc: {log["valid_acc"]:.3f}%, Test Acc: {log["test_acc"]:.3f}%')

    if arch_name == "lstm":
        print("Loading Model : LSTM")
        model = LSTMClassifier(param["batch_size"], output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "selfattn":
        print("Loading Model : SelfAttention")
        model = SelfAttention(param["batch_size"], output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "rnn":
        print("Loading Model : RNN")
        model = RNN(param["batch_size"], output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "rcnn":
        print("Loading Model : RCNN")
        model = RCNN(*args,**kwargs)
    elif arch_name == "lstm+attn":
        print("Loading Model : LSTM + Attention")
        model = AttentionModel(param["batch_size"], output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "cnn":
        print("Loading Model : CNN")
        model = CNN(param["batch_size"], output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "transformer":
        print("Loading Model : Transformer Encoder")
        model = TransformerModel(param["batch_size"], output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif arch_name == "rcnn_attn":
        print("Loading Model : RCNN + Attention")
        model = RCNN_attn(param["batch_size"], output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
