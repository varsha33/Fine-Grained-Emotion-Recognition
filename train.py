import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models.BERT import BERT
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
import random 

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

if config.embedding_type != "bert":
    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset_glove()
else:
    train_iter, valid_iter, test_iter = load_data.load_dataset_bert()


def save_checkpoint(state, is_best,filename='checkpoint.pth.tar'):
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,filename.replace('checkpoint.pth.tar','model_best.pth.tar'))

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        if input_type == "speaker+listener":
            text = batch.utterance_data
        elif input_type == "speaker":
            text = batch.speaker_data
        elif input_type == "prompt":
            text = batch.prompt

        target = batch.emotion
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not config.batch_size):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optimizer.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optimizer.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter,confusion=False,per_class=False):
    total_epoch_loss = 0
    total_epoch_acc = 0
    if confusion:
        conf_matrix = torch.zeros(output_size, output_size)
    if per_class:
           class_correct = list(0. for i in range(output_size))
           class_total = list(0. for i in range(output_size))

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            if input_type == "speaker+listener":
                text = batch.utterance_data
            elif input_type == "speaker":
                text = batch.speaker_data
            elif input_type == "prompt":
                text = batch.prompt

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

            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
        if confusion:
            import seaborn as sns
            sns.heatmap(conf_matrix, annot=True,xticklabels=list(config.emo_label_map.keys()),yticklabels=list(config.emo_label_map.keys()))
            plt.show()
        if per_class:
            for i in range(output_size):
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                config.label_emo_map[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)

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
    

    learning_rate = config.learning_rate
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    embedding_length = config.embedding_length
    nepoch = config.nepoch
    patience = config.patience
    arch_name = config.arch_name
    input_type = config.input_type
    max_seq_len = config.max_len
    output_size = 32
    log_dict = {}
    if config.embedding_type != "bert":
        log_dict["param"] = {"arch_name":arch_name,"lr":learning_rate,"hidden_size":hidden_size,"batch_size":batch_size,"embedding_length":embedding_length,"max_seq_len":max_seq_len}
    else:
        log_dict["param"] = {"arch_name":arch_name,"lr":learning_rate,"batch_size":batch_size,"max_seq_len":max_seq_len}

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
        model = BERT()

    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,config.step_size, gamma=0.5)
    best_acc1 = 0

    
    patience_flag = 0
    model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    writer = SummaryWriter('./runs/'+input_type+"/"+arch_name+"/")
    save_home = "./save/"+input_type+"/"+arch_name+"/"+model_run_time
    os.makedirs(save_home,exist_ok=True)
    

    for epoch in range(nepoch):
        train_loss, train_acc = train_model(model, train_iter, epoch)
        val_loss, val_acc = eval_model(model, valid_iter)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
        test_loss, test_acc = eval_model(model, test_iter,confusion=config.confusion,per_class=config.per_class)
        print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
        is_best = test_acc > best_acc1
        save_checkpoint({'epoch': epoch + 1,'arch': arch_name,'state_dict': model.state_dict(),'test_acc': test_acc,'train_acc':train_acc,"val_acc":val_acc,'param':log_dict["param"],'optimizer' : optimizer.state_dict(),},is_best,save_home+"/checkpoint.pth.tar")
        best_acc1 = max(test_acc, best_acc1)
        lr_scheduler.step()
        

        writer.add_scalar('Loss/train',train_loss,epoch)
        writer.add_scalar('Accuracy/train',train_acc,epoch)
        writer.add_scalar('Loss/val',val_loss,epoch)
        writer.add_scalar('Accuracy/val',val_acc,epoch)
        if is_best:
            patience_flag = 0
            log_dict["train_acc"] = train_acc
            log_dict["test_acc"] = test_acc
            log_dict["val_acc"] = val_acc
            log_dict["train_loss"] = train_loss
            log_dict["test_loss"] = test_loss
            log_dict["val_loss"] = val_loss
            log_dict["epoch"] = epoch+1
            
            with open(save_home+"/log.json", 'w') as fp:
                json.dump(log_dict, fp,indent=4)
            fp.close()
        else:
            patience_flag += 1

        if patience_flag == patience or epoch == nepoch-1:
            print(log_dict)
            break
