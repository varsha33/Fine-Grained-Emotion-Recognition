
import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import dataset
# from load_data import ED_dataset,batchify,get_dataloader
from models.BERT import BERT,Speaker_Listener_BERT,Hierarchial_BERT,Hierarchial_BERT_wLSTM
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
import time

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = False

print('Loading dataset')
start_time = time.time()

if config.embedding_type == "glove":
    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset_glove()
elif config.embedding_type == "bert":
    # train_iter, valid_iter, test_iter = load_data.load_dataset_bert()
    train_iter, test_iter,valid_iter = dataset.get_dataloader(config.tokenizer)

finish_time = time.time()
print('Finished loading. Time taken:{:06.3f} sec'.format(finish_time-start_time))

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
    start_train_time = time.time()
    for idx, batch in enumerate(train_iter):
        # print(batch)
        speaker_text = batch["speaker_idata"]
        listener_text = batch["listener_idata"] 
        utterance_data_list = batch["utterance_data_list"]
        if (speaker_text.size()[0] is not config.batch_size):
            continue

  
        target = batch["emotion"]
        target = torch.Tensor(target)
        target = torch.autograd.Variable(target).long()

        if torch.cuda.is_available():
            speaker_text = speaker_text.cuda()
            listener_text = listener_text.cuda()
            utterance_data_list = utterance_data_list.cuda()
            target = target.cuda()


        optimizer.zero_grad()
        prediction = model(speaker_text,listener_text,utterance_data_list)
        loss = loss_fn(prediction, target)
        # print("T",target)
        # print("P",torch.max(prediction, 1)[1])
        num_corrects = float((torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum())
        
        acc = 100.0 * num_corrects/config.batch_size
        loss.backward()
        clip_gradient(model, 1e-1)  ## check this
        optimizer.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc: .2f}%, Time taken: {((time.time()-start_train_time)/60): .2f} min')
            start_train_time = time.time()
        total_epoch_loss += loss.item()
        total_epoch_acc += acc
        
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
            
            speaker_text = batch["speaker_idata"]
            listener_text = batch["listener_idata"] 
            utterance_data_list = batch["utterance_data_list"]

            if (speaker_text.size()[0] is not config.batch_size):
                continue

            target = batch["emotion"]
            target = torch.Tensor(target)
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                speaker_text = speaker_text.cuda()
                listener_text = listener_text.cuda()
                utterance_data_list = utterance_data_list.cuda()
                target = target.cuda()

            prediction = model(speaker_text,listener_text,utterance_data_list)
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

            acc = 100.0 * num_corrects/config.batch_size
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
    
    
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model,optimizer,start_epoch

if __name__ == '__main__':
    

    log_path = config.resume.replace("model_best.pth.tar","log.json")
    
    with open(log_path,'r') as f:
        log = json.load(f)
    f.close()
    
    learning_rate = log["param"]["learning_rate"]
    batch_size = log["param"]["batch_size"]
    input_type = log["param"]["input_type"]
    arch_name = log["param"]["arch_name"]
    hidden_size = log["param"]["hidden_size"]
    embedding_length = log["param"]["embedding_length"]
    log_dict = {}
    log_dict["param"] = log["param"]
    output_size = 32
    nepoch = config.nepoch

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
        model = BERT(output_size)
    elif arch_name == "sl_bert":
        model = Speaker_Listener_BERT(output_size,hidden_size)
    elif arch_name == "h_bert":
        model = Hierarchial_BERT(output_size,hidden_size)
    elif arch_name == "h_lstm_bert":
        model = Hierarchial_BERT_wLSTM(batch_size,output_size,hidden_size)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,config.step_size, gamma=0.5)
    best_acc1 = 0

    model,optimizer,start_epoch =load_model(config.resume,model,optimizer)
    patience_flag = 0
    model_run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    # writer = SummaryWriter('./runs/'+input_type+"/"+arch_name+"/")
    save_home = "./save/"+input_type+"/"+arch_name+"/loadrun_"+model_run_time
    


    for epoch in range(start_epoch,nepoch):

        train_loss, train_acc = train_model(model, train_iter, epoch)
        val_loss, val_acc = eval_model(model, valid_iter)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
        test_loss, test_acc = eval_model(model, test_iter,confusion=config.confusion,per_class=config.per_class)
        print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
        is_best = test_acc > best_acc1
    
        os.makedirs(save_home,exist_ok=True)
        save_checkpoint({'epoch': epoch + 1,'arch': arch_name,'state_dict': model.state_dict(),'test_acc': test_acc,'train_acc':train_acc,"val_acc":val_acc,'param':log_dict["param"],'optimizer' : optimizer.state_dict(),},is_best,save_home+"/checkpoint.pth.tar")
        best_acc1 = max(test_acc, best_acc1)
        lr_scheduler.step()
        

        # writer.add_scalar('Loss/train',train_loss,epoch)
        # writer.add_scalar('Accuracy/train',train_acc,epoch)
        # writer.add_scalar('Loss/val',val_loss,epoch)
        # writer.add_scalar('Accuracy/val',val_acc,epoch)
        if is_best:
            patience_flag = 0
            log_dict["train_acc"] = train_acc
            log_dict["test_acc"] = test_acc
            log_dict["val_acc"] = val_acc
            log_dict["train_loss"] = train_loss
            log_dict["test_loss"] = test_loss
            log_dict["val_loss"] = val_loss
            log_dict["epoch"] = [start_epoch,epoch+1]
            
            with open(save_home+"/log.json", 'w') as fp:
                json.dump(log_dict, fp,indent=4)
            fp.close()
        else:
            patience_flag += 1

        if patience_flag == patience or epoch == nepoch-1:
            print(log_dict)
            break
