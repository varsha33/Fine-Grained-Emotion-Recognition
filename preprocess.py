import os
import pandas as pd
import numpy as np
import pickle
import argparse

## torch packages
import torch
from transformers import BertTokenizer,AutoTokenizer
import re
## for visualisation
import matplotlib.pyplot as plt
import collections

## custom packages
from extract_lexicon import get_arousal_vec,get_valence_vec,get_dom_vec
from utils import flatten_list,tweet_preprocess
from label_dict import ed_label_dict as emo_map
from label_dict import ed_emo_dict as emo_map_inverse

def get_one_hot(emo, class_size):

    targets = np.zeros(class_size)
    emo_list = [int(e) for e in emo.split(",")]
    for e in emo_list:
        targets[e] = 1
    return list(targets)


def get_speaker_info(speaker_id):
    if int(speaker_id) % 2 == 0:
        speaker = 1 # listener utterance
    else:
        speaker = 0  # speaker utterance
    return speaker

def data_reader(data_folder, datatype,save=True):

    '''
    Reads the raw data from EmpatheticDialogues dataset, preprocess the data and save it in a pickle file
    '''
    print("Datatype:",datatype)

    ongoing_utterance_list = []
    ids = []
    speaker_info = []

    data = {'prompt':[],'utterance_data_list':[],'utterance_data':[],'utterance_id':[],"speaker_info":[],'emotion_label':[],'emotion':[]}
    df = open(os.path.join(data_folder, f"{datatype}.csv")).readlines()

    for i in range(2,len(df)): # starts with 2 becauase df[0] is the coloumn headers, so i-1 i.e. 2-1=1 will start from the actual data

        prev_utterance_parts = df[i-1].strip().split(",")
        current_utterance_parts = df[i].strip().split(",")

        if prev_utterance_parts[0] == current_utterance_parts[0]: #to detect if its the ongoing conversation or the next conversation
            prev_utterance_str = prev_utterance_parts[5].replace("_comma_", ",") #replace _comma_ for utterance

            ongoing_utterance_list.append(prev_utterance_str)
            ids.append((prev_utterance_parts[0],prev_utterance_parts[1]))
            speaker_info.append(get_speaker_info(prev_utterance_parts[1]))

            if i == len(df)-1 : # reaches the end of the dataset and this adds the last utterance to the ongoing utterance list


                current_utterance_str = current_utterance_parts[5].replace("_comma_", ",") #replace _comma_ for utterance
                emotion_label_str = current_utterance_parts[2]
                prompt_str = current_utterance_parts[3].replace("_comma_", ",")
                emotion_label_int = emo_map[current_utterance_parts[2]]

                ongoing_utterance_list.append(current_utterance_str)
                ids.append((current_utterance_parts[0],current_utterance_parts[1]))
                speaker_info.append(get_speaker_info(current_utterance_parts[1]))

                data["prompt"].append(prompt_str)
                data["utterance_data_list"].append(ongoing_utterance_list)
                data["utterance_data"].append("".join(ongoing_utterance_list))
                data["utterance_id"].append(ids)
                data["speaker_info"].append(speaker_info)
                data["emotion_label"].append(emotion_label_str)
                data["emotion"].append(emotion_label_int)


        else:  # condition where it reaches the end of a conversation, so the prev_utterance was part of the previous conversation which is added to the ongoing utterance list

            prev_utterance_str = prev_utterance_parts[5].replace("_comma_", ",") #replace _comma_ for utterance
            emotion_label_str = prev_utterance_parts[2]
            prompt_str = prev_utterance_parts[3].replace("_comma_", ",")
            emotion_label_int = emo_map[prev_utterance_parts[2]]


            ongoing_utterance_list.append(prev_utterance_str)
            ids.append((prev_utterance_parts[0],prev_utterance_parts[1]))
            speaker_info.append(get_speaker_info(prev_utterance_parts[1]))

            data["prompt"].append(prompt_str)
            data["utterance_data_list"].append(ongoing_utterance_list)
            data["utterance_data"].append("".join(ongoing_utterance_list))
            data["utterance_id"].append(ids)
            data["speaker_info"].append(speaker_info)
            data["emotion_label"].append(emotion_label_str)
            data["emotion"].append(emotion_label_int)

            ongoing_utterance_list = []
            ongoing_utterance_inter_list = []
            ids = []
            speaker_info = []

    processed_data = {"prompt":data["prompt"],"utterance_data_list":data["utterance_data_list"],"utterance_data":data["utterance_data"],"speaker_info":data["speaker_info"],"emotion":data["emotion"]}

    return processed_data


def tokenize_data(processed_data,tokenizer_type="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    tokenized_inter_speaker, tokenized_inter_listener = [],[]
    tokenized_total_data,tokenized_speaker,tokenized_listener = [],[],[]
    tokenized_list_data,tokenized_turn_data = [],[]
    arousal_data,valence_data,dom_data = [],[],[]

    for u,val_utterance in enumerate(processed_data["utterance_data_list"]): #val utterance is one conversation which has multiple utterances

        tokenized_i= tokenizer.batch_encode_plus(val_utterance,add_special_tokens=False)["input_ids"]

        speaker_utterance,listener_utterance,speaker_iutterance,listener_iutterance,total_utterance = [101],[101],[101],[101],[101]

        total_utterance_list = []

        for s,val_speaker in enumerate(tokenized_i): ## for each utterance inside a conversation

            if s%2 == 0: # when person is the "speaker"
                speaker_utterance.extend(val_speaker+[102])
                speaker_iutterance.extend(val_speaker+[102])
                listener_iutterance.extend([0 for _ in range(len(val_speaker))]+[102])
    #
            else:
                listener_utterance.extend(val_speaker+[102])
                listener_iutterance.extend(val_speaker+[102])
                speaker_iutterance.extend([0 for _ in range(len(val_speaker))]+[102])

            total_utterance.extend(val_speaker+[102])
            total_utterance_list.append(val_speaker+[102])


        turn_data = [[101]+a+b for a, b in zip(total_utterance_list[::2],total_utterance_list[1::2])] # turnwise data, [[s1],[l1],[s2],[l2],..] --> [[s1;l1],[s2;l2],..]

        total_utterance_list = [[101]+i for i in total_utterance_list] #appending 101 to every utterance start

        arousal_vec = get_arousal_vec(tokenizer,total_utterance)
        valence_vec = get_valence_vec(tokenizer,total_utterance)
        dom_vec = get_dom_vec(tokenizer,total_utterance)


        tokenized_inter_speaker.append(speaker_iutterance)
        tokenized_inter_listener.append(listener_iutterance)

        tokenized_speaker.append(speaker_utterance)
        tokenized_listener.append(listener_utterance)
        tokenized_total_data.append(total_utterance)

        tokenized_list_data.append(total_utterance_list)
        tokenized_turn_data.append(turn_data)

        arousal_data.append(arousal_vec)
        valence_data.append(valence_vec)
        dom_data.append(dom_vec)


    assert len(tokenized_list_data) == len(tokenized_turn_data) ==len(tokenized_inter_speaker) == len(tokenized_inter_listener) == len(tokenized_total_data) ==len(tokenized_listener) ==len(tokenized_speaker) == len(processed_data["emotion"]) == len(tokenized_total_data) == len(arousal_data) == len(valence_data) == len(dom_data)

    save_data = {"utterance_data_list":tokenized_list_data,"utterance_data":tokenized_total_data,"utterance_data_str":processed_data["utterance_data_list"],"speaker_idata":tokenized_inter_speaker,"listener_idata":tokenized_inter_listener,"speaker_data":tokenized_speaker,"listener_data":tokenized_listener,"turn_data":tokenized_turn_data,"arousal_data":arousal_data,"valence_data":valence_data,"dom_data":dom_data,"emotion":processed_data["emotion"]}

    return save_data


def go_emotions_preprocess(tokenizer_type="bert-base-uncased"):
        data_dict = {}
        data_home = "./.data/goemotions/"
        nlabel = 28

        for datatype in ["train","valid","test"]:
            datafile = data_home + datatype + ".tsv"
            ## cause => tweet, changed for uniformity sake
            data = pd.read_csv(datafile, sep='\t',names=["cause","emotion","user"])

            emotion,cause = [],[]

            for i,emo in enumerate(data["emotion"]):
                emotion.append(get_one_hot(emo,nlabel))
                cause.append(data["cause"][i])

            print("Tokenizing data")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
            tokenized_cause =tokenizer.batch_encode_plus(cause).input_ids

            processed_data = {}
            maximum_utterance = max([len(i) for i in tokenized_cause])
            average_utterance = np.mean([len(i) for i in tokenized_cause])
            print(len(cause),len(emotion),len(tokenized_cause))
            print("Max utterance length:",maximum_utterance,"Avg utterance length:",average_utterance)


            ## changed prompt --> cause for uniformity
            processed_data["tokenized_cause"] = tokenized_cause
            processed_data["emotion"] = emotion
            processed_data["cause"] = cause

            arousal_vec,valence_vec,dom_vec = [],[],[]
            for cause_i in tokenized_cause:
                arousal_vec.append(get_arousal_vec(tokenizer,cause_i))
                valence_vec.append(get_valence_vec(tokenizer,cause_i))
                dom_vec.append(get_dom_vec(tokenizer,cause_i))


            processed_data["arousal_data"] = arousal_vec
            processed_data["valence_data"] = valence_vec
            processed_data["dom_data"] = dom_vec

            processed_data = pd.DataFrame.from_dict(processed_data)
            data_dict[datatype] = processed_data

            print(len(emotion),len(tokenized_cause),len(arousal_vec),len(valence_vec),len(dom_vec))
        if tokenizer_type == "bert-base-uncased":
            with open("./.preprocessed_data/goemotions_preprocessed_bert.pkl", 'wb') as f:
                pickle.dump(data_dict, f)
            f.close()

def sem_eval_preprocess(tokenizer_type):

    data_dict = {}
    for datatype in ["train","valid","test"]:

        with open("./.data/sem_eval/"+datatype+".txt", 'r') as fd:
            data = [l.strip().split('\t') for l in fd.readlines()][1:]

        X = [d[1] for d in data]

        y = [[int(d) for d in d[2:]] for d in data]


        # return X, y

        cause,emotion = [],[]
        count = 0

        for i,x_i in enumerate(X):

            ## Affect in Tweets preprocessing in the utils.py
            cause.append(tweet_preprocess(x_proc_i))
            emotion.append(y[i])

        print("Tokenizing data")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        tokenized_cause =tokenizer.batch_encode_plus(cause).input_ids

        processed_data = {}
        maximum_utterance = max([len(i) for i in tokenized_cause])
        average_utterance = np.mean([len(i) for i in tokenized_cause])
        print(len(cause),len(emotion),len(tokenized_cause))
        print("Max utterance length:",maximum_utterance,"Avg utterance length:",average_utterance)


        ## changed prompt --> cause for uniformity
        processed_data["tokenized_cause"] = tokenized_cause
        processed_data["emotion"] = emotion
        processed_data["cause"] = cause

        arousal_vec,valence_vec,dom_vec = [],[],[]
        for cause_i in tokenized_cause:
            arousal_vec.append(get_arousal_vec(tokenizer,cause_i))
            valence_vec.append(get_valence_vec(tokenizer,cause_i))
            dom_vec.append(get_dom_vec(tokenizer,cause_i))


        processed_data["arousal_data"] = arousal_vec
        processed_data["valence_data"] = valence_vec
        processed_data["dom_data"] = dom_vec

        processed_data = pd.DataFrame.from_dict(processed_data)
        data_dict[datatype] = processed_data

        print(len(emotion),len(tokenized_cause),len(arousal_vec),len(valence_vec),len(dom_vec))

        if tokenizer_type == "bert-base-uncased":
            with open("./.preprocessed_data/semeval_preprocessed_bert.pkl", 'wb') as f:
                pickle.dump(data_dict, f)
            f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Enter tokenizer type')

    parser.add_argument('-t', default="bert-base-uncased",type=str,
                   help='Enter tokenizer type')
    parser.add_argument('-d', default="goemotions",type=str,
                   help='Enter dataset')

    args = parser.parse_args()
    tokenizer_type = args.t

    if args.d == "ed":
        train_pdata = data_reader("./.data/raw/empatheticdialogues/","train")
        valid_pdata = data_reader("./.data/raw/empatheticdialogues/","valid")
        test_pdata = data_reader("./.data/raw/empatheticdialogues/","test")

        train_save_data = tokenize_data(train_pdata,tokenizer_type)
        valid_save_data = tokenize_data(valid_pdata,tokenizer_type)
        test_save_data = tokenize_data(test_pdata,tokenizer_type)

        ## used previously during model design
        glove_vocab_size = 0
        glove_word_embeddings = []

        if tokenizer_type == "bert-base-uncased":
            with open('./.preprocessed_data/ed_dataset_preproc.p', "wb") as f:
                pickle.dump([train_save_data, valid_save_data, test_save_data, glove_vocab_size,glove_word_embeddings], f)
                print("Saved PICKLE")
    elif args.d == "goemotions":
        go_emotions_preprocess(tokenizer_type)
    elif args.d == "semeval":
        sem_eval_preprocess(tokenizer_type)












