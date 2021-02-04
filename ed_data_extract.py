import pandas as pd
import torch
import os
import numpy as np
import pickle
import collections
import matplotlib.pyplot as plt
from label_dict import ed_label_dict, ed_emo_dict


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

    data = {'utterance_data':[],'emotion_label':[],'emotion':[],'prompt':[], 'utterance_data_combined':[],'utterance_id':[],"speaker_info":[],"speaker_utterance":[]}
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

                data["utterance_data"].append(ongoing_utterance_list)
                data["emotion_label"].append(emotion_label_str)
                data["emotion"].append(emotion_label_int)
                data["utterance_id"].append(ids)
                data["prompt"].append(prompt_str)
                data["speaker_info"].append(speaker_info)
                data["utterance_data_combined"].append("".join(ongoing_utterance_list))
                data["speaker_utterance"].append("".join(ongoing_utterance_list[0::2]))


        else:  # condition where it reaches the end of a conversation, so the prev_utterance was part of the previous conversation which is added to the ongoing utterance list

            prev_utterance_str = prev_utterance_parts[5].replace("_comma_", ",") #replace _comma_ for utterance
            emotion_label_str = prev_utterance_parts[2]
            prompt_str = prev_utterance_parts[3].replace("_comma_", ",")
            emotion_label_int = emo_map[prev_utterance_parts[2]]


            ongoing_utterance_list.append(prev_utterance_str)
            ids.append((prev_utterance_parts[0],prev_utterance_parts[1]))
            speaker_info.append(get_speaker_info(prev_utterance_parts[1]))

            data["utterance_data"].append(ongoing_utterance_list)
            data["emotion_label"].append(emotion_label_str)
            data["emotion"].append(emotion_label_int)
            data["utterance_id"].append(ids)
            data["prompt"].append(prompt_str)
            data["speaker_info"].append(speaker_info)
            data["utterance_data_combined"].append("".join(ongoing_utterance_list))
            data["speaker_utterance"].append("".join(ongoing_utterance_list[0::2]))

            ongoing_utterance_list = []
            ids = []
            speaker_info = []


    assert len(data["prompt"]) == len(data["emotion_label"]) == len(data["utterance_data"]) == len(data["utterance_id"]) == len(data["speaker_info"])

    save_data = {"prompt":data["prompt"],"emotion":data["emotion"],"utterance_data":data["utterance_data_combined"],"speaker_utterance":data["speaker_utterance"]}
    df = pd.DataFrame(save_data)
    df.to_csv("./.data/empathetic_dialogues/"+datatype+".csv",index=False)

def extract_data_stats():

    total_label_count,label_count_absolute,label_count_percentage = {},{},{}
    for i in ["train","test","valid"]:
        data_home = "./.data/empathetic_dialogues/"+i+".csv"
        df = pd.read_csv(data_home)
        counter=collections.Counter(df["emotion"])
        total_label_count[i] = len(df["emotion"])
        label_count_absolute =  {ed_emo_dict[k]: v for k, v in dict(counter).items()}
        label_count_percentage =  {k: round((v/total_label_count[i])*100,2) for k, v in label_count_absolute.items()}
        df = pd.DataFrame({"emotion":list(label_count_percentage.keys()),"percentage":list(label_count_percentage.values()),"absolute_value":list(label_count_absolute.values())})
        df.sort_values(by=['emotion'],inplace=True)
        df.to_csv("./.data/empathetic_dialogues/"+i+"_stats.csv",index=False)


if __name__ == '__main__':

    for i in ["train","test","valid"]:
        data_reader("./.data/raw/empatheticdialogues/",i)
    # extract_data_stats()
