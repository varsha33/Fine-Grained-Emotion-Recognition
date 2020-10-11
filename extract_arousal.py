import pandas as pd
from transformers import BertTokenizer,AutoTokenizer
from label_dict import arousal_dict,valence_dict

def get_arousal_vec(text,tokenizer,len_check,add_start=True,add_end=True):
	output_vec = []
	for i,val in enumerate(text):
		arousal_vec = list(map(arousal_dict.get,tokenizer.tokenize(val)))
		arousal_vec = [float(0) if v is None else float(v-0.5) for v in arousal_vec]
		if add_start:
			arousal_vec = [float(0)]+arousal_vec
		if add_end:
			arousal_vec = arousal_vec + [float(0)]
		# assert len(arousal_vec) == len(len_check[i])  ##checks whether the number of tokens is similar to the number of arousal values
		output_vec.append(arousal_vec)
	return output_vec



def get_valence_vec(text,tokenizer,len_check,add_start=True,add_end=True):
	output_vec = []
	for i,val in enumerate(text):
		valence_vec = list(map(valence_dict.get,tokenizer.tokenize(val)))
		valence_vec = [float(0) if v is None else float(v-0.5) for v in valence_vec]
		if add_start:
			valence_vec = [float(0)]+valence_vec
		if add_end:
			valence_vec = valence_vec + [float(0)]
		# assert len(valence_vec) == len(len_check[i])  ##checks whether the number of tokens is similar to the number of valence values
		output_vec.append(valence_vec)
	return output_vec
