import pandas as pd
from transformers import BertTokenizer,AutoTokenizer
from label_dict import arousal_dict

def get_arousal_vec(text,tokenizer,len_check):
	output_vec = []
	for i,val in enumerate(text):
		arousal_vec = list(map(arousal_dict.get,tokenizer.tokenize(val)))
		arousal_vec = [0.5]+[float(0.5) if v is None else v for v in arousal_vec]+[0.5]
		assert len(arousal_vec) == len(len_check[i])
		output_vec.append(arousal_vec)
	return output_vec
	

