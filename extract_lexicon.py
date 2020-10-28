import pandas as pd
from transformers import BertTokenizer,AutoTokenizer

##custom packages
from label_dict import arousal_dict,valence_dict,dom_dict

def get_arousal_vec(tokenizer,utterance):

	token_str = tokenizer.convert_ids_to_tokens(utterance) #converts numerical tokens to corresponding strings

	arousal_vec = [arousal_dict.get(i) for i in token_str]
	arousal_vec = [float(0) if v is None else float(v-0.5) for v in arousal_vec]  ## range from 0-1 is converted to -0.5 to 0.5

	return arousal_vec



def get_valence_vec(tokenizer,utterance):
	output_vec = []
	token_str = tokenizer.convert_ids_to_tokens(utterance) #converts numerical tokens to corresponding strings

	valence_vec = [valence_dict.get(i) for i in token_str]
	valence_vec = [float(0) if v is None else float(v-0.5) for v in valence_vec]  ## range from 0-1 is converted to -0.5 to 0.5

	return valence_vec


def get_dom_vec(tokenizer,utterance):
    output_vec = []
    token_str = tokenizer.convert_ids_to_tokens(utterance) #converts numerical tokens to corresponding strings

    dom_vec = [dom_dict.get(i) for i in token_str]
    dom_vec = [float(0) if v is None else float(v-0.5) for v in dom_vec]  ## range from 0-1 is converted to -0.5 to 0.5

    return dom_vec

