import re
import torch

def flatten_list(l):
    flattened_list=[item for sublist in l for item in sublist]
    return flattened_list

def tweet_preprocess(tweet):
    x_proc_i= [''.join([i if ord(i) < 128 else '' for i in text]) for text in x_i]
    x_proc_i = "".join(x_proc_i).replace(r'(RT|rt)[ ]*@[ ]*[\S]+',r'')

    ##
    ## <https... or www... --> URL
    x_proc_i = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ',x_proc_i)

    ## @<handle> --> USER_MENTION
    x_proc_i = re.sub(r'@[\S]+', 'USER_MENTION', x_proc_i)

    ## &amp; --> and
    x_proc_i = x_proc_i.replace(r'&amp;',r'and')

    ## #<hastag> --> <hastag>
    x_proc_i = re.sub(r'#(\S+)', r' \1 ', x_proc_i)

    ## remove rt --> space
    x_proc_i = re.sub(r'\brt\b', '', x_proc_i)

    ## remove more than 2 dots (..) --> space
    x_proc_i = re.sub(r'\.{2,}', ' ', x_proc_i)

    x_proc_i = x_proc_i.strip(' "\'')

    ## remove multiple space with single space
    x_proc_i = re.sub(r'\s+', ' ', x_proc_i)

    return x_proc_i

def save_checkpoint(state, is_best,filename):
    if is_best:
        torch.save(state,filename)

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
