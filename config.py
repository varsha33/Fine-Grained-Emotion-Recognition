tokenizer = ""
input_type = "speaker+listener"
embedding_type = "bert"
arch_name = "electra"

# learning_rate = 0.005 # for models until bert
learning_rate = 3e-05 # for bert-based models

batch_size = 12
# batch_size = 64 # for models until bert
output_size = 32

# hidden_size = 128 # for models until bert
hidden_size = 768 # for bert-based models, cannot change as fine-tuning
# hidden_size = 256 # electra


embedding_length = None

step_size = 10
start_epoch = 0 # for start training
nepoch = 5
patience = 30

# Accuracy display
confusion = False #confusion matrix
per_class = False # per class accuracy


param = {"input_type":input_type,"tokenizer":tokenizer,"embedding_type":embedding_type,"arch_name":arch_name,"learning_rate":learning_rate,"batch_size":batch_size,"hidden_size":hidden_size,"embedding_length":embedding_length,"max_seq_len":max_seq_len,"output_size":output_size,"step_size":step_size,"freeze":False}

tuning = False
