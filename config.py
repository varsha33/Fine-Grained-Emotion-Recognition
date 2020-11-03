tokenizer = "bert"
input_type = "speaker+listener"
embedding_type = "bert"
arch_name = "kea_bert"

# learning_rate = 0.005 # for models until bert
learning_rate = 3e-05 # for bert-based models

batch_size = 20
# batch_size = 64 # for models until bert
output_size = 32

# hidden_size = 128 # for models until bert
hidden_size = 768 # for bert-based models, cannot change as fine-tuning
# hidden_size = 256 # electra

if embedding_type == "bert":
	embedding_length = None
else:
	embedding_length = 200 # used only for glove embedding type


max_seq_len = 512 # used only for glove embedding type

step_size = 10
start_epoch = 0 # for start training
nepoch = 3
patience = 30

# Accuracy display
confusion = False #confusion matrix
per_class = False # per class accuracy

#Only for freezing models
freeze = False
resume_path = "/home/ashvar/varsha/Emotion-Recognition/save/speaker+listener/bert/2020_10_22_20_18_02/model_best.pth.tar"


param = {"input_type":input_type,"tokenizer":tokenizer,"embedding_type":embedding_type,"arch_name":arch_name,"learning_rate":learning_rate,"batch_size":batch_size,"hidden_size":hidden_size,"embedding_length":embedding_length,"max_seq_len":max_seq_len,"output_size":output_size,"step_size":step_size,"freeze":False}

tuning = False
