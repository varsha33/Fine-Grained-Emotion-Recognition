tokenizer = "bert"
input_type = "speaker+listener"
embedding_type = "bert"
arch_name = "a_bert"


learning_rate = 0.0001 # for models until bert
# learning_rate = 2.3e-05 # for bert-based models

batch_size = 16
# batch_size = 64 # for models until bert (128 also works well)
output_size = 32

# hidden_size = 128 # for models until bert (256 also works well)
hidden_size = 768 # for bert-based models, cannot change as fine-tuning


if embedding_type == "bert":
	embedding_length = None
else:
	embedding_length = 200 # used only for glove embedding type


max_seq_len = 500 # used only for glove embedding type

step_size = 2
start_epoch = 0 # for start training
nepoch = 20
patience = 10
param = {"input_type":input_type,"tokenizer":tokenizer,"embedding_type":embedding_type,"arch_name":arch_name,"learning_rate":learning_rate,"batch_size":batch_size,"hidden_size":hidden_size,"embedding_length":embedding_length,"max_seq_len":max_seq_len,"output_size":output_size,"step_size":step_size}

# Accuracy display
confusion = False
per_class = False

#Only for freezing models
freeze= True
resume_path = "/home/ashvar/varsha/Emotion-Recognition/save/speaker+listener/bert/2020_09_27_19_49_06/model_best.pth.tar"


