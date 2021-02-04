arch_name = "kea_electra"
dataset = "ed"

learning_rate = 3e-05

batch_size = 1

if dataset == "ed":
    output_size = 32

hidden_size = 768 # for bert-based models, cannot change as fine-tuning

embedding_length = None

step_size = 10
start_epoch = 0 # for start training
nepoch = 6
patience = 30

# Accuracy display for  case study
confusion = False #confusion matrix
per_class = False # per class accuracy


param = {"arch_name":arch_name,"learning_rate":learning_rate,"batch_size":batch_size,"hidden_size":hidden_size,"output_size":output_size,"step_size":step_size,"dataset":dataset,"nepoch":nepoch,"confusion":confusion,"per_class":per_class,"patience":patience}

tuning = False ## if tuning == True, add the parameter list in train.py

