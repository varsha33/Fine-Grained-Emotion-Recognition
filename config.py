emo_label_map = {'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}

label_emo_map =  {v: k for k, v in emo_label_map.items()}


input_type = "speaker+listener"
tokenizer = "wordpiece+punkt"
embedding_type = "glove"
arch_name = "rcnn"

# tokenizer = "wordpiece+punkt"
# embedding_type = "bert"
# arch_name = "bert"


# learning_rate = 0.000023
learning_rate = 0.001
batch_size = 64
output_size = 32
hidden_size = 128
embedding_length = 200
nepoch = 20
max_len = 512
step_size = 5

patience = 700




# Accuracy display
confusion = False
per_class = False


# for eval.py
resume = "/home/ashvar/varsha/Text-Classification-Pytorch/save/speaker/transformer/2020_09_06_23_38_19/model_best.pth"
