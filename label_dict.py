ed_label_dict = {'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}

ed_emo_dict =  {v: k for k, v in ed_label_dict.items()}

## this is for confusion matrix and for the case study for EmpatheticDialogues
class_names = list(ed_label_dict.keys())
class_indices = list(range(0,32))

filepath = './.data/nrc_vad/a-scores.txt'
arousal_dict = {}
with open(filepath) as fp:
	for cnt, line in enumerate(fp):
		arousal_dict[line.split("\t")[0]] = float(line.split("\t")[1].split("\n")[0])
fp.close()


filepath = './.data/nrc_vad/v-scores.txt'
valence_dict = {}
with open(filepath) as fp:
    for cnt, line in enumerate(fp):
        valence_dict[line.split("\t")[0]] = float(line.split("\t")[1].split("\n")[0])
fp.close()


filepath = './.data/nrc_vad/d-scores.txt'
dom_dict = {}
with open(filepath) as fp:
    for cnt, line in enumerate(fp):
        dom_dict[line.split("\t")[0]] = float(line.split("\t")[1].split("\n")[0])
fp.close()

goemotions_label_dict= {"admiration":0,"amusement":1,"anger":2, "annoyance":3,"approval":4,"caring":5,"confusion":6,"curiosity":7,"desire":8,"disappointment":9,"disapproval":10,"disgust":11,"embarrassment":12,"excitement":13,"fear":14,"gratitude":15,"grief":16,"joy":17,"love":18,"nervousness":19,"optimism":20,"pride":21,"realization":22,"relief":23,"remorse":24,"sadness":25,"surprise":26,"neutral":27}
goemotions_emo_dict= {v: k for k, v in goemotions_label_dict.items()}

semeval_label_dict= {"anger":0, "anticipation":1, "disgust":2, "fear":3,"joy":4,"love":5,"optimism":6, "pessimism":7,"sadness":8,"surprise":9,"trust":10}
semeval_emo_dict = {v: k for k, v in semeval_label_dict.items()}
