from captum.attr import IntegratedGradients,LayerIntegratedGradients,TokenReferenceBase, visualization
import torch
import torch.nn as nn
from label_dict import label_emo_map
from transformers import BertTokenizer,AutoTokenizer

class model_wrapper(nn.Module):

    def __init__(self,model):
        super(model_wrapper, self).__init__()
 
        self.softmax = nn.Softmax(dim=1)
        self.model = model
    def forward(self, text): #here text is utterance based on the input type specified

        output = self.model(text)
        return self.softmax(output)

def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    # storing couple samples in an array for visualization purposes
    return visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            label_emo_map[pred_ind],
                            label_emo_map[label],
                            label_emo_map[pred_ind],
                            attributions.sum(),       
                            text,
                            delta)

def explain_model(model,binput_ids,btarget,binput_str,bpred_ind,bpred_softmax):
	tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
	vis_data_records_ig = []
	token_reference = TokenReferenceBase(reference_token_idx=0)
	model = model.cpu()
	model_explain = model_wrapper(model)
	
	for i,val in enumerate(binput_str):
		input_id = binput_ids[i,:].unsqueeze(0).cpu()
		seq_len = len(input_id.squeeze(0).tolist())
		target = btarget[i].item()
		pred_ind  = bpred_ind[i].item()
		pred_softmax = bpred_softmax[i,:][pred_ind].item()
		input_str = tokenizer.tokenize("".join(binput_str[i]))

		if label_emo_map[target] == "sentimental" or label_emo_map[target] == "nostalgic":
			if label_emo_map[pred_ind] == "nostalgic" or label_emo_map[pred_ind] == "sentimental":
				device = torch.device("cpu")
				reference_ids = token_reference.generate_reference(seq_len,device=device).unsqueeze(0)
				ig =LayerIntegratedGradients(model_explain,model_explain.model.encoder.bert.embeddings)
				attributions, delta = ig.attribute(input_id,reference_ids,target=target,n_steps=10,return_convergence_delta=True)

				# print('pred: ', label_emo_map[pred_ind], '(', '%.2f'%pred_softmax, ')', ', delta: ', abs(delta))
				vis_data_records_ig.append(add_attributions_to_visualizer(attributions,input_str,pred_softmax, pred_ind,target, delta))

	if len(vis_data_records_ig) != 0:
		visualization.visualize_text(vis_data_records_ig)
	# return vis_data_records_ig