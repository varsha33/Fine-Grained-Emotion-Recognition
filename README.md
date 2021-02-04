# Fine-Grained Emotion Recognition

## Setup

For data preprocessing, run the following command

```
python preprocess.py
```
Note: For EmpatheticDialogue dataset, please use ed_data_extract.py to transform the data before preprocessing. 

For training the model, go to config.py/config_multilabel.py to set the required parameters. 

The training for this work was done entirely in Google Colab due to resource requirements. Use kea_singlelabel_colab_notebook for single label setting and kea_multilabel_colab notebook for multilabel settings. 

### Alternative

Follow the below instructions to use the python scripts

```
python train.py
```

## Requirements

Install the required packages mentioned in requirements.txt using pip.

```
pip install -r requirements.txt
```

## Credits

This application uses Open Source components. You can find the source code of their open source projects along with license information below. We acknowledge and are grateful to these developers for their contributions to open source.

1. Project: Text-Classification-Pytorch <https://github.com/prakashpandey9/Text-Classification-Pytorch>  
License <https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/LICENSE.txt>

2. Project:EmpatheticDialogues <https://github.com/facebookresearch/EmpatheticDialogues>
License <https://github.com/facebookresearch/EmpatheticDialogues/blob/master/LICENSE>

3. Project:GoEmotions <https://github.com/google-research/google-research/tree/master/goemotions>

4. Dataset:Affect in Tweets <https://competitions.codalab.org/competitions/17751#learn_the_details-datasets>

5. Project:MoEL <https://github.com/HLTCHKUST/MoEL>
License <https://github.com/HLTCHKUST/MoEL/blob/master/LICENSE>


## References

* [BERT Text Classification Using Pytorch](https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b)
