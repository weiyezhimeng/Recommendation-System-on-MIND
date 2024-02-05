import tqdm
from load_data import load_data
from train import train 
import torch
from transformers import BertModel,BertTokenizer
import torch.nn as nn
epoch=1
batch=64
file_train="../MIND/MINDlarge_train/behaviors.tsv"
device='cuda:0'
tokenizer_path = model_path = f"../bert-mini"
lr=1e-5
tokenizer = BertTokenizer.from_pretrained(tokenizer_path, padding_side='left')   
model_bert=BertModel.from_pretrained(model_path)  
class BertClassifier(nn.Module):
    def __init__(self, ):
        """
        """
        super(BertClassifier, self).__init__()
        self.bert = model_bert
        self.news_layer = nn.Sequential(nn.Linear(256, 500),
                                        nn.Tanh(),  
                                        nn.Linear(500, 1),
                                        nn.Flatten(), nn.Softmax(dim=0)) 

        for name, param in self.bert.named_parameters():
            param.requires_grad = True
        for name, param in self.news_layer.named_parameters():
            param.requires_grad = True
    def forward(self, input_ids,attention_mask):
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        attention_weight = self.news_layer(last_hidden_state_cls)  
        new_emb = torch.sum(last_hidden_state_cls * attention_weight, dim=0)  
        return new_emb
model = BertClassifier().to(device)

loader_train=load_data(file_train,batch)

train(tokenizer,model,model_bert,device,lr,epoch,loader_train,batch)





