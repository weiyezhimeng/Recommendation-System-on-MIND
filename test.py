from utils import test
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--s1', type=int, default = None)
    parser.add_argument('--e1', type=int, default= None)
    args = parser.parse_args()
    file="../MIND/MINDlarge_test/behaviors.tsv"
    tokenizer_path =  f"../bert-mini"
    model_path_bert = f"../bert-mini"
    device='cpu:0'
    class BertClassifier(nn.Module):
        def __init__(self, ):
            """
            freeze_bert (bool): 设置是否进行微调，0就是不，1就是调
            """
            super(BertClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(model_path_bert)
            self.news_layer = nn.Sequential(nn.Linear(256, 300),
                                            nn.Tanh(),  # 20 12 64
                                            nn.Linear(300, 1),
                                            nn.Flatten(), nn.Softmax(dim=0))  # 20 12

            for name, param in self.bert.named_parameters():
                param.requires_grad = False
            for name, param in self.news_layer.named_parameters():
                param.requires_grad = True
        def forward(self, input_ids,attention_mask):
            outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
            last_hidden_state_cls = outputs[0][:, 0, :]
            attention_weight = self.news_layer(last_hidden_state_cls)  # 64 12
            new_emb = torch.sum(last_hidden_state_cls * attention_weight, dim=0)  # 64 128
            return new_emb

    model = torch.load('./user.pth').to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, padding_side='left')                  
    model_bert = torch.load('./bert-news.pth').to(device)
    test(file,model,model_bert,tokenizer,device,args.s1,args.e1)
