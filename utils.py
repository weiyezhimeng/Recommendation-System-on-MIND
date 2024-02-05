import pandas as pd
import torch
import random
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gc
from GPU import GPU
from tqdm import *
import os
news = pd.read_csv("../MIND/MINDlarge_train/news.tsv", delimiter='\s*\t\s*',header=None,index_col=0)
def reflect_news_to_str(example):
    """
    input news' index <str>.
    output news' string <str>.
    """
    if example=="None":
        return example
    return news.loc[example][3]
def label_handle(label,batch):
	"""
	input news' label <tuple>.[batch,?]
	output news' string <list>.label1[batch,[news1,news2,...]] label0[batch,[news1,news2,...]]
	"""
	label_1=[]
	label_0=[]
	for i in range(batch):
		#处理每一个batch中的一组
		label_1_temp=[]
		label_0_temp=[]
		label_all=label[i].split()
		for j in range(len(label_all)):
			if label_all[j][-1]=="1":
				label_1_temp.append(label_all[j][:-2])
			if label_all[j][-1]=="0":
				label_0_temp.append(label_all[j][:-2])
		#转换成字符串
		for j in range(len(label_1_temp)):
			label_1_temp[j]=reflect_news_to_str(label_1_temp[j])
		for j in range(len(label_0_temp)):
			label_0_temp[j]=reflect_news_to_str(label_0_temp[j])
		label_1.append(label_1_temp)
		label_0.append(label_0_temp)
	return label_1,label_0
def loss(history,label,batch,model,model_bert,tokenizer,device):
	num=0
	loss_all=torch.tensor(0.0).to(device)
	label_1,label_0=label_handle(label,batch)
	for i in range(batch):

		#计算user的历史信息
		prompt_user=[]
		history_split=history[i].split()
		for j in range(len(history_split)-1,-1,-1):
			prompt_user.append(reflect_news_to_str(history_split[j]))
		user_tokens=tokenizer(prompt_user, return_tensors="pt", padding=True)
		input_ids=user_tokens['input_ids'].to(device)
		attention_mask=user_tokens['attention_mask'].to(device)
		logits_user=model(input_ids,attention_mask=attention_mask)

		#随机取4个0标签
		if len(label_0[i])>4:
			label_0[i]=random.sample(label_0[i],4)

		#把所有需要计算的样本合并成一个tensor
		prompt_1_and_0=label_1[i].copy()#深拷贝
		prompt_0=label_0[i].copy()
		prompt_1_and_0.extend(prompt_0)

		#计算预测目标news的编码信息
		target_tokens_1_0 = tokenizer(prompt_1_and_0, return_tensors="pt", padding=True)
		input_ids_1_0=target_tokens_1_0['input_ids'].to(device)
		attention_mask_1_0=target_tokens_1_0['attention_mask'].to(device)
		logits_1_0=model_bert(input_ids=input_ids_1_0,attention_mask=attention_mask_1_0)[0][:, 0, :]

		#前len(label_1)个是1的结果，前len(label_0)个是0的结果
		#算1的
		for j in range(len(label_1[i])):
			if j==0:
				score_1=torch.dot(logits_1_0[j],logits_user).unsqueeze(0)
			else:
				temp=torch.dot(logits_1_0[j],logits_user).unsqueeze(0)
				score_1=torch.cat((score_1,temp),dim=0)

		#算0的
		for j in range(len(label_0[i])):
			if j==0:
				score_0=torch.dot(logits_1_0[j+len(label_1[i])],logits_user).unsqueeze(0)
			else:
				temp=torch.dot(logits_1_0[j+len(label_1[i])],logits_user).unsqueeze(0)
				score_0=torch.cat((score_0,temp),dim=0)

		#每一个1样本都计算一下结果，拼接一个1和所有0
		for j in range(len(label_1[i])):
			temp=score_1[j].unsqueeze(0)
			temp_for_loss=torch.cat((temp,score_0),dim=0)
			loss_all-=torch.log(F.softmax(temp_for_loss-torch.max(temp_for_loss),dim=0)[0])
			#print(F.softmax(temp_for_loss-torch.max(temp_for_loss),dim=0))
		num+=len(label_1[i])
	return loss_all/num

def test(file,model,model_bert,tokenizer,device,start,end):
	result=[]
	a=pd.read_csv(file, delimiter='\t',header=None)
	for i in tqdm(range(start,end)):
		with torch.no_grad():
			#选取数据
			index=a.iloc[i,0]
			history=a.iloc[i,3]
			if not isinstance(history,str):
				history="None"
			history=history.split()
			target=a.iloc[i,4]
			target=target.split()
			sort_all=[]
			prompt_user=[]
			prompt_target=[]

			#获取user的编码信息
			for j in range(len(history)):
				prompt_user.append(reflect_news_to_str(history[j]))
			user_tokens=tokenizer(prompt_user, return_tensors="pt", padding=True)
			input_ids=user_tokens['input_ids'].to(device)
			attention_mask=user_tokens['attention_mask'].to(device)
			logits_user=model(input_ids,attention_mask=attention_mask)

			#获取样本的编码信息
			for j in range(len(target)):
				prompt_target.append(reflect_news_to_str(target[j]))
			target_tokens = tokenizer(prompt_target, return_tensors="pt", padding=True)
			a_for=target_tokens['input_ids'].to(device)
			b_for=target_tokens['attention_mask'].to(device)
			logits=model_bert(input_ids=a_for,attention_mask=b_for)[0][:, 0, :]

			#和user的编码信息点乘，计算结果
			for j in range(len(target)):
				loss_predict_1 = torch.dot(logits[j],logits_user)
				sort_all.append(loss_predict_1.item())
			prediction = np.array(sort_all)
			result_sort = np.argsort(-prediction)
			result_final=np.empty([len(result_sort)],dtype = int) 
			for j in range(len(result_sort)):
				result_final[result_sort[j]]=j+1
			wow=str(index)+" ["+','.join(str(i) for i in result_final)+"]\n"
			result.append(wow)
	filename=f"result_{start}_{end}.txt"
	with open(filename, 'w') as f:
		f.writelines(result)
