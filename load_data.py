import pandas as pd
from torch.utils.data import Dataset, DataLoader

class mydataset(Dataset):
	"""
	getitem return history's click and new to judge.
	"""
	def __init__(self,file):
		self.all = pd.read_csv(file, delimiter='\t',header=None)
		self.all = self.all.drop(self.all[pd.isna(self.all[3])].index)
	def __len__(self):
		return self.all.shape[0]
	def __getitem__(self, idx):
		return self.all.iloc[idx,3],self.all.iloc[idx,4]

def load_data(file,batch):
    train_dataset=mydataset(file)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    return train_loader



