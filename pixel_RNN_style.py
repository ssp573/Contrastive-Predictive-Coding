import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#input_size=1024
#hidden_size=1024
#code_size=1024
#batch_size=1

class Autoregressive_RNN(nn.Module):
	def __init__(self,input_size,hidden_size,batch_size,num_layers=1):
		super(Autoregressive_RNN, self).__init__()
		self.hidden_size=hidden_size
		self.batch_size=batch_size
		self.num_layers=num_layers
		self.rnn=nn.GRU(input_size,hidden_size,num_layers=self.num_layers)
		#self.linear=nn.Linear(hidden_size,code_size)

	def forward(self,X):
		#print(X.shape)
		return self.autoregress(X)

	def autoregress(self,X):
		hidden=self.init_hidden(X.shape[1])
		output,hidden=self.rnn(X,hidden)
		return output

	#def generate(self,X):
	#	return self.linear(X)

	def init_hidden(self,num_batches):
		return torch.zeros(self.num_layers,num_batches,self.hidden_size,device=device)
