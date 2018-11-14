import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

code_size=128
hidden_size=64
gru_hidden_size=256


class Encoder:
	def __init__(self,code_size,hidden_size):
		super(Encoder, self).__init()
		self.conv1=nn.Conv2d(3,hidden_size,kernel_size=3,stride=2)
		self.conv2=nn.Conv2d(hidden_size,hidden_size,kernel_size=3,stride=2)
		self.conv3=nn.Conv2d(hidden_size,hidden_size,kernel_size=3,stride=2)
		self.conv4=nn.Conv2d(hidden_size,hidden_size,kernel_size=3,stride=2)
		self.conv_bn=nn.BatchNorm2d(64)
		sel.linear_bn=nn.BatchNorm2d(64)
		self.dropout=nn.Dropout2d(0.2)
		self.linear1=nn.linear(_,256)
		self.linear2=nn.linear(256,code_size)

	def forward(self,X):
		X=F.leaky_relu(self.conv_bn(self.conv1(X)))
		X=F.leaky_relu(self.conv_bn(self.conv2(X)))
		X=F.leaky_relu(self.conv_bn(self.conv3(X)))
		X=F.leaky_relu(self.conv_bn(self.conv4(X)))
		X=X.view(-1,64*_*_)
		X=self.leaky_relu(self.linear_bn(self.linear1(X)))
		X=self.linear2(X)

class Autoregressive:
	def __init__(self,gru_hidden_size,num_layers=1):
		self.num_layers,self.hidden_size=num_layers,gru_hidden_size
		self.rnn=nn.GRU(code_size,gru_hidden_size,batch_first=True)

	def forward(self,X):
		batch_size, seq_len = x.size()
		if torch.cuda.is_available():
			self.hidden=self.init_hidden(batch_size).cuda()
		else:
			self.hidden=self.init_hidden(batch_size)
		X , self.hidden = self.rnn(X,self.hidden)

		return X

	def init_hidden(self,batch_size):
		hidden=torch.randn(self.num_layers, batch_size, self.hidden_size)
		return hidden


