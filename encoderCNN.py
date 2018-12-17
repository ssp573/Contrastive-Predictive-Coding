import torch
import torch.nn as nn
import torch.nn.functional as F


class encoderCNN(nn.Module):
	def __init__(self,num_channels,code_size):
		super(encoderCNN, self).__init__()
		self.conv=nn.Conv2d(num_channels,code_size,kernel_size=8,stride=4)

	def forward(self,x):
		#print(self.conv(x).shape)
		return F.relu(self.conv(x))


