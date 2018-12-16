import torch
import torch.nn as nn
import torch.nn.functional as F


class encoderCNN(nn.Module):
	def __init__(self,code_size):
		super(encoderCNN, self).__init__()
		self.conv=nn.Conv2d(1,code_size,kernel_size=8,stride=4)

	def forward(self,x):
		return F.relu(self.conv(x))


