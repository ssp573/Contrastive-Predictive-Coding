import torch
import torch.nn as nn
import torch.nn.functional as F


class encoderCNN(nn.Module):
	def __init__(self):
		super(encoderCNN, self).__init__()
		self.conv=nn.Conv2d(1,1024,kernel_size=8,stride=4)

	def forward(self,x):
		return self.conv(x)


