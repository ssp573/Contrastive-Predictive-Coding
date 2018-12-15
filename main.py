import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import argparse
from encoderCNN import *

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataroot', default="data/" ,help='path to dataset')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
help='input batch size for training (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
help='SGD momentum (default: 0.5)')
parser.add_argument('--train', default=True, action='store_true',
help='training a ConvNet model on MNIST dataset')
args = parser.parse_args()


# use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=args.dataroot, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=args.dataroot, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
batch_size=args.batch_size, shuffle=True, **kwargs)

model=encoderCNN()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train():
	model.train()
	for batch_idx,(data,_) in enumerate(train_loader):
		if args.cuda:
			data=data.to("cuda")
		data=Variable(data)
		optimizer.zero_grad()
		print(data.shape)
		output = model(data)
		print(output.shape)
		break

if args.train:
	for epoch in range(1,args.epochs+1):
		train()





