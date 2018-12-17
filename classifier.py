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
parser.add_argument('--no_cuda', action='store_true', default=False,
help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
help='number of epochs to train (default: 10)')
parser.add_argument('--code_size', type=int, default=1024, metavar='N',
help='Encoded size (default: 256)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
help='SGD momentum (default: 0.5)')
parser.add_argument('--train', default=True, action='store_true',
help='training a ConvNet model on MNIST dataset')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
help='how many batches to wait before logging training status')
parser.add_argument('--save_dir', type=str, default="cpc_model", metavar='N',
help='Where to save the encoder?')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
help='input batch size for testing (default: 1000)')
args = parser.parse_args()

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
batch_size=args.test_batch_size, shuffle=True, **kwargs)

encoder = encoderCNN(args.code_size)
if args.cuda:
	encoder.to("cuda")
encoder.load_state_dict(torch.load(args.save_dir+"/cpc_encoder.pth"))

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier,self).__init__()
		self.linear=nn.Linear(28*28*1,10)
	def forward(self,x):
		x=self.linear(x)
		return F.log_softmax(x)

model=Classifier()
if args.cuda:
	model.to('cuda')

optimizer=optim.Adam(model.parameters(), lr=0.001)

def train():
	model.train()
	for batch_idx,(data,target) in enumerate(train_loader):
		if args.cuda:
			data,target=data.to("cuda"),target.to('cuda')
		data=encoder(data)
		#print(data.shape)
		data=torch.mean(torch.mean(data,-1),-1)
		#print(data.shape)
		data=data.view(data.shape[0],-1)
		data,target=Variable(data),Variable(target)
		optimizer.zero_grad()
		#print(data.shape)
		output = model(data)
		#print(output.shape)
		loss=F.nll_loss(output,target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def validate():
	model.eval()
	correct=0
	test_loss=0
	for batch,(data,target) in enumerate(test_loader):
		if args.cuda:
			data,target=data.to("cuda"),target.to('cuda')
		data=encoder(data)
		#print(data.shape)
		data=torch.mean(torch.mean(data,-1),-1)
		#print(data.shape)
		#data=data.view(data.shape[0],-1)
		data,target=Variable(data),Variable(target)
		with torch.no_grad():
			output = model(data)
			test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
			pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	test_loss, correct, len(test_loader.dataset),
	100. * correct / len(test_loader.dataset)))

if args.train:
	for epoch in range(1,args.epochs+1):
		train()
		validate()
