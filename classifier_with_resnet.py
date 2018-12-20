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
from resnet import *

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataroot', default="data/" ,help='path to dataset')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
help='input batch size for training (default: 64)')
parser.add_argument('--no_cuda', action='store_true', default=False,
help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
parser.add_argument('--dataset', type=str, default="CIFAR10", metavar='N',
help='Which dataset?(MNIST/CIFAR10)(Default: MNIST)')
parser.add_argument('--use_cpc', type=bool, default="True", metavar='N',
help='Use CPC Features?(Default:True) set to False to use pixels flattened image.')
args = parser.parse_args()

args.use_cpc = True
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


if args.dataset=="MNIST":
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
        num_channels=1
else:
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.dataroot, train=True, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])),
batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.dataroot, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
batch_size=args.batch_size, shuffle=True, **kwargs)
        num_channels=3

encoder = ResNet101()
if args.cuda:
	encoder.to("cuda")
encoder.load_state_dict(torch.load(args.save_dir+"/cpc_encoder.pth"))

for param in encoder.parameters():
    param.requires_grad = False

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier,self).__init__()
#		args.use_cpc = False
		if args.use_cpc:
			self.linear=nn.Linear(args.code_size,10)
		else:
			self.linear=nn.Linear(28*28*1,10)
			#self.linear=nn.Linear(32*32*3,10)
	def forward(self,x):
		x=self.linear(x)
		return F.log_softmax(x)

model=Classifier()
if args.cuda:
	model.to('cuda')

optimizer=optim.Adam(model.parameters(), lr=0.001)

data_backup_list=[]
# def generateFeatures():
# 	print("backing up")
# 	for batch_idx,(data,target) in enumerate(train_loader):
# 		if args.cuda:
# 			data,target=data.to("cuda"),target.to('cuda')
# #		args.use_cpc = False
# 		if args.use_cpc:
# 			data=cropdata(data)
# 			data = encoder(data)
# 			data= F.avg_pool2d(data,2).squeeze()
# 			data = data.view(data.shape[0]//args.batch_size, args.batch_size, data.shape[1])
# 			data = torch.mean(data,0)
# 			data_backup_list.append(data)
# 	print("backup done")

def train():
	model.train()
	for batch_idx,(data,target) in enumerate(train_loader):
		if args.cuda:
			data,target=data.to("cuda"),target.to('cuda')
#		args.use_cpc = False
		if args.use_cpc:
			data=cropdata(data)
			data = encoder(data)
			data= F.avg_pool2d(data,2).squeeze()
			data = data.view(data.shape[0]//args.batch_size, args.batch_size, data.shape[1])

			# print(data.shape)
			# data_backup_list[batch_idx]
			data = torch.mean(data,0)

			# print(data.shape)
			# data=torch.mean(torch.mean(data,-1),-1)
			# print(data.shape)
		else:
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
#		args.use_cpc = False
		if args.use_cpc:
			data=cropdata(data)
			data = encoder(data)
			data= F.avg_pool2d(data,2).squeeze()
			data = data.view(data.shape[0]//args.batch_size, args.batch_size, data.shape[1])
			# print(data.shape)

			data = torch.mean(data,0)

			# print(data.shape)
			# data=torch.mean(torch.mean(data,-1),-1)
			# #print(data.shape)
		else:
			data=data.view(data.shape[0],-1)
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
	# generateFeatures()
	for epoch in range(1,args.epochs+1):
		train()
		validate()
