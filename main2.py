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
from pixel_RNN_style import *
from plotter import *
from resnet import *

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataroot', default="data/" ,help='path to dataset')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
help='input batch size for training (default: 64)')
parser.add_argument('--no_cuda', action='store_true', default=False,
help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
parser.add_argument('--dataset', type=str, default="CIFAR10", metavar='N',
help='Which dataset?(MNIST/CIFAR10)(Default: CIFAR10)')
args = parser.parse_args()

kernel_size = 8 #image width/464
overlap = 4 #overla
num_channels = 3

# use CUDA?
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
elif args.dataset=="LSUN":
        train_loader = torch.utils.data.DataLoader(
        datasets.LSUN("LSUN", classes=['dining_room_train','bridge_train','bedroom_train','church_outdoor_train','tower_train'],
        transform=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor()])),
batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
        datasets.LSUN("LSUN", classes=['dining_room_val','bridge_val','bedroom_val','church_outdoor_val','tower_val'], transform=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor()])),
batch_size=args.batch_size, shuffle=True, **kwargs)
        num_channels=3

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



encoder = ResNet101() #encoderCNN(num_channels,args.code_size)
autoregressor = Autoregressive_RNN(args.code_size,args.code_size,args.batch_size)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.lr, momentum=args.momentum)
autoregressor_optimizer = optim.SGD(autoregressor.parameters(), lr=args.lr, momentum=args.momentum)

def loss_compute(encoded,predicted):
	#print(encoded.shape)
	eye_shape=encoded.shape[0]
	#print(eye_shape)
	target=torch.eye(eye_shape).reshape(1,eye_shape,eye_shape).repeat(encoded.shape[1],1,1)
	#print(target.shape)
	if args.cuda:
		target=target.to("cuda")

	#print("diag_shape:",target.shape)
	m=nn.Softmax(dim=1)
	#loss_method=nn.BCELoss()
	loss_method=nn.BCEWithLogitsLoss()
	prod=torch.bmm(encoded.contiguous().view(encoded.shape[1],eye_shape,-1),predicted.view(predicted.shape[1],-1,eye_shape))
	#prod=m(prod)
	#print(softmaxed)
	#print(softmaxed.shape)
	return loss_method(prod,target)

def cropdata(data):
	data = data.unfold(2,kernel_size, overlap).unfold(3,kernel_size, overlap)
	data = data.contiguous().view(-1,num_channels,kernel_size,kernel_size)
	return data

def train():
	encoder.train()
	autoregressor.train()
	epoch_loss=0
	if args.cuda:
		encoder.to("cuda")
		autoregressor.to("cuda")
	for batch_idx,(data,_) in enumerate(train_loader):
		if args.cuda:
			data=data.to("cuda")
		data=Variable(data)
		encoder_optimizer.zero_grad()
		autoregressor_optimizer.zero_grad()
		data=cropdata(data)
		output = encoder(data)
		output= F.avg_pool2d(output,2).squeeze()
		# print(output)
		output = output.contiguous().view(args.batch_size,output.shape[0]//args.batch_size, output.shape[1]).permute(1,0,2)
		# print(output)

		# print(output.shape)
		# output = output.view(output.shape[0]//args.batch_size, args.batch_size, output.shape[1])
		# output=output.view(output.shape[-1]*output.shape[-1],output.shape[0],-1)
		# print(output.shape)
		# print("---")
		predicted_output=autoregressor(output)
		#print(predicted_output.shape)
		loss=loss_compute(output,predicted_output)
		epoch_loss+=loss.item()
		loss.backward()
		encoder_optimizer.step()
		autoregressor_optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))
	return epoch_loss/output.shape[0]

def validate():
	encoder.eval()
	autoregressor.eval()
	epoch_loss=0
	if args.cuda:
				encoder.to("cuda")
				autoregressor.to("cuda")
	for batch_idx,(data,_) in enumerate(test_loader):
		if args.cuda:
			data=data.to("cuda")
		data=Variable(data)
		with torch.no_grad():
			data=cropdata(data)
			output = encoder(data)
			output= F.avg_pool2d(output,2).squeeze()
			#print(output.shape)
			output = output.contiguous().view(args.batch_size,output.shape[0]//args.batch_size, output.shape[1]).permute(1,0,2)
			#print(output.shape)
			predicted_output=autoregressor(output)
			#print(predicted_output.shape)
			loss=loss_compute(output,predicted_output)
			epoch_loss+=loss.item()
			if batch_idx % args.log_interval == 0:
				print('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(test_loader.dataset),
				100. * batch_idx / len(test_loader), loss.item()))
	return epoch_loss/output.shape[0]


if args.train:
	max_val_loss=float("inf")
	all_train_losses=[]
	all_valid_losses=[]
	for epoch in range(1,args.epochs+1):
		train_loss=train()
		valid_loss=validate()
		all_train_losses.append(train_loss)
		all_valid_losses.append(valid_loss)
		print("\n\n\nEpoch Summary: Train Loss:",train_loss,"Valid loss:",valid_loss,"\n\n\n")
		if valid_loss<max_val_loss:
			max_val_loss=valid_loss
			torch.save(encoder.state_dict(),args.save_dir+"/cpc_encoder.pth")
	plot(all_train_losses,all_valid_losses)
