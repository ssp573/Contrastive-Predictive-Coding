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
#from sklearn.cluster import KMeans,preprocessing
from encoderCNN import *
from plotter import *
import pickle as pkl

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
parser.add_argument('--dataset', type=str, default="MNIST", metavar='N',
help='Which dataset?(MNIST/CIFAR10)(Default: MNIST)')
parser.add_argument('--use_cpc', type=str, default="True", metavar='N',
help='Use CPC Features?(Default:True) set to False to use pixels flattened image.')
args = parser.parse_args()


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


encoder = encoderCNN(num_channels,args.code_size)
if args.cuda:
	encoder.to("cuda")
	encoder.load_state_dict(torch.load(args.save_dir+"/cpc_encoder.pth"))
encoder.load_state_dict(torch.load(args.save_dir+"/cpc_encoder.pth",map_location="cpu"))

def get_features_cpc(data_loader):
	#model.eval()
	cpc_features=[]
	for batch_idx,(data,target) in enumerate(data_loader):
		if args.cuda:
			data,target=data.to("cuda"),target.to('cuda')
		if args.use_cpc:
			with torch.no_grad():
				data=encoder(data)
				#print(data.shape)
				data=torch.mean(torch.mean(data,-1),-1)
			#print(data.shape)
		else:
			data=data.view(data.shape[0],-1)
		cpc_features.append((data.cpu().numpy()[0],target.item()))
	print(len(cpc_features))
	return cpc_features

def k_means(k):
	data_tuples=get_features_cpc(train_loader)
	#print(data_tuples[:10])
	data=np.array([data_tuples[i][0] for i in range(len(data_tuples))])
	target=[data_tuples[i][1] for i in range(len(data_tuples))]
	#print(data[:5])
	#print(target[:5])
	#X_Norm = preprocessing.normalize(data)
	with open('/home/ssp573/Contrastive-Predictive-Coding/cpc_features_'+args.dataset+".pkl",'wb') as f:
		pkl.dump(data,f)
	with open('/home/ssp573/Contrastive-Predictive-Coding/targets_'+args.dataset+".pkl",'wb') as f:
		pkl.dump(target,f)
	#kmeans = KMeans(n_clusters=10, random_state=0).fit(X_norm)
	#print(kmeans.labels_[:5],target[:5])

k_means(10)
