
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
	def __init__(self):
		super(Encoder, self).__init()
		self.conv1=nn.Conv2d(3,hidden_size,kernel_size=3,stride=2)
		self.conv2=nn.Conv2d(hidden_size,hidden_size,kernel_size=3,stride=2)
		self.conv3=nn.Conv2d(hidden_size,hidden_size,kernel_size=3,stride=2)
		self.conv4=nn.Conv2d(hidden_size,hidden_size,kernel_size=3,stride=2)
		self.conv_bn=nn.BatchNorm2d(64)
		self	.linear_bn=nn.BatchNorm2d(64)
		self.dropout=nn.Dropout2d(0.2)
		self.linear1=nn.linear(128,256)
		self.linear2=nn.linear(256,code_size)

	def forward(self,X):
		X=F.leaky_relu(self.conv_bn(self.conv1(X)))
		X=F.leaky_relu(self.conv_bn(self.conv2(X)))
		X=F.leaky_relu(self.conv_bn(self.conv3(X)))
		X=F.leaky_relu(self.conv_bn(self.conv4(X)))
		X=X.view(-1,64*2*2)
		X=self.leaky_relu(self.linear_bn(self.linear1(X)))
		X=self.linear2(X)

class Autoregressive:
	def __init__(self,num_layers=1):
		super(Autoregressive, self).__init()
		self.num_layers,self.hidden_size=num_layers,gru_hidden_size
		self.rnn=nn.GRU(code_size,gru_hidden_size,num_layers=num_layers,batch_first=True)

	def forward(self,X):
		batch_size, seq_len = x.size()
		if torch.cuda.is_available():
			self.hidden=self.init_hidden(batch_size).cuda()
		else:
			self.hidden=self.init_hidden(batch_size)
		X , self.hidden = self.rnn(X,self.hidden)

		return self.hidden

	def init_hidden(self,batch_size):
		hidden=torch.randn(self.num_layers, batch_size, self.hidden_size)
		return hidden

class Generator:
	def __init__(self): #,predict_terms):
		super(Autoregressive, self).__init()
		self.predictor=nn.Linear(gru_hidden_size,code_size)

	def forward(self,x):
		return self.predictor(x)

'''def network_prediction(context, code_size, predict_terms):
    linear1=nn.linear(code_size,code_size)
    outputs = []
    outputs.append(linear1(context))

    if len(outputs) == 1:
        output = outputs.reshape(-1,1)
    #else:
    #    output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

	return output

class CPC_Layer:
	def __init__(self):
		super(CPC_Layer, self).__init()

	def call(self,inputs):
		preds,y_encoded=inputs
		dot_product=torch.mean(preds*y_encoded,2,True)
		dot_product=torch.mean(dot_product,1)
		dot_product_probs=torch.sigmoid(dot_product)
		return dot_product_probs

	def compute_output_shape(self,input_shape):
		return (input_shape[0][0],1)'''

class Model:
	def __init__(self,encoder,autoregressor,generator):
		super(Model, self).__init()
		self.encoder=encoder
		self.autoregressor=autoregressor
		self.generator=generator
	def forward(self,terms):
		return self.generate(self.autoregress(self.encode(terms)))

	def autoregress(self,x):
		return self.autoregressor.forward(x)

	def encode(self,x):
		return self.encoder.forward(x)

	def generate(self,x):
		return self.generator.forward(x)

#class Loss_Compute_CPC:
#	def __init__(self,x,predict_terms)

def make_model():
	enc=Encoder()
	autoregressor=Autoregressive()
	generator=Generator()
	model=Model(encoder, autoregressor, generator)
	return model



