import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from model import *


def train(epochs,batch_size,output_dir,code_size,lr=0.001,terms=1,predict_terms=1,image_size=28,color=False):
	train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                       positive_samples=batch_size // 2, predict_terms=predict_terms,
                                       image_size=image_size, color=color, rescale=True)
	validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=batch_size // 2, predict_terms=predict_terms,
                                            image_size=image_size, color=color, rescale=True)

	model=make_model()
	model_opt=torch.optim.Adam(model.parameters(),lr=lr)



