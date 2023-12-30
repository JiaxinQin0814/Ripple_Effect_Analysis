
import sys
sys.path.append("/home/zixuan11/qjx/OpenKE")
import json
import os
import torch
torch.cuda.set_device(5)
torch.cuda.current_device()
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from typing import List, Optional
import torch
import seaborn as sns
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import copy

import argparse

# import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


# training 

def train(args):
	# dataloader for training
	train_dataloader = TrainDataLoader(
		in_path = "/home/zixuan11/qjx/OpenKE/benchmarks/FB15K237/", 
		nbatches = 100,
		threads = 8, 
		sampling_mode = "normal", 
		bern_flag = 1, 
		filter_flag = 1, 
		neg_ent = 25,
		neg_rel = 0)
	# dataloader for test
	test_dataloader = TestDataLoader("/home/zixuan11/qjx/OpenKE/benchmarks/FB15K237/", "link")
 
		# define the model
	transe = TransE(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim = (int)(args.dim), 
		p_norm = 1, 
		norm_flag = True)

	# define the loss function
	model = NegativeSampling(
		model = transe, 
		loss = MarginLoss(margin = 5.0),
		batch_size = train_dataloader.get_batch_size()
	)

	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = (int)(args.training_times), alpha = 1.0, use_gpu = True)
	trainer.run()
	transe.save_checkpoint(f'/home/zixuan11/qjx/gradient_experiment/transE_checkpoint/transe_dim{args.dim}.ckpt')
 
 
def editing(args):
	transe = TransE(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim = (int)(args.dim), 
		p_norm = 1, 
		norm_flag = True)
	transe.load_checkpoint(f'/home/zixuan11/qjx/gradient_experiment/transE_checkpoint/transe_dim{args.dim}.ckpt')

	
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', default=200)
    parser.add_argument('--training_times', default=1000)
    args = parser.parse_args()
    train(args)
    
	