import wavenet.wave_model
from dataset import WavenetLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import os
import shutil

import logging

log = logging.getLogger(__name__)


def fetch_dataloader(dataset, model, hparams):
	num_train = len(dataset)
	indices = list(range(num_train))
	valid_size = hparams.valid_size
	split = int(np.floor(valid_size * num_train))
	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	sample_size = model.receptive_field + hparams.output_length
	print("sample size: ", sample_size)

	dl_train = WavenetLoader(dataset, receptive_length=model.receptive_field,
							 sample_size=sample_size, batch_size=hparams.batch_size,
							 sampler=train_sampler, num_workers=hparams.num_workers, cin_channels=hparams.cin_channels,
							 max_time_steps=hparams.max_time_steps)

	dl_val = WavenetLoader(dataset, receptive_length=model.receptive_field,
						   sample_size=sample_size, batch_size=hparams.batch_size,
						   sampler=valid_sampler, num_workers=hparams.num_workers, cin_channels=hparams.cin_channels,
						   max_time_steps=hparams.max_time_steps)

	return {
		'train': dl_train,
		'val': dl_val
	}


def fetch_model(hparams):
	model = wavenet.wave_model.WavenetModel(layers=hparams.layers, stacks=hparams.stacks,
											dilation_channels=hparams.dilation_channels,
											residual_channels=hparams.residual_channels,
											skip_channels=hparams.skip_channels, classes=hparams.classes,
											output_length=hparams.output_length,
											kernel_size=hparams.kernel_size, cin_channels=hparams.cin_channels,
											bias=hparams.bias)
	#model = model.to(hparams.device)
	return model


def fetch_optimizer(model, hparams):
	optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams.lr, betas=(
		hparams.adam_beta1, hparams.adam_beta2), eps=hparams.adam_eps, weight_decay=hparams.weight_decay,
								 amsgrad=hparams.amsgrad)

	return optimizer


def save_checkpoint(state, is_best, checkpoint):
	"""Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
	filepath = os.path.join(checkpoint, 'last.pth.tar')
	if not os.path.exists(checkpoint):
		log.info("Log directory does not exist! Making directory {}".format(checkpoint))
		os.mkdir(checkpoint)
	torch.save(state, filepath)
	if is_best:
		shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint):
	"""Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
    Return:
        (dict): return saved state
    """
	if not os.path.exists(checkpoint):
		raise ("File doesn't exist {}".format(checkpoint))
	state = torch.load(checkpoint)

	return state
