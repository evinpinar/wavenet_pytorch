import torch
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import wavenet.wave_util as util
from tensorboardX import SummaryWriter
import numpy as np
import time

import logging

log = logging.getLogger(__name__)


def train(dataloader, model, optimizer, global_step, hparams, writer):
	log.info("Train phase...")
	model.train()

	total_time = []

	# net = torch.nn.DataParallel(model)

	for values in dataloader:
		tic = time.time()

		current_lr = util.noam_learning_rate_decay(hparams.initial_learning_rate, global_step)
		for param_group in optimizer.param_groups:
			param_group['lr'] = current_lr
		optimizer.zero_grad()

		x = values[0]
		target = values[1]

		x, target = Variable(x.float()).to(hparams.device), Variable(target.long()).to(hparams.device)

		if hparams.cin_channels != -1:
			mel = values[2]
			mel = Variable(mel)
			mel = mel.to(hparams.device)
			y = model(x, mel)
		else:
			y = model(x)

		# y = y.view(-1, hparams.classes)
		# y = torch.t(y.squeeze())
		# print("y size: ", y.size())
		# print("target size", target.size())
		#y = y.squeeze()
		loss = F.cross_entropy(y.squeeze(), target.squeeze())

		loss.backward()
		optimizer.step()

		tac = time.time()

		if global_step % hparams.log_every_n_samples == 0:
			log.info("Step: {} Loss: {}".format(global_step, loss))
			log.info(" time passed one forward: {}".format(tac - tic))

			# write loss and support
			writer.add_scalar('train/loss', loss, global_step)

			# write learning rate
			lr = np.mean([pg['lr'] for pg in optimizer.param_groups])
			writer.add_scalar('train/learning-rate', lr, global_step)

		# log.info(" Generating audio...")
		# audio = model.generate(hparams.sample_size)
		# audio_name = 'train/wav'+str(global_step)
		# writer.add_audio(audio_name, audio, global_step, hparams.sample_rate)

		global_step += 1
		total_time.append(tac - tic)

	log.info("Total time: {}".format(np.sum(total_time)))
	return global_step


def val(dataloader, model, global_step, hparams, writer):
	log.info("Validation phase...")
	model.eval()

	# net = torch.nn.DataParallel(model)

	all_loss = []
	with torch.no_grad():
		for values in dataloader:
			x = values[0]
			target = values[1]

			x, target = Variable(x.float()).to(hparams.device), Variable(target.long()).to(hparams.device)

			target = target.to(hparams.device)

			if hparams.cin_channels != -1:
				mel = values[2]
				mel = Variable(mel)
				mel = mel.to(hparams.device)
				y = model(x, mel)
			else:
				y = model(x)

			# y = y.view(-1, hparams.classes)

			# loss = F.cross_entropy(y, target)
			loss = F.cross_entropy(y.squeeze(), target.squeeze())
			all_loss.append(loss)

	avg_loss = np.mean(all_loss)

	log.info("Step: {} Loss: {}".format(global_step, avg_loss))

	# write loss and support
	writer.add_scalar('val/loss', avg_loss, global_step)

	# write audio
	# TODO: number could be controlled by generate_every_n_samples?
	# log.info(" Generating audio...")
	# audio = model.generate(hparams.sample_size)
	# audio_name = 'val/wav' + +str(global_step)
	# writer.add_audio(audio_name, audio, global_step)

	return avg_loss


def train_and_evaluate(dataset, hparams, logdir, checkpoint=None):
	log.info("Fetch model...")
	model = util.fetch_model(hparams)
	log.info("Fetch dataloader...")
	dataloader = util.fetch_dataloader(dataset, model, hparams)
	log.info("Fetch optimizer...")
	optimizer = util.fetch_optimizer(model, hparams)

	global_step = 0
	best_metric = 0.0
	best_model = model

	writer = SummaryWriter(logdir)

	# load model or resume from checkpoint if possible
	if checkpoint:
		state = util.load_checkpoint(checkpoint)
		if hparams.resume:
			log.info('Resuming training from checkpoint: {}'.format(checkpoint))
			best_metric = state['best_metric']
			global_step = state['global_step']
			optimizer.load_state_dict(state['optim_dict'])
		log.info('Loading model from checkpoint: {}'.format(checkpoint))
		model.load_state_dict(state['state_dict'])

	log.info("Start training...")
	run_tic = time.time()
	for epoch in range(hparams.num_epochs):

		log.info("Epoch {}/{}".format(epoch + 1, hparams.num_epochs))

		global_step = train(dataloader['train'], model, optimizer, global_step, hparams, writer)

		metric = val(dataloader['val'], model, global_step, hparams, writer)

		is_best = False
		if metric >= best_metric:
			log.info('Found new best! Metric: {}'.format(metric))
			is_best = True
			best_metric = metric
			best_model = model

		# Save weights
		log.info('Saving checkpoint at global step {}'.format(global_step))
		util.save_checkpoint({'global_step': global_step,
							  'best_metric': best_metric,
							  'metric': metric,
							  'state_dict': model.state_dict(),
							  'optim_dict': optimizer.state_dict()},
							 is_best=is_best,
							 checkpoint=logdir)

	run_tac = time.time()

	log.info("Generating a sample sound with the best model...")
	gen_tic = time.time()
	audio = best_model.generate(hparams.sample_size)
	gen_tac = time.time()
	# write audio to the tensorboard
	log.info("{} epochs with batchsize:{}, total time passed:{} ".format(hparams.num_epochs, hparams.batch_size,
																		 run_tac - run_tic))
	log.info("Sample size: {}, generation time: {}".format(hparams.sample_size, gen_tac - gen_tic))
	writer.add_audio('final/wav', audio, global_step, hparams.sample_rate)
