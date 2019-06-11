import argparse
import os
import torch
from wavenet.wave_train import *
from dataset import TextToSpeechDataset
import hparam as util
from wavenet.utils import wav_to_spectrogram
from datetime import datetime

import logging
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger()


def parse_args():
	"""Parse command line arguments.
	Returns:
		(Namespace): arguments
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--logdir', type=str,
						default='../log/wavenet/',
						help='parent directory of experiment logs (checkpoints/tensorboard events)')
	parser.add_argument('--log_level', type=str, default='INFO',
						help='log level to be used')
	parser.add_argument('-n', '--name', type=str,
						help='name of experiment')
	parser.add_argument('-c', '--checkpoint', type=str, default=None,
						required=False, help='path to checkpoint')
	parser.add_argument('--default_hparams', type=str,
						default='wavenet/wave_hparams.yaml', help='path to .yaml with default hparams')
	parser.add_argument('--hparams', type=str,
						required=False, help='comma separated name=value pairs')
	parser.add_argument('--csv_file', type=str, default='../data/metadata.csv',
						help='csv file of texts and audio names in LLJDS Format')
	parser.add_argument('--wav_dir', type=str, default='../data/wavs',
						help='Directory of audio files listed in csv file.')
	return parser.parse_args()


def run():
	args = parse_args()

	# set log level
	assert (args.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
	# log_level = logging.getLevelName(args.log_level)
	log.setLevel(logging.getLevelName(args.log_level))

	args.name = "test_run"
	log_event_name = args.name + str(datetime.now()).replace(" ", "_")
	print("TensorBoard event log time: {}".format(log_event_name))

	# make sure log directory exists
	logdir = os.path.join(args.logdir, log_event_name)
	# make sure log directory exists
	logdir = os.path.join(args.logdir, args.name)
	if not os.path.isdir(logdir):
		log.info("Creating directory {}".format(logdir))
		os.makedirs(logdir)
		os.chmod(logdir, 0o775)

	# load default hparams
	hparams = util.load_params_from_yaml(args.default_hparams)
	# overwrite with new values if given
	hparams.parse(args.hparams)
	# write params to logdir
	util.write_params_to_yaml(hparams, os.path.join(logdir, 'hparams.yaml'))

	# set seed for reproducible experiments
	torch.manual_seed(hparams.seed)
	if torch.cuda.is_available():
		log.info("CUDA is available. Using GPU.")
		torch.cuda.manual_seed(hparams.seed)
		torch.backends.cudnn.benchmark = True
		hparams.device = torch.device("cuda:0")
		hparams.cuda = True
	else:
		log.info("CUDA is not available. Using CPU.")
		hparams.device = torch.device("cpu")
		hparams.cuda = False

	# start training and evaluation
	TTSDataset = TextToSpeechDataset(args.csv_file, args.wav_dir, hparams, mel_transforms=wav_to_spectrogram)
	train_and_evaluate(TTSDataset, hparams, logdir, args.checkpoint)


if __name__ == "__main__":
	run()
