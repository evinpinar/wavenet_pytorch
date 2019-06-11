import torch
import librosa
from  src.Tacotron_model.taco_model import MelSpectrogramNet
from src.dataset import TextToSpeechDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

char_to_id = {char: i for i, char in enumerate(hparams.chars)}
id_to_char = {i: char for i, char in enumerate(hparams.chars)}

def fetch_dataloader(dataset, model, hparams):
	num_train = len(dataset)
	indices = list(range(num_train))
	valid_size = hparams.valid_size
	split = int(np.floor(valid_size * num_train))
	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	print("Fetch dataloader: ")

	dl_train = TextToSpeechDataset(dataset,csv_file= text_PATH,
                                   wav_folder = wav_PATH,
                                  text_embeddings=text_to_sequence,
                                  mel_transforms=wav_to_spectrogram)

	dl_val = TextToSpeechDataset(dataset,csv_file= text_PATH,
                                   wav_folder = wav_PATH,
                                  text_embeddings=text_to_sequence,
                                  mel_transforms=wav_to_spectrogram)

	return {
		'train': dl_train,
		'val': dl_val
	}


def fetch_model(hparams):
	model = MelSpectrogramNet()
	return model


def fetch_optimizer(model, hparams):
	optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

	return optimizer

def text_to_sequence(text, eos=hparams.eos):

    text += eos
    return [char_to_id[char] for char in text]


def sequence_to_text(sequence):
    return "".join(id_to_char[i] for i in sequence)


def ms_to_frames(ms, sample_rate):
    return int((ms / 1000) * sample_rate)


def wav_to_spectrogram(wav, sample_rate=hparams.sample_rate,
                       fft_frame_size=hparams.fft_frame_size,
                       fft_hop_size=hparams.fft_hop_size,
                       num_mels=hparams.num_mels,
                       min_freq=hparams.min_freq,
                       max_freq=hparams.max_freq,
                       floor_freq=hparams.floor_freq):
    """
    Converts a wav file to a transposed db scale mel spectrogram.
    Args:
        wav:
        sample_rate:
        fft_frame_size:
        fft_hop_size:
        num_mels:
        min_freq:
        max_freq:
        floor_freq:

    Returns:

    """
    n_fft = ms_to_frames(fft_frame_size, sample_rate)
    hop_length = ms_to_frames(fft_hop_size, sample_rate)
    mel_spec = librosa.feature.melspectrogram(wav, sr=sample_rate,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              n_mels=num_mels,
                                              fmin=min_freq,
                                              fmax=max_freq)
    return librosa.power_to_db(mel_spec, ref=floor_freq).T

