import librosa.display
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
import os
import numpy as np
import torch.nn.functional as F

# from src.Tacotron_model import taco_util
import wavenet.audio_util as util


class TextToSpeechDataset(Dataset):

	def __init__(self, csv_file, wav_folder, hparams, text_embeddings=None, mel_transforms=None):
		"""
        Args:
            hparams (object):
            csv_file (string): Path to the csv file with texts.
            wav_file (string): Path to audio files in wav format.
        """
		self.text_frames = pd.read_csv(csv_file, delimiter='|', header=None)
		self.text_frames.dropna(inplace=True)
		self.wav_file = wav_folder
		self.mel_transforms = mel_transforms
		self.text_embeddings = text_embeddings
		self.hparams = hparams

	def __len__(self):
		return self.text_frames.shape[0]

	def __getitem__(self, idx):

		embedded_text = None
		mel_spectograms = None
		# mel_resized = None

		audio_name = os.path.join(self.wav_file,
								  self.text_frames.iloc[idx, 0]) + '.wav'
		audio, _ = librosa.load(audio_name)
		audio, _ = librosa.effects.trim(audio)  # trim the silences in beginning and endings
		audio_array = np.asarray(audio)  # convert tuple to np array
		audio_array = util.mulaw_encode(audio_array, self.hparams.classes) # Mulaw quantization
		audio_tensor = torch.tensor(audio_array)
		text = self.text_frames.iloc[idx, 2:]
		text = text.values  # convert pandas dataframe to np array
		text_for_embeddings = str(text)
		text_for_embeddings = text_for_embeddings.strip('[]')
		if self.text_embeddings:
			embedded_text = self.text_embeddings(text_for_embeddings)

		# audio_for_mel = np.array(audio)
		if self.mel_transforms:
			mel_spectogram = self.mel_transforms(audio, self.hparams)
			# print("original mel size: ", mel_spectogram.shape)
			mel_resized = util.mel_resize(len(audio), mel_spectogram)

		sample = {'text': list(text),
				  'speech': audio_tensor,
				  'embedded_text': embedded_text,
				  'mel_spectograms': mel_spectogram,
				  'mel_resized': mel_resized
				  }

		return sample


class WavenetLoader(DataLoader):

	def __init__(self, dataset, receptive_length, q_channels=256, batch_size=1,
				 sampler=None, cin_channels=-1, max_time_steps=8000, num_workers=1):
		"""
		Gets the data as TTSDataset format and loads the audio outputs for training
			sample_size = receptive_length+target_size - 1

		Args:
			dataset ():
			receptive_length ():
			sample_size (): Minimum size of the sample for feeding the network.
			q_channels ():
			batch_size ():
			sampler ():
			num_workers ():
		"""

		self.cin_channels = cin_channels
		self.dataset = dataset

		super().__init__(dataset, batch_size, shuffle=False, sampler=sampler)  # num_workers

		self.receptive_fields = receptive_length
		self.q_channels = q_channels
		self.max_time_steps = max_time_steps
		if cin_channels != -1:
			self.collate_fn = self._collate_fn3
		else:
			self.collate_fn = self._collate_fn2

	def _collate_fn1(self, batch):
		"""
		Gets b batches and stacks them with the same size of samples.

		TODO: return the merged data from consecutive datafiles -> LATER
		Args:
			batch ():

		Returns:
			the audio and targets with required lengths during iteration.
		"""
		print("collate call: ", len(batch))

		input_batch, target_batch = [], []
		for sample in batch:
			ins, target = self._prepare1(sample)
			input_batch.append(ins)
			target_batch.append(target)
		return input_batch, target_batch

	def _collate_fn2(self, batch):
		"""
		Gets batches, pads the short audio data, embeds and returns.
		Args:
			batch ():

		Returns:

		"""
		# print("collate call!: ", len(batch))
		max_time_steps = self.max_time_steps
		# input_lengths = [len(x['speech']) for x in batch]
		# print(input_lengths)
		# max_input_length = max(input_lengths)
		input_batch = torch.zeros(0, 0).float()
		target_batch = torch.zeros(0, 0).int()

		for x in batch:
			audio = self._prepare3(x['speech'], max_time_steps)
			# encoded = self._prepare2(audio, max_time_steps)
			# targets = encoded[:-self.receptive_fields] # IS THIS TRUE?
			# targets = audio[self.receptive_fields:]
			#encoded = util.mulaw_encode(np.asarray(audio), self.q_channels)
			target = audio[1:].view(1, -1)
			input = util.one_hot_encode(audio[:-1], channels=self.q_channels).transpose(0, 1)
			input.unsqueeze_(0)
			# print("target size: ", target.size())
			input_batch = torch.cat((input_batch, input))
			target_batch = torch.cat((target_batch, target))
		#print(" batch input: ", input_batch.size())
		#print(" batch output: ", target_batch.size())

		# print("return batches")

		return input_batch, target_batch

	def _prepare2(self, audio, max_len):
		# print(" length of audio: ", len(audio))
		# print(" max len: ", max_len)
		encoded = util.mulaw_encode(audio, self.q_channels)
		if len(encoded) < max_len:
			encoded = F.pad(encoded, (0, max_len - len(encoded)), mode='constant')
		return encoded

	def _prepare3(self, audio, max_time_steps):
		if len(audio) > max_time_steps:
			s = np.random.randint(0, len(audio) - max_time_steps)
			audio = audio[s:s + max_time_steps]
		else:
			audio = F.pad(audio, (0, max_time_steps - len(audio)), mode='constant')

		return audio

	def _prepare4(self, audio, mel, max_time_steps):
		if len(audio) > max_time_steps:
			s = np.random.randint(0, len(audio) - max_time_steps)
			audio = audio[s:s + max_time_steps]
			mel = mel[s:s + max_time_steps, :]
		else:
			audio = F.pad(audio, (0, max_time_steps - len(audio)), mode='constant')
		# pad mel

		return audio, mel


	def _collate_fn3(self, batch):
		"""
		Gets batches, pads the short audio data, embeds, prepares the
		regarding mel-spectrogram along with it for using in training.
		Args:
			batch ():

		Returns:

		"""
		# print("collate call!: ", len(batch))
		batch_size = len(batch)
		max_time_steps = self.max_time_steps
		# input_lengths = [len(x['speech']) for x in batch]
		# print(input_lengths)
		# max_input_length = max(input_lengths)

		input_batch = torch.zeros(0, 0).float()
		target_batch = torch.zeros(0, 0).int()
		mel_batch = torch.zeros(0, 0, 0).float()

		i = 0
		for x in batch:
			# print(i)
			i += 1

			audio, mel = self._prepare4(x['speech'], x['mel_resized'], max_time_steps)
			target = audio[1:].view(1, -1)
			input = util.one_hot_encode(audio[:-1], channels=self.q_channels).transpose(0, 1)
			input.unsqueeze_(0)

			#print(mel.shape)
			#input = audio.view(1, 1, -1)
			#target = util.mulaw_encode(np.asarray(input), self.q_channels).view(1, -1)

			melT = np.transpose(mel[:-1])
			# print("transposed mel: ", mel.shape)
			melTor = torch.from_numpy(melT).float().unsqueeze(0)
			# print("mel tensor: ", melTor.size())

			input_batch = torch.cat((input_batch, input))
			target_batch = torch.cat((target_batch, target))
			mel_batch = torch.cat((mel_batch, melTor))

		#print(" batch input: ", input_batch.size())
		#print(" batch output: ", target_batch.size())

		# print("return batches")

		return input_batch, target_batch, mel_batch


def pad_text(text, max_len):
	return np.pad(text, (0, max_len - len(text)), mode='constant', constant_values=0)


def pad_spectrogram(S, max_len):
	padded = np.zeros((max_len, 80))
	padded[:len(S), :] = S
	return padded


def collate_fn(batch):
	"""
    Pads Variable length sequence to size of longest sequence.
    Args:
        batch:
    Returns: Padded sequences and original sizes.
    """
	text = [item[0] for item in batch]
	audio = [item[1] for item in batch]

	text_lengths = [len(x) for x in text]
	audio_lengths = [len(x) for x in audio]

	max_text = max(text_lengths)
	max_audio = max(audio_lengths)

	text_batch = np.stack(pad_text(x, max_text) for x in text)
	audio_batch = np.stack(pad_spectrogram(x, max_audio) for x in audio)

	return (torch.LongTensor(text_batch),
			torch.FloatTensor(audio_batch).permute(1, 0, 2),
			text_lengths, audio_lengths)
