import torch
import torch.nn as nn
import torch.nn.functional as F
import wavenet.audio_util as util
import math
import numpy as np
import sys

import logging

from numpy.core.multiarray import ndarray

log = logging.getLogger(__name__)


class WavenetModel(nn.Module):

	def __init__(self,
				 layers=10,
				 stacks=3,
				 dilation_channels=32,
				 residual_channels=32,
				 skip_channels=256,
				 classes=256,
				 kernel_size=2,
				 cin_channels=-1,
				 bias=True):

		super(WavenetModel, self).__init__()

		self.layers = layers
		self.stacks = stacks
		self.dilation_channels = dilation_channels
		self.residual_channels = residual_channels
		self.skip_channels = skip_channels
		self.classes = classes
		self.kernel_size = kernel_size

		self.filters = nn.ModuleList()
		self.gates = nn.ModuleList()
		self.residuals = nn.ModuleList()
		self.skips = nn.ModuleList()

		# for local conditioning
		self.cin_channels = cin_channels
		if cin_channels != -1:
			self.lc_filters = nn.ModuleList()
			self.lc_gates = nn.ModuleList()

		receptive_field = 0

		self.causal = nn.Conv1d(in_channels=classes,
								out_channels=residual_channels,
								kernel_size=1,
								dilation=1,
								padding=0,
								bias=bias)

		# TODO: add upsampling layers
		# self.upsamples = nn.ModuleList()
		# for scales:
		# nn.ConvTranspose2d(in_channels = 1, out_channels = 1, kernel_size=(freq x scale), stride=(1 x scale), bias=1, padding=1)

		for s in range(stacks):
			dilation = 1
			additional_scope = kernel_size - 1
			for i in range(layers):

				padding = (kernel_size - 1) * dilation

				# filter convolution
				self.filters.append(nn.Conv1d(in_channels=residual_channels,
											  out_channels=dilation_channels,
											  kernel_size=kernel_size,
											  padding=padding,
											  dilation=dilation,
											  bias=bias))

				# gate convolution
				self.gates.append(nn.Conv1d(in_channels=residual_channels,
											out_channels=dilation_channels,
											kernel_size=kernel_size,
											dilation=dilation,
											padding=padding,
											bias=bias))

				# Local conditioning convolutions for mel spec
				if cin_channels != -1:
					self.lc_filters.append(nn.Conv1d(in_channels=cin_channels,
													 out_channels=dilation_channels,
													 kernel_size=1,
													 bias=bias))

					self.lc_gates.append(nn.Conv1d(in_channels=cin_channels,
												   out_channels=dilation_channels,
												   kernel_size=1,
												   bias=bias))

				# conv output splitted into two
				dilation_out_channels = dilation_channels // 2

				# residual convolutions
				self.residuals.append(nn.Conv1d(in_channels=dilation_out_channels,
												out_channels=residual_channels,
												kernel_size=1,
												bias=bias))

				# skip connection convolutions
				self.skips.append(nn.Conv1d(in_channels=dilation_out_channels,
											out_channels=skip_channels,
											kernel_size=1,
											dilation=1,
											bias=bias))

				receptive_field += additional_scope
				dilation *= 2
				additional_scope *= 2

		self.end1 = nn.Conv1d(in_channels=skip_channels,
							  out_channels=skip_channels,
							  kernel_size=1,
							  bias=True)

		self.end2 = nn.Conv1d(in_channels=skip_channels,
							  out_channels=classes,
							  kernel_size=1,
							  bias=True)

		self.receptive_field = receptive_field

	def forward(self, y, c=None):
		"""

        Args:
            y (torch.tensor): Batch X 1 x Time, audio data in scalar format, values between [-1,1]
            c (torch.tensor): Batch x cin_channels x T, local conditioned data eg. mel spectrogram

        Returns:
            	B X T' X Q ,
                T' = T - Receptive fields
                Q = self.classes or quantization channels(=256)

        """
		# print("forward step! ")

		# print("input size: ", y.size())

		x = self.causal(y)
		skip = 0

		# print("first causal layer size: ", x.size())

		# TODO: local conditioning upsample layer (currently expanding the input)
		for i in range(self.layers * self.stacks):

			# f = self.filters[i](x)
			# g = self.gates[i](x)

			# f = f[:, :, :x.size(-1)]
			# g = g[:, :, :x.size(-1)]

			residual = x
			x = self.filters[i](x)
			x = x[:, :, :residual.size(-1)]
			a, b = x.split(x.size(1)//2, dim=1)

			# print("size after filter:", x.size())

			# Local conditioning convolutions
			if self.cin_channels != -1:
				# f_c = self.lc_filters[i](c)
				# g_c = self.lc_gates[i](c)
				# print(" input size: ", f.size())
				# print(" mel convolution: ", f_c.size(-1))
				# f[:, :, :f_c.size(-1)] = f[:, :, :f_c.size(-1)] + f_c
				# g[:, :, :g_c.size(-1)] = g[:, :, :g_c.size(-1)] + g_c
				# print("local size: ", c.size())
				f_c = self.lc_filters[i](c)
				# print("after conv local:", f_c.size())
				ca, cb = f_c.split(f_c.size(1)//2, dim=1)
				a, b = a+ca, b+cb

			#f = F.tanh(f)
			#g = F.sigmoid(g)
			#output = f * g
			x = F.tanh(a) * F.sigmoid(b)

			# print("output after f*g size: ", x.size())


			# skip connections
			s = self.skips[i](x)
			skip = s + skip
			# print("skip size: ", skip.size())

			# add residual and dilated output
			x = self.residuals[i](x)
			# print("after residual conv size: ", x.size())
			# print("residual size: ", residual.size())
			x = (x + residual) * math.sqrt(0.5)

		# print("  size after dilations: ", x.size())

		x = F.relu(skip)
		x = self.end1(x)

		x = F.relu(x)
		x = self.end2(x).transpose(1, 2)

		# print("  output size: ", x.size())

		# x = F.softmax(x, dim=1)

		return x

	def generate(self, sample_size, hparams, S=None):
		""" S (Tc, D): local conditioning input eg. mel spec """
		# sample size is the length of generation audio
		receptive_length = self.receptive_field
		# print("input length: ", input_length)

		init = torch.zeros(1, 256, 1)
		init = init.to(hparams.device)
		audio_out = np.zeros(1)
		if S is not None:
			upsample_factor = hparams.hop_length # TODO: get hop size here
			# sample size should change accordingly
			sample_size = S.shape[0] * upsample_factor
			# c input should be shape (B x C x T)
			c = np.repeat(S, upsample_factor, axis=0)
			c = torch.FloatTensor(c.T).unsqueeze(0)
		log.info("Required generation size: {}".format(sample_size))
		while init.size(-1) != sample_size:
			k = init[:, :, -receptive_length:]
			k = k.to(hparams.device)
			if S is not None:
				gen_out = self.forward(k, c)[0]
			else:
				gen_out = self.forward(k)[0]
			last = gen_out[-1]
			p = F.softmax(last, dim=0)
			sample = np.random.choice(np.arange(256), p=p.cpu().detach().numpy())
			decoded = sample
			audio_out = np.append(audio_out, decoded)
			# two options here:
			#   1. feed with one-hot vector
			#   2. feed with the softmax output
			one_hot = torch.zeros(1, 256, 1)
			one_hot[:, decoded, :] = 1
			init = torch.cat((init.float(), one_hot.float().cuda()), 2)

		# audio = util.mulaw_decode(init.squeeze(), self.classes)
		# audio = init.view(1, -1).squeeze().cpu()
		final_audio = util.mulaw_decode(audio_out).numpy()

		return final_audio