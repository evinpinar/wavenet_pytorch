import torch
import torch.nn as nn
import torch.nn.functional as F
import wavenet.utils as util
import numpy as np
import sys

import logging

from numpy.core.multiarray import ndarray

log = logging.getLogger(__name__)


class WavenetModel(nn.Module):

	def __init__(self,
				 layers=10,
				 dilation_channels=32,
				 stacks=3,
				 residual_channels=32,
				 skip_channels=256,
				 classes=256,
				 output_length=256,
				 kernel_size=2,
				 cin_channels=-1,
				 dtype=torch.FloatTensor,
				 bias=False):

		super(WavenetModel, self).__init__()

		self.layers = layers
		self.stacks = stacks
		self.dilation_channels = dilation_channels
		self.residual_channels = residual_channels
		self.skip_channels = skip_channels
		self.classes = classes
		self.kernel_size = kernel_size
		self.dtype = dtype

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

		self.causal = nn.Conv1d(in_channels=1,
								out_channels=residual_channels,
								kernel_size=1,
								bias=1)

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
													 out_channels=residual_channels,
													 kernel_size=kernel_size,
													 bias=bias))

					self.lc_gates.append(nn.Conv1d(in_channels=cin_channels,
												   out_channels=residual_channels,
												   kernel_size=kernel_size,
												   bias=bias))

				# residual convolutions
				self.residuals.append(nn.Conv1d(in_channels=dilation_channels,
												out_channels=residual_channels,
												kernel_size=1,
												bias=bias))

				# skip connection convolutions
				self.skips.append(nn.Conv1d(in_channels=dilation_channels,
											out_channels=skip_channels,
											kernel_size=1,
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

		self.output_length = output_length
		self.receptive_field = receptive_field

	def forward(self, y, c=None):
		"""

        Args:
            y (): audio data, B X T X NumInputChannels
            c (): local conditioned data ie. mel spectrogram

        Returns:
            audio data B X T' X Q
                T' = T - Receptive fields
                Q = self.classes or quantization channels, usually 256

        """
		# print("forward step! ")

		# print("input size: ", y.size())

		x = self.causal(y)
		skip = 0

		# print("first causal layer size: ", x.size())

		# TODO: local conditioning upsample layer (currently expanding the input)
		for i in range(self.layers * self.stacks):

			f = self.filters[i](x)
			g = self.gates[i](x)

			# Local conditioning convolutions
			if self.cin_channels != -1:
				f_c = self.lc_filters[i](c)
				g_c = self.lc_gates[i](c)
				f = f + f_c[:, :, :f.size(-1)]  # find a better solution to adjust time?
				g = g + g_c[:, :, :g.size(-1)]

			f = F.tanh(f)
			g = F.sigmoid(g)
			output = f * g

			output = output[:, :, :x.size(-1)]

			# print("input size: ", output.size())

			# skip connections
			s = self.skips[i](output)

			# print("s size: ", s.size())
			# try:
			# skip = skip[:, :, -s.size(2):]
			# skip = skip[:, :, :s.size(-1)]
			# except:
			# skip = 0

			skip = s + skip
			# print("skip size: ", skip.size())

			# add residual and dilated output
			output = self.residuals[i](output)
			# print("after residual size: ", output.size())
			# x = output + x[:, :, -output.size(2):]
			x = x + output

		# print("  size after dilations: ", x.size())

		# last layers, x is the input of softmax
		x = F.relu(skip)
		x = self.end1(x)

		x = F.relu(x)
		x = self.end2(x).transpose(1, 2)

		# print("  output size: ", x.size())

		return x

	def generate(self, sample_size):
		# sample size is the length of generation audio
		receptive_length = self.receptive_field
		# print("input length: ", input_length)

		# if sample_size <= receptive_length:
		# 	log.info("sample size is short! we change it to:{}".format(receptive_length))
		# 	sample_size = receptive_length + 1

		init = np.zeros(receptive_length)
		init = torch.tensor(init).float().view(1, 1, -1)
		if torch.cuda.is_available():
			init = init.cuda()
		log.info("Required generation size: {}".format(sample_size))
		while init.size(-1) != sample_size:
			k = init[:, :, -receptive_length:]
			print("generation starting size: ", k.size())
			if torch.cuda.is_available():
				k = k.cuda()
			gen_out = self.forward(k)
			last = gen_out[:, -1, :]
			decoded = util.one_hot_decode(last)
			init = torch.cat((init.float(), decoded.float().view(1, 1, -1)), 2)

		audio = util.mulaw_decode(init.squeeze(), self.classes)

		return audio.view(1, -1).cpu()

	def generate2(self, sample_size):
		"""
        Start with zeros, produces one by one.
        TODO: add first samples

        Args:
            sample_size (): 
            hparams (): sample size,

        Returns:
            Generated audio with size = sample_size
        """

		output_length = self.output_length
		receptive_length = self.receptive_field
		input_length = receptive_length + output_length
		# print("input length: ", input_length)

		if sample_size <= input_length:
			log.info("sample size is short! we change it to:{}".format(input_length))
			sample_size = input_length + 1

		init = np.zeros(input_length)
		init = torch.tensor(init).float().view(1, 1, -1)
		init = init.cuda()
		log.info("Required generation size: {}".format(sample_size))
		while init.size(-1) != sample_size:
			k = init[:, :, -input_length:]
			print("generation starting size: ", k.size())
			if torch.cuda.is_available():
				k = k.cuda()
			gen_out = self.forward(k)
			last = gen_out[:, :, -1]
			decoded = util.one_hot_decode(last)
			init = torch.cat((init.float(), decoded.float().view(1, 1, -1)), 2)

		audio = util.mulaw_decode(init.squeeze(), self.classes)

		return audio.view(1, -1).cpu()
