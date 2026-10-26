"""
A densely connected convolutional network for brain age prediction in human/marmoset 
based on structual MRI 

Writted by Yang Xiao, Lei Zhao, Shuer Ye in Jun 17, 2025

@: xiaoyang9604@gmail.com

"""

import torch
from torch import nn
import torch.nn.functional as F

class ScaleLayer(nn.Module):
	"""
	https://github.com/flyyufelix/DenseNet-Keras/blob/master/densenet161.py
	output = input * gamma + beta
	"""

	def __init__(self, num_feas, axis = 1, beta_init = 'zero', gamma_init = 'one'):
		super(ScaleLayer, self).__init__()
		self.axis = axis # default to channels axis 
	
		if beta_init == 'zero':
			self.beta = nn.Parameter(torch.zeros(num_feas))
		elif beta_init == 'one':
			self.beta = nn.Parameter(torch.ones(num_feas))
		else:
			raise ValueError(f"Unsupported beta_init: {beta_init}")
 	
		if gamma_init == 'zero':
			self.gamma = nn.Parameter(torch.zeros(num_feas))
		elif gamma_init == 'one':
			self.gamma = nn.Parameter(torch.ones(num_feas))
		else:
			raise ValueError(f"Unsupported beta_init: {gamma_init}")

	def forward(self, x):
		shape = [1] * x.dim()
		shape[self.axis] = -1
		beta = self.beta.view(*shape)
		gamma = self.gamma.view(*shape)
		return x * gamma + beta


class ConvBlock(nn.Module):
	"""
	Apply BatchNorm, Relu, bottleneck 1*1*1 Conv3D, 3*3*3 Conv3D, and optional dropout
	"""
	
	def __init__(self, in_chns, growth_rate, dropout_rate):
		super(ConvBlock, self).__init__()
		inter_chns = growth_rate * 4 # huang et al., 2017 & densenet-121/169/161/201
	
		# bottleneck 1*1*1 convolution
		self.bn1 = nn.BatchNorm3d(in_chns, eps = 1.1e-5) # eps = 1.1e-5 ref to Nature Aging, 2021 
		self.scale1 = ScaleLayer(in_chns)
		self.relu1 = nn.ReLU(inplace = True)
		self.conv1 = nn.Conv3d(in_chns, inter_chns, kernel_size = 1, bias = False)

		# dense 3*3*3 convolution
		self.bn2 = nn.BatchNorm3d(inter_chns, eps = 1.1e-5)
		self.scale2 = ScaleLayer(inter_chns)
		self.relu2 = nn.ReLU(inplace = True)
		self.conv2 = nn.Conv3d(inter_chns, growth_rate, kernel_size = 3, padding = 1, bias = False)

		self.dropout = nn.Dropout3d(p = dropout_rate) if dropout_rate else nn.Identity()
    
	def forward(self, x):
		out = self.conv1(self.relu1(self.scale1(self.bn1(x))))
		out = self.conv2(self.relu2(self.scale2(self.bn2(out))))
		out = self.dropout(out)
		return out


class TransitionBlock(nn.Module):
	def __init__(self, in_chns, out_chns, dropout_rate):
		super(TransitionBlock, self).__init__()
		self.bn = nn.BatchNorm3d(in_chns, eps = 1.1e-5)
		self.scale = ScaleLayer(in_chns)
		self.relu = nn.ReLU(inplace = True)
		self.conv = nn.Conv3d(in_chns, out_chns, kernel_size = 1, stride = 1, padding = 0, bias = False)
		self.pool = nn.AvgPool3d(kernel_size = 2, stride = 2)
		self.dropout = nn.Dropout3d(p = dropout_rate) if dropout_rate else nn.Identity()

	def forward(self, x):
		x = self.conv(self.relu(self.scale(self.bn(x))))
		x = self.dropout(x)
		x = self.pool(x)
		return x


class DenseBlock(nn.Module):
	def __init__(self, num_layers, in_chns, growth_rate, dropout_rate):
		super(DenseBlock, self).__init__()
		self.layers = nn.ModuleList()
		self.current_chns = in_chns

		for i in range(num_layers):
			block = ConvBlock(self.current_chns, growth_rate, dropout_rate)
			self.layers.append(block)
			self.current_chns += growth_rate

	def forward(self, x):
		for layer in self.layers:
			new_feas = layer(x)
			x = torch.cat([x, new_feas], dim = 1)
		return x


class DenseNet3DMRI(nn.Module):
	def __init__(self, input_chns = (1, 121, 145, 121), 
			  init_feas = 96, # huang et al., 2017 densenet-121(64)/169(64)/161(*96*)/201(64)
			  growth_rate = 48, # huang et al., 2017 & densenet-121(32)/169(32)/161(*48*)/201(32)
			  block_config = [3, 6, 12, 8], #[6, 12, 36, 24] in densenet-161; but [3, 6, 12, 8] be found in Nature Aging, 2021 
			  dropout_rate = 0.2):
		
		super(DenseNet3DMRI, self).__init__()
		self.growth_rate = growth_rate
		self.block_config = block_config

		self.feas = nn.Sequential()
		# initial convolution
		self.feas.add_module('conv1_pad', nn.ConstantPad3d(3, 0))
		self.feas.add_module('conv1', nn.Conv3d(1, init_feas, kernel_size = 5, stride = 2, bias = False))
		self.feas.add_module('conv1_bn', nn.BatchNorm3d(init_feas, eps = 1.1e-5))
		self.feas.add_module('conv1_scale', ScaleLayer(init_feas))
		self.feas.add_module('conv1_relu', nn.ReLU(inplace = True))
		self.feas.add_module('pool1', nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1))
	    
		# add dense blocks
		num_feas = init_feas
		for i, num_layers in enumerate(self.block_config):
			block = DenseBlock(num_layers, num_feas, self.growth_rate, dropout_rate)
			self.feas.add_module(f'denseblock{i+1}', block)
			num_feas += num_layers * self.growth_rate
	    
			if i != len(self.block_config) - 1:
				trans = TransitionBlock(num_feas, int(num_feas * 1.0), dropout_rate)
				self.feas.add_module(f'transition{i+1}', trans)

		# final bn, scale, relu
		self.final_bn = nn.BatchNorm3d(num_feas, eps = 1.1e-5)
		self.final_scale = ScaleLayer(num_feas)
		self.final_relu = nn.ReLU(inplace = True)
		
		# add lastconv and global average pool 
		self.last_conv = nn.Conv3d(num_feas, num_feas, kernel_size = 3, padding = 1, bias = False)
		self.global_ave_pool = nn.AdaptiveAvgPool3d(1)
		self.fc = nn.Linear(num_feas, 1)	

	def forward(self, x):
		x = self.feas(x)
		x = self.final_relu(self.final_scale(self.final_bn(x)))
		x = self.last_conv(x)
		x = self.global_ave_pool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x



