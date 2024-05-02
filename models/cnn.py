from torch import nn
import torch

'''
Two Architectures:

Using all the three fisheye images:




Baseline architecture- Using only one fisheye image:




'''

class CNN(nn.Module):
	def __init__(self, num_channels = 10):
		# call the parent constructor
		super(CNN, self).__init__()

		# layers 1
		self.conv1 = nn.Conv2d( in_channels = num_channels,
								out_channels = 20,
								kernel_size = (3, 3),
								stride = 1, padding = 1)
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)) # [1, 20, 514, 612]

		# layers 2
		self.conv2 = nn.Conv2d( in_channels = 20,
								out_channels = 40,
								kernel_size = (3, 3),
								stride = 1, padding = 1)
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)) # [1, 40, 257, 306]

		# layers 3
		self.conv3 = nn.Conv2d( in_channels = 40,
								out_channels = 60,
								kernel_size = (3, 3),
								stride = 1, padding = 1)
		self.relu3 = nn.ReLU()
		self.maxpool3 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)) # [1, 60, 128, 153]

		# encoder
		self.encoder = nn.Sequential(   self.conv1, self.relu1, self.maxpool1,
		                                self.conv2, self.relu2, self.maxpool2,
		                                self.conv3, self.relu3, self.maxpool3)

		# layer 1
		self.conv4 = nn.ConvTranspose2d(in_channels = 60,
										out_channels = 40,
										kernel_size = (4, 3),
										stride = 2, padding = 1, output_padding = 1)
		self.relu4 = nn.ReLU() # [1, 40, 257, 306]

		# layer 2
		self.conv5 = nn.ConvTranspose2d(in_channels = 40,
										out_channels = 20,
										kernel_size = (3, 3),
										stride = 2, padding = 1, output_padding = 1)
		self.relu5 = nn.ReLU() # [1, 20, 514, 612]

		# layer 3
		self.conv6 = nn.ConvTranspose2d(in_channels = 20,
										out_channels = num_channels,
										kernel_size = (3, 3),
										stride = 2, padding = 1, output_padding = 1)
		self.relu6 = nn.ReLU() # [1, 9, 1028, 1224]

		# decoder
		self.decoder = nn.Sequential(self.conv4, self.relu4, self.conv5, self.relu5, self.conv6, self.relu6)

	def forward(self, x):
		x = self.encoder(x) # [1, 60, 128, 153]
		x = self.decoder(x) # [1, 9, 1028, 1224]
		out = torch.mean(x, 1).reshape(x.shape[0], -1, x.shape[2], x.shape[3])
		return out