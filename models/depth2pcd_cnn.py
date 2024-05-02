from torch import nn
import torch

import ipdb
st = ipdb.set_trace

'''
Architecture:

Input:
Images
shape:[1, 3, 224, 224]

Output:
Point Cloud
shape: [1, 29184, 3]
'''

# class CNN(nn.Module):
# 	def __init__(self, num_channels = 3):
# 		# call the parent constructor
# 		super(CNN, self).__init__()

# 		# Encoder: CNN
# 		self.encoder = nn.Sequential(
# 			nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # Output size: 112 x 112
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # Output size: 56 x 56
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # Output size: 28 x 28
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # Output size: 14 x 14
#             nn.Flatten()  # Flatten the output
#         )

#         # Decoder: MLP
# 		self.decoder = nn.Sequential(
#             nn.Linear(128 * 14 * 14, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 14592 * 3),  # Output point cloud of shape (14592, 3)
# 			nn.Unflatten(1, (14592, 3))
#         )


# 	def forward(self, x):
# 		x = self.encoder(x)
# 		out = self.decoder(x)
		
# 		return out


'''
Architecture:

Input:
Images
shape:[1, 3, 1028, 1224]

Output:
Point Cloud
shape: [1, 29184, 3]
'''

class CNN(nn.Module):
	def __init__(self, num_channels = 3):
		# call the parent constructor
		super(CNN, self).__init__()

		# Encoder: CNN
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: 514 x 612
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: 257 x 306
            nn.Flatten()  # Flatten the output
        )

        # Decoder: MLP
		self.decoder = nn.Sequential(
            nn.Linear(32 * 257 * 306, 32 * 257 * 306 / 2),
            nn.ReLU(),
            nn.Linear(32 * 257 * 306 / 2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 14592 * 3),  # Output point cloud of shape (14592, 3)
			nn.Unflatten(1, (14592, 3))
        )


	def forward(self, x):
		x = self.encoder(x)
		out = self.decoder(x)
		
		return out