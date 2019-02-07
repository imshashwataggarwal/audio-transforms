# coding: utf-8
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle


class Embedding(nn.Module):
	def __init__(self, n=8):
		super(Embedding, self).__init__()
		self.conv1 = nn.Conv2d(1, 96, 7, stride=(2, 2), padding=1)
		self.bn1 = nn.BatchNorm2d(96)
		self.pool1 = nn.MaxPool2d(3, stride=2)

		self.conv2 = nn.Conv2d(96, 256, 5, stride=(2, 2), padding=1)
		self.bn2 = nn.BatchNorm2d(256)
		self.pool2 = nn.MaxPool2d(3, stride=2)

		self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(384)
		self.conv4 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(256)
		self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(256)
		self.pool5 = nn.MaxPool2d((5, 3), stride=(3, 2))

		self.fc6 = nn.Conv2d(256, 4096, (9, 1), stride=1, padding=0)
		self.bn6 = nn.BatchNorm2d(4096)
		self.pool6 = nn.AvgPool2d((1, n), stride=1)

		self.fc7 = nn.Conv2d(4096, 1024, 1, stride=1, padding=0)
		self.fc8 = nn.Conv2d(1024, 1300, 1, stride=1, padding=0)

		self.relu = nn.LeakyReLU(negative_slope=0.0, inplace=True)

	def forward(self, x):
		x = x.unsqueeze(1)  # (B, 1, F, I)

		# Block 1
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.pool1(x)

		# Block 2
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.pool2(x)

		# Block 3
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)
		x = self.conv4(x)
		x = self.bn4(x)
		x = self.relu(x)
		x = self.conv5(x)
		x = self.bn5(x)
		x = self.relu(x)
		x = self.pool5(x)

		# Block 4
		x = self.fc6(x)
		x = self.bn6(x)
		x = self.relu(x)
		x = self.pool6(x)

		# Block 5
		x = self.fc7(x)
		x = self.relu(x)
		x = self.fc8(x)

		x = x.view(x.size(0), -1)
		return x


if __name__ == '__main__':
	x = torch.FloatTensor(1, 512, 500)
	emb = Embedding(14)
	x = emb(x)
	print(x.size())
	# model = pickle.load(open("model.p", "rb"))
	# emb = Embedding()

	# emb.conv1.weight.data = torch.from_numpy(model['conv1']['weights'].astype(np.float32))
	# emb.conv1.bias.data = torch.from_numpy(model['conv1']['bias'].astype(np.float32))

	# emb.conv2.weight.data = torch.from_numpy(model['conv2']['weights'].astype(np.float32))
	# emb.conv2.bias.data = torch.from_numpy(model['conv2']['bias'].astype(np.float32))

	# emb.conv3.weight.data = torch.from_numpy(model['conv3']['weights'].astype(np.float32))
	# emb.conv3.bias.data = torch.from_numpy(model['conv3']['bias'].astype(np.float32))

	# emb.conv4.weight.data = torch.from_numpy(model['conv4']['weights'].astype(np.float32))
	# emb.conv4.bias.data = torch.from_numpy(model['conv4']['bias'].astype(np.float32))

	# emb.conv5.weight.data = torch.from_numpy(model['conv5']['weights'].astype(np.float32))
	# emb.conv5.bias.data = torch.from_numpy(model['conv5']['bias'].astype(np.float32))

	# emb.fc6.weight.data = torch.from_numpy(model['fc6']['weights'].astype(np.float32))
	# emb.fc6.bias.data = torch.from_numpy(model['fc6']['bias'].astype(np.float32))

	# emb.fc7.weight.data = torch.from_numpy(model['fc7']['weights'].astype(np.float32))
	# emb.fc7.bias.data = torch.from_numpy(model['fc7']['bias'].astype(np.float32))

	# emb.fc8.weight.data = torch.from_numpy(model['fc8']['weights'].astype(np.float32))
	# emb.fc8.bias.data = torch.from_numpy(model['fc8']['bias'].astype(np.float32))

	# torch.save(emb.state_dict(), './embeddings.pt')
