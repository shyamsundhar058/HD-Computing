'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import time
import sys
import numpy as np
from sklearn import cluster
import random
import os
import pickle
from copy import deepcopy

from HD import train as train_HD

N_P = 16
N_G = 16
EPOCH = 100
cut_layer = 13
#free_percent = [0.57]*7+[0.22]*5+[0.82]
#free_percent_st = '[0.57]*7+[0.22]*5+[0.82]'.replace('*', 'x').replace(' ', '')
free_percent = [1.0]*13
free_percent_st = '[1.0]*13'.replace('*', 'x').replace(' ', '')
rewrite = False #if True, trains from scratch and removes existing model files with the same name
D = 4096 #HD dimension
BW = 1 #HD bitwidth
base_path = './models/vgg16_cifar100_base.pth'
pattern_path = './models/vgg16_cifar100_pattern{}.pth'.format(free_percent_st)
hd_path = './models/vgg16_cifar100_pattern{}_cut{}_hd{},{}.pickle'.format(free_percent_st, cut_layer, D, BW)
pattern_hd_path = './models/vgg16_cifar100_pattern{}_cut{}_hd{},{}.pth'.format(free_percent_st, cut_layer, D, BW)

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

use_cuda = torch.cuda.is_available()

print('==> Preparing data..')
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	#transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR10
	transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)), #CIFAR100
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	#transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR10
	transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)), #CIFAR100
])

train_data = torchvision.datasets.CIFAR100(root='~/research/HDnn/data/', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR100(root='~/research/HDnn/data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)

class ConvLayer(nn.Module):
	""" Custom Linear layer but mimics a standard linear layer """
	def __init__(self, in_channels, out_channels, kernel_size, padding = 0, num_groups = 16, num_patterns = 1):
		super().__init__()
		self.in_size, self.out_size, self.kernel_size = in_channels, out_channels, kernel_size
		self.conv = nn.Conv2d(self.in_size, self.out_size, kernel_size=self.kernel_size, padding = padding)
		self.free_filter_num = 0
		self.num_groups = num_groups #number of groups (unique weights) in a pattern
		self.num_patterns = num_patterns #number of different patterns in a layer
		self.patterns = []
		self.filter_pattern = [] #given a filter index, returns its pattern index

	def create_patterns(self, free_percent):
		self.free_filter_num = int(self.out_size * free_percent)
		self.num_patterns = min(self.num_patterns, (self.out_size - self.free_filter_num)) #number of different patterns in a layer

		if self.free_filter_num != self.out_size:
			np.random.seed(0)
			free_index = random.sample(range(0, self.out_size), self.free_filter_num )
			free_index.sort()
			
			kernels_flat = []
			for i in range(0, self.out_size):
				if i not in free_index:
					kernels_flat.append(self.conv.weight[i].detach().cpu().numpy().ravel())

			kmeans = cluster.KMeans(n_clusters=self.num_patterns)
			kmeans.fit(kernels_flat)
			centers = kmeans.cluster_centers_
			self.filter_pattern = kmeans.labels_ #returns the index of each filter's pattern

			for i in free_index:
				self.filter_pattern = np.insert(self.filter_pattern, i, -1)
			for center in centers:
				kmeans = cluster.KMeans(n_clusters=self.num_groups)
				kmeans.fit(center.reshape(-1, 1))
				pattern = np.reshape(kmeans.labels_, self.conv.weight[0].shape).astype(np.int32)
				self.patterns.append(pattern)

	#after detemining the groups we apply this function to change weights
	@torch.no_grad()
	def change_weights(self):
		if self.free_filter_num != self.out_size:
			w_np_cpu = self.conv.weight.detach().cpu().numpy()
			# print(self.patterns)
			for i in range(self.out_size):
				if self.filter_pattern[i] != -1:
					# print(self.filter_pattern[i])
					pattern_flat = (self.patterns[self.filter_pattern[i]]).ravel()
					sums = np.zeros(self.num_groups)
					counts = np.zeros(self.num_groups)
					means = np.zeros(self.num_groups)
					w_flat = w_np_cpu[i].ravel()
					sums = np.bincount(pattern_flat, weights=w_flat)
					uniq = np.unique(pattern_flat, return_counts=True)
					counts[uniq[0]] += uniq[1]
					means = sums/counts
					w_flat = means[pattern_flat]
					self.conv.weight[i] = torch.Tensor(w_flat.reshape(self.conv.weight[i].shape)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) 

	def forward(self, x):
		return self.conv(x)

class VGG(nn.Module):
	def __init__(self, cut_layer=-1, enable_hd=False, base_matrix=None, class_hvs=None, class_norms=None):
		super(VGG, self).__init__()
		self.cut_layer = cut_layer
		self.enable_hd = enable_hd
		self.base_matrix, self.class_hvs, self.class_norms = base_matrix, class_hvs, class_norms
		
		self.conv1_1 = ConvLayer(in_channels=3, out_channels=64, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer1_1 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True))
		self.conv1_2 = ConvLayer(in_channels=64, out_channels=64, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer1_2 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
		
		self.conv2_1 = ConvLayer(in_channels=64, out_channels=128, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer2_1 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(inplace=True))
		self.conv2_2 = ConvLayer(in_channels=128, out_channels=128, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer2_2 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
		
		self.conv3_1 = ConvLayer(in_channels=128, out_channels=256, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer3_1 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(inplace=True))
		self.conv3_2 = ConvLayer(in_channels=256, out_channels=256, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer3_2 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(inplace=True))
		self.conv3_3 = ConvLayer(in_channels=256, out_channels=256, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer3_3 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
		
		self.conv4_1 = ConvLayer(in_channels=256, out_channels=512, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer4_1 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(inplace=True))
		self.conv4_2 = ConvLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer4_2 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(inplace=True))
		self.conv4_3 = ConvLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer4_3 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
		
		self.conv5_1 = ConvLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer5_1 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(inplace=True))
		self.conv5_2 = ConvLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer5_2 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(inplace=True))
		self.conv5_3 = ConvLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1, num_groups=N_G, num_patterns=N_P)
		self.layer5_3 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
		
		#self.fc = nn.Linear(512, 10)
		#'''
		self.fc1 = nn.Linear(512, 4096)
		self.relu1 = nn.ReLU(True)
		self.fc2 = nn.Linear(4096, 4096)
		self.relu2 = nn.ReLU(True)
		self.fc3 = nn.Linear(4096, 100)
		#'''

	def forward(self, x):
		out = self.conv1_1(x)
		out = self.layer1_1(out)
		if self.cut_layer > 1 or self.cut_layer == -1:
			out = self.conv1_2(out)
			out = self.layer1_2(out)
		if self.cut_layer > 2 or self.cut_layer == -1:
			out = self.conv2_1(out)
			out = self.layer2_1(out)
		if self.cut_layer > 3 or self.cut_layer == -1:
			out = self.conv2_2(out)
			out = self.layer2_2(out)
		if self.cut_layer > 4 or self.cut_layer == -1:
			out = self.conv3_1(out)
			out = self.layer3_1(out)
		if self.cut_layer > 5 or self.cut_layer == -1:
			out = self.conv3_2(out)
			out = self.layer3_2(out)
		if self.cut_layer > 6 or self.cut_layer == -1:
			out = self.conv3_3(out)
			out = self.layer3_3(out)
		if self.cut_layer > 7 or self.cut_layer == -1:
			out = self.conv4_1(out)
			out = self.layer4_1(out)
		if self.cut_layer > 8 or self.cut_layer == -1:
			out = self.conv4_2(out)
			out = self.layer4_2(out)
		if self.cut_layer > 9 or self.cut_layer == -1:
			out = self.conv4_3(out)
			out = self.layer4_3(out)
		if self.cut_layer > 10 or self.cut_layer == -1:
			out = self.conv5_1(out)
			out = self.layer5_1(out)
		if self.cut_layer > 11 or self.cut_layer == -1:
			out = self.conv5_2(out)
			out = self.layer5_2(out)
		if self.cut_layer > 12 or self.cut_layer == -1:
			out = self.conv5_3(out)
			out = self.layer5_3(out)

		if self.cut_layer == -1:
			out = out.view(out.size(0), -1)
			out = self.fc1(out)
			out = self.relu1(out)
			out = self.fc2(out)
			out = self.relu2(out)
			out = self.fc3(out)
			return out
		out = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2)
		if self.enable_hd:
			out = torch.matmul(out, self.base_matrix) #enc
			if self.training: #tanh makes trainable
				out = torch.tanh(out)
			else:
				out = torch.sign(out)
				out[out == 0] = 1
			out = torch.matmul(out, self.class_hvs) #score
			out = torch.div(out, self.class_norms)
			return out
		return out #when it is cut, e.g., cut_layer=13

def change_weights(model):
	for m in model.modules():
		if(isinstance(m, ConvLayer)):
			m.change_weights()

def create_patterns(model, free_percent):
	index = 0
	for m in model.modules():
		if(isinstance(m, ConvLayer)):
			m.create_patterns(free_percent[index])
			index += 1

criterion = nn.CrossEntropyLoss()
def train(net, epoch, optimizer, manip_weights=False):
	start_time = time.time()
	net.train()
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cpu(), targets.cpu()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()		
		optimizer.step()
		if manip_weights and (batch_idx%10 == 0 or batch_idx == len(train_loader)-1):
			change_weights(net)
	
	net.eval()
	test_loss = 0.0
	correct = 0.0
	total = 0.0
	for batch_idx, (inputs, targets) in enumerate(test_loader):
		if use_cuda:
			inputs, targets = inputs.cpu(), targets.cpu()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		test_loss += loss.data
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()

	accuracy = correct / (total + 0.0)
	end_time = time.time()
	print('Epoch: ' , epoch+1, '| test loss: %.4f' % test_loss, '| test accuracy: %.4f' % accuracy, '| time: %.1f' % (end_time - start_time))
	return accuracy

# step (1): train the original unpatterned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = VGG().to(device)
if os.path.isfile(base_path) and rewrite == False:
	net = torch.load(base_path)
else:
	epoch = EPOCH
	milestones = [int(0.3*epoch), int(0.6*epoch), int(0.8*epoch)]
	lr = 0.1
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
	acc_max = -1
	for epc in range(epoch):
		acc = train(net, epc, optimizer, False)
		if acc > acc_max:
			acc_max = acc
			net_best = deepcopy(net)
		if epc in milestones:
			lr = lr * 0.2
			optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
	print('Max accuracy = %.4f' % acc_max)
	net = deepcopy(net_best)
	torch.save(net, base_path)
	with open(base_path.replace('.pth', '.txt'), 'w') as f:
		f.write(str(acc_max))

# step (2): train the patterned model
if os.path.isfile(pattern_path) and rewrite == False:
	net_pattern = VGG().to(device)
	net_pattern = torch.load(pattern_path)
else:
	create_patterns(net, free_percent)
	change_weights(net)
	epoch = EPOCH
	milestones = [int(0.3*epoch), int(0.6*epoch), int(0.8*epoch)]
	lr = 0.1
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
	acc_max = -1
	for epc in range(epoch):
		acc = train(net, epc, optimizer, True) #manipulate weights
		if acc > acc_max:
			acc_max = acc
			net_best = deepcopy(net)
		if epc in milestones:
			lr = lr * 0.2
			optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
	print('Max accuracy = %.4f' % acc_max) 
	net_pattern = deepcopy(net_best)
	torch.save(net_pattern, pattern_path)
	with open(pattern_path.replace('.pth', '.txt'), 'w') as f:
		f.write(str(acc_max))

# step (3): extract features and train HD model
if os.path.isfile(hd_path) and rewrite == False:
	with open(hd_path, 'rb') as f:
		base_matrix, class_hvs_best, class_norms_best, hd_acc_train = pickle.load(f, encoding='latin1')	
else:
	train_features = []
	test_features = []
	train_y = []
	test_y = []
	net_pattern.cut_layer = cut_layer
	net_pattern.eval()
	for idx, data in enumerate(train_loader):
		if idx%10 == 0:
			print(idx)
		images, labels = data
		images, labels = images.cpu(), labels.cpu()
		for i in labels:
			train_y.append(i)
		feature = net_pattern(images)
		feature = torch.mean(feature.view(feature.size(0), feature.size(1), -1), dim=2)
		feature = feature.view(feature.size(0), -1)
		(b,_) = feature.shape
		for i in range(b):
			sub = feature[i,:]
			sub = sub.view(-1)
			train_features.append(sub.data.tolist()) 
		np.save("train100.npy",train_features)
		np.save("train100_labels.npy",train_y)
	for idx, data in enumerate(test_loader):
		if idx%10 == 0:
			print(idx)
		images, labels = data
		images, labels = images.cpu(), labels.cpu()
		for i in labels:
			test_y.append(i)
		feature = net_pattern(images)
		feature = torch.mean(feature.view(feature.size(0), feature.size(1), -1), dim=2)
		feature = feature.view(feature.size(0), -1)
		(b,_) = feature.shape
		for i in range(b):
			sub = feature[i,:]
			sub = sub.view(-1)
			test_features.append(sub.data.tolist())
		np.save("test100.npy",test_features)
		np.save("test100_labels.npy",test_y)
	base_matrix, class_hvs_best, class_norms_best, hd_acc_train = train_HD(X_train=train_features, y_train=train_y, X_test=test_features, y_test=test_y, D=D, BW=BW, epochs=20, lr=1.0, rp_sign=True, enable_test=False, log_=True)
	base_matrix, class_hvs_best, class_norms_best = np.array(base_matrix), np.array(class_hvs_best), np.array(class_norms_best)
	with open(hd_path, 'wb') as f:
		pickle.dump([base_matrix, class_hvs_best, class_norms_best, hd_acc_train], f)

# step (4): make a patterned CNN+HD model by attaching the trained HD
base_matrix_tensor = (torch.from_numpy(base_matrix.T)).type(torch.FloatTensor)
class_hvs_best_tensor =  (torch.from_numpy(class_hvs_best.T)).type(torch.FloatTensor)
class_norms_best_tensor = (torch.from_numpy(class_norms_best)).type(torch.FloatTensor)
net_pattern_hd = deepcopy(net_pattern)
net_pattern_hd.enable_hd = True
net_pattern_hd.base_matrix, net_pattern_hd.class_hvs, net_pattern_hd.class_norms = base_matrix_tensor, class_hvs_best_tensor, class_norms_best_tensor
if os.path.isfile(pattern_hd_path) and rewrite == False:
	net_pattern_hd = torch.load(pattern_hd_path)
else:
	epoch = EPOCH
	milestones = [int(0.3*epoch), int(0.6*epoch), int(0.8*epoch)]
	lr = 0.02 #maybe start with small lr, e.g., 0.02 or 0.004 (I guess it should also depend on how many layers are cut)
	optimizer = optim.SGD(net_pattern_hd.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
	acc_max = -1
	for epc in range(epoch):
		acc = train(net_pattern_hd, epc, optimizer, True) #manipulate weights
		if acc > acc_max:
			acc_max = acc
			net_best = deepcopy(net_pattern_hd)
		if epc in milestones:
			lr = lr * 0.2
			optimizer = optim.SGD(net_pattern_hd.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
	print('Max accuracy = %.4f' % acc_max)
	net_pattern_hd = deepcopy(net_best)
	torch.save(net_pattern_hd, pattern_hd_path)
	with open(pattern_hd_path.replace('.pth', '.txt'), 'w') as f:
		f.write(str(acc_max))
