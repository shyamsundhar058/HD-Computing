import numpy as np
import pickle
import sys
import time
import math
from copy import deepcopy
from sklearn import preprocessing

log = True

def binarize(base_matrix):
	return np.where(base_matrix < 0, -1, 1)

def encoding_rp(X_data, base_matrix, rp_sign=False):
	enc_hvs = []
	for i in range(len(X_data)):
		if i % int(len(X_data)/20) == 0:
			if log:
				sys.stdout.write(str(int(i/len(X_data)*100)) + '% ')
				sys.stdout.flush()
		hv = np.matmul(base_matrix, X_data[i])
		if rp_sign:
			hv = binarize(hv)
		enc_hvs.append(hv)
	return enc_hvs


def max_match(class_hvs, enc_hv, class_norms):
		max_score = -np.inf
		max_index = -1
		for i in range(len(class_hvs)):
			score = np.matmul(class_hvs[i], enc_hv) / class_norms[i]
			if score > max_score:
				max_score = score
				max_index = i
		return max_index


def train(X_train, y_train, X_test, y_test, D=1024, BW = -1, epochs=20, lr=1.0, rp_sign=True, enable_test=False, log_=True):
	
	global log
	log = log_
	
	validation_perc = 0.1
	permvar = np.arange(0, len(X_train))
	np.random.shuffle(permvar)
	X_train = [X_train[i] for i in permvar]
	y_train = [y_train[i] for i in permvar]
	cnt_vld = int(validation_perc * len(X_train))
	X_validation = X_train[0:cnt_vld]
	y_validation = y_train[0:cnt_vld]
	X_train = X_train[cnt_vld:]
	y_train = y_train[cnt_vld:]

	#generate base/projection matrix
	base_matrix = np.random.rand(D, len(X_train[0]))
	base_matrix = np.where(base_matrix > 0.5, 1, -1)
	base_matrix = np.array(base_matrix, np.int8) #for potential SIMD

	#encode train and validatoin data
	if log: print('\nEncoding %s train data ' %len(X_train))
	train_enc_hvs = encoding_rp(X_train, base_matrix, rp_sign=rp_sign)
	if log: print('\n\nEncoding %s validation data ' %len(X_validation))
	validation_enc_hvs = encoding_rp(X_validation, base_matrix, rp_sign=rp_sign)

	#initialize classes
	class_hvs = [[0.] * D] * (max(y_train) + 1)
	for i in range(len(train_enc_hvs)):
		class_hvs[y_train[i]] += train_enc_hvs[i]
	class_norms = [np.linalg.norm(hv) for hv in class_hvs]
	#keep track of the best model
	class_hvs_best = deepcopy(class_hvs)
	class_norms_best = deepcopy(class_norms)

	#start normal training
	if epochs > 0 and log:
		if rp_sign:
			print('\n\nstarted %s retraining epochs for model with bipolar encoding' %(epochs))
		else:
			print('\n\nstarted %s retraining epochs for model with integer encoding' %(epochs))
	acc_max_train = -np.inf
	for epoch in range(epochs):
		if log:
			sys.stdout.write('epoch %s: ' %epoch)
			sys.stdout.flush()
		pickList = np.arange(0, len(train_enc_hvs))
		np.random.shuffle(pickList)
		#shuffle train data, re-train:
		for index in pickList:
			predict = max_match(class_hvs, train_enc_hvs[index], class_norms)
			if predict != y_train[index]:
				class_hvs[predict] -= np.multiply(lr, train_enc_hvs[index])
				class_hvs[y_train[index]] += np.multiply(lr, train_enc_hvs[index])
		#obtain validation accuracy and update the model
		correct = 0
		for i in range(len(validation_enc_hvs)):
			predict = max_match(class_hvs, validation_enc_hvs[i], class_norms)
			if predict == y_validation[i]:
				correct += 1
		acc = float(correct)/len(validation_enc_hvs)
		if log:
			sys.stdout.write('%.4f ' %acc)
			sys.stdout.flush()
			if epoch > 0 and epoch%5 == 0:
				print('')
		if acc > acc_max_train:
			acc_max_train = acc
			class_hvs_best = deepcopy(class_hvs)
			class_norms_best = deepcopy(class_norms)
	
	if enable_test:
		if log: print('\nEncoding %s test data ' %len(X_test))
		test_enc_hvs = encoding_rp(X_test, base_matrix, rp_sign=rp_sign)
	if BW == -1: #no quantization
		if enable_test == False:
			return base_matrix, class_hvs_best, class_norms_best, acc_max_train
		else:
			correct = 0
			for i in range(len(test_enc_hvs)):
				predict = max_match(class_hvs_best, test_enc_hvs[i], class_norms_best)
				if predict == y_test[i]:
					correct += 1
			acc_test = float(correct)/len(test_enc_hvs)
			return base_matrix, class_hvs_best, class_norms_best, acc_max_train, acc_test
			
	
	#quantization
	class_hvs = np.array(deepcopy(class_hvs_best))
	quantiles = [float(x)/2**BW for x in range(1, 2**BW)]
	bins = []
	for q in quantiles:
		bins.append(np.quantile(class_hvs, q))
	class_hvs_q = np.digitize(class_hvs, bins=bins) - 2**(BW-1)
	class_norms_q = [np.linalg.norm(hv) for hv in class_hvs_q]
	#keep track of the best model
	class_hvs_q_best = deepcopy(class_hvs_q)
	class_norms_q_best = deepcopy(class_norms_q)
	if BW < 4: #need more epochs for quantized model
		epochs = epochs*2 #in HDnn, the CNN part is retrained, so a factor of 2 is enough here
	else:
		epochs = epochs*2

	if epochs > 0 and log:
		print('\n\nStarted %s retraining epochs for quantization model' %(epochs))
	acc_max_train = -np.inf
	for epoch in range(epochs):
		if log:
			sys.stdout.write('epoch %s: ' %epoch)
			sys.stdout.flush()
		pickList = np.arange(0, len(train_enc_hvs))
		np.random.shuffle(pickList)
		#shuffle train data, re-train:
		for index in pickList:
			predict = max_match(class_hvs_q, train_enc_hvs[index], class_norms_q) #similarity check using quantized classes
			if predict != y_train[index]:
				class_hvs[predict] -= np.multiply(lr, train_enc_hvs[index])
				class_hvs[y_train[index]] += np.multiply(lr, train_enc_hvs[index])
		#obtain validation accuracy and update the model
		correct = 0
		for i in range(len(validation_enc_hvs)):
			predict = max_match(class_hvs_q, validation_enc_hvs[i], class_norms_q)
			if predict == y_validation[i]:
				correct += 1
		acc = float(correct)/len(validation_enc_hvs)
		if log:
			sys.stdout.write('%.4f ' %acc)
			sys.stdout.flush()
			if epoch > 0 and epoch%5 == 0:
				print('')
		if acc > acc_max_train:
			acc_max_train = acc
			class_hvs_q_best = deepcopy(class_hvs_q)
			class_norms_q_best = deepcopy(class_norms_q)
		#re-quantize (probably no strict need to update bins; bins are based on the full precison model that do not change a lot)
		#bins = []
		#for q in quantiles:
		#	bins.append(np.quantile(class_hvs, q))
		class_hvs_q = np.digitize(class_hvs, bins=bins) - 2**(BW-1)
		class_norms_q = [np.linalg.norm(hv) for hv in class_hvs_q]

	if enable_test == False:
		return base_matrix, class_hvs_q_best, class_norms_q_best, acc_max_train
	else:
		correct = 0
		for i in range(len(test_enc_hvs)):
			predict = max_match(class_hvs_q_best, test_enc_hvs[i], class_norms_q_best)
			if predict == y_test[i]:
				correct += 1
		acc_test = float(correct)/len(test_enc_hvs)
		return base_matrix, class_hvs_q_best, class_norms_q_best, acc_max_train, acc_test




def load (directory,dataset):
    traindirectory = directory
    traindataset = dataset
    testdirectory = directory
    testdataset = dataset
    import pandas as pd
    data = pd.read_excel("/content/drive/MyDrive/windfarm/Simulation Measurements.xlsx", header=None, sheet_name="Measurements 1", usecols=[0, 4, 5, 6], names=['Time', 'current1', 'current2', 'current3'], engine='openpyxl')
    segment_2_0_to_2_1 = data[(data['Time'] >= 2.0) & (data['Time'] <= 2.1)]
    
    data = data[(data['Time'] < 2.0) | (data['Time'] > 2.1)]
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    sampling_rate = 30
    num_images = 6 * 30
    store=num_images
    data_points_per_segment = 120001// num_images
    num_images=len(data)//data_points_per_segment
    labels_df = pd.DataFrame(columns=['Image_name','label'])
    oscillation_start_time = pd.Timestamp('1970-01-01') + pd.to_timedelta(2, unit='s')
    oscillation_end_time = pd.Timestamp('1970-01-01') + pd.to_timedelta(2.2, unit='s')
    for i in range(num_images):
	    segment = data.iloc[i * data_points_per_segment: (i + 1) * data_points_per_segment]
	    current1 = segment['current1']
	    label="normal"
	    image_name=f'image_{i+1}.png'
		# Append the label to the labels DataFrame
		labels_df = labels_df.append({'label': label,'Image_name':image_name}, ignore_index=True)

		# Plot and save the image
		plt.plot(current1, 'k')
		plt.axis('off')
		#plt.savefig(f'/content/drive/MyDrive/windfarm/51/image_{i+1}.png', bbox_inches='tight', pad_inches=0)
		plt.clf()

	# Print the resulting DataFrame with labels
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd

	# Define the number of data points per segment

	sampling_rate = 30
	num_images = 6 * 30
	store=num_images
	data_points_per_segment = 120001// num_images
	num_images=len(segment_2_0_to_2_1)//data_points_per_segment
	# Create an empty DataFrame for storing the labels
	#labels_df = pd.DataFrame(columns=['Image_name','label'])

	# Define the time range for oscillation
	oscillation_start_time = pd.Timestamp('1970-01-01') + pd.to_timedelta(2, unit='s')
	oscillation_end_time = pd.Timestamp('1970-01-01') + pd.to_timedelta(2.2, unit='s')

	j=178
	# Calculate the number of segments
	for i in range(num_images):
		# Extract the segment of data
		segment = segment_2_0_to_2_1.iloc[i * data_points_per_segment: (i + 1) * data_points_per_segment]

		# Separate the current channels
		current1 = segment['current1']
		label="oscillated"

		image_name=f'image_{j}.png'
		# Append the label to the labels DataFrame
		labels_df = labels_df.append({'label': label,'Image_name':image_name}, ignore_index=True)

		# Plot and save the image
		plt.plot(current1, 'k')
		plt.axis('off')
		#plt.savefig(f'/content/drive/MyDrive/windfarm/51/image_{j}.png', bbox_inches='tight', pad_inches=0)
		plt.clf()
		j+=1
	# Print the resulting DataFrame with labels
	print(labels_df)
	pathTrain = ''
    pathTrain = pathTrain + traindirectory
    pathTrain = pathTrain + traindataset
    pathTest = ''
    pathTest = pathTest + testdirectory
    pathTest = pathTest + testdataset
    print('Loading datasets')
    nTestFeatures, nTestClasses, testdata, testlabels = parse_example.readChoirDat(pathTest)
    nTrainFeatures, nTrainClasses, traindata, trainlabels = parse_example.readChoirDat(pathTrain)
    traindata = np.asarray(traindata)
    trainlabels = np.asarray(trainlabels)
    testdata = np.asarray(testdata)
    testlabels = np.asarray(testlabels)
    return traindata, trainlabels, testdata, testlabels,nTrainFeatures,nTrainClasses