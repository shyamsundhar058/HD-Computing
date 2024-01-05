import Config
import joblib
import sys
import random
import math
import numpy as np
import parse_example

D = Config.D
sparse = Config.sparse

def kitchen(datum,base,base2):
  #print("first")
  data = np.array([])
  size = int(D)
  #print(size)
  #print(base2.shape[0])
  #sys.exit(0)
  #if expand < 1:
    #data = datum
  #else:
    #for i in range(expand + 1):
      #data = np.append(data,datum)
  data = datum
  encoded = np.empty(size)
  #assert size == base.shape[0]
  #assert size == base2.shape[0]
  #print("Here")
  datum=datum.ravel()
  for i in range(size):
     #encoded[i] = np.cos(np.dot(datum,base2[i]))
     encoded[i] = np.cos(np.dot(datum,base2[i]) + base[i])*np.sin(np.dot(datum,base2[i]))
     #encoded[i] = encoded[i] / math.sqrt(size)
     
     #encoded[i] = np.exp(-1*base[i]*np.dot(datum,base2[i]))#np.cos(data[i] + base[i])
     #encoded[i] = np.dot(datum,base2[i])
  #print(encoded.shape)
  #sys.exit(0)
  return np.where(encoded < 0, -1, 1)

def encode(data,base,i,t):
  import pandas as pd
  import numpy as np
  data=data
  base2 = joblib.load("base.pkl")
  newData = list()
  N=i
  if len(data)>250000:
    data1 = pd.read_excel("Simulation Measurement_2nd dataset.xlsx", header=None, sheet_name="Measurements", usecols=[19,23], names=['Time', 'current1'], engine='openpyxl')  
    #data1=np.load("wind_train_labels.npy")
    #data1=np.load("train_time.npy")

  else:
    data1 = pd.read_excel("Simulation Measurements.xlsx", header=None, sheet_name="Measurements 1", usecols=[0, 4], names=['Time', 'current'])

    #data1=np.load("wind_test_labels.npy")
    #data1=np.load("test_time.npy")

  time=data1['Time']
  new_label=list()
  print("data1",len(data1))
  print("in encode train shape",len(data))
  print("N",N)
  file_path = "details.txt"
  with open(file_path, "a") as file:
    file.write(str(N) + "\n")
    file.close()

  if len(data1)>250000:
    start=5.0
    end=5.1
  else:
    start=2.0
    end=2.05
  print("start",start)
  print("overlap",int(N*0.85))
  with open(file_path, "a") as file:
    file.write(str("single element") + "\n")
    file.close()
  with open(file_path, "a") as file:
    file.write(str(N) + "\n")
    file.close()

  print(int(N*0.85))
  with open(file_path, "a") as file:
    file.write(str(N*0.85) + "\n")
    file.close()

  for i in range(0,len(data) - N + 1,int(N*(0.85))):
    #print("in loop")
    n_gram = data[i : i + N]
    ngram_time = time[i:i+N]
    newData.append(kitchen(n_gram,base,base2))
    is_oscillated = any(start <= t <= end for t in ngram_time)
    labels = "Oscillated" if is_oscillated else "Normal"
    new_label.append(labels)
  
  from sklearn.preprocessing import LabelEncoder
  label_encoder = LabelEncoder()
  # Encode the string labels to integer values
  new_label = label_encoder.fit_transform(new_label)
  print(np.min(newData),np.max(newData),"minmax")
  print(np.asarray(newData).shape)
  return np.asarray(newData), np.array(new_label)


def condense(data,labels):
    nClasses = np.unique(labels)
    nClasses = len(nClasses)
    data = np.asarray(data)
    num = data.shape[0]
    dim = data.shape[1]
    smaller = np.zeros((nClasses,dim))
    for i in range(0,num):
        smaller[labels[i]] = smaller[labels[i]] + data[i]
    return smaller,np.arange(nClasses)

def retrain (model,traindata,trainlabels,retNum,rate):
    # we assume one model per class, i.e. label 0 is model 0 is the 0th index
    # of the model, etc.
    from copy import deepcopy
    j=deepcopy(model)
    model=np.where(j<0,-1,1)
    modelLabels = np.arange(len(model))
    # retrain iterations or epochs 
    for ret in range(retNum):
        # go stochastically, in random order
        r = list(range(len(traindata)))
        random.shuffle(r)
        correct = 0
        for i in r:
            query = traindata[i]
            answer = trainlabels[i]
            #guess = closestGuess(query,model,modelLabels)
            maxVal = -1
            for m in range(len(model)):
              val = kernel(model[m],query)
              if val > maxVal:
                maxVal = val
                guess = m
            if guess != answer:
                # if wrongly matched, use naive perceptron rule: 
                j[guess] = j[guess] - rate*query  #j not model
                j[answer] = j[answer] + rate*query #j not model
                model=np.where(j<0,-1,1) #
            else:
                correct = correct + 1
        print('Retraining epoch: ' + str(ret) + ' Epoch accuracy:' + str(correct / len(traindata)))
    print("model",np.min(model),np.max(model))
    return model



def sgn(i):
  if i > 0:
    return 1
  else:
    return -1

def gauss(x,y,std):
  n = np.linalg.norm(x - y)
  n = n ** 2
  n = n * -1
  n = n / (2 * (std**2))
  n = np.exp(n)
  return n

def poly(x,y,c,d):
  return (np.dot(x,y) + c) ** d  

def kernel(x,y):
  dotKernel = np.dot
  gaussKernel = lambda x, y : gauss(x,y,25)
  polyKernel = lambda x,y : poly(x,y,3,5)
  cosKernel = lambda x,y : np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))
  #k = gaussKernel
  #k = polyKernel
  k = dotKernel
  #k = cosKernel 
  return k(x,y)



def binarizeSamples(a,b,data,labels):
  newData = list()
  newLabels = list()
  for i in range(data.shape[0]):
    sample = data[i]
    answer = labels[i]
    if answer == a:
      newData.append(sample)
      newLabels.append(1)
    elif answer == b:
      newData.append(sample)
      newLabels.append(-1)
  return np.asarray(newData), np.asarray(newLabels)

def binarize(datum,big,small):
  return np.where(datum > 0,big,small)

def binarizeAll(data,big,small):
  for i in range(data.shape[0]):
    data[i] = binarize(data[i],big,small)
  return data

#def normalize(data):
    




def load (directory,dataset):
    import pandas as pd
    import numpy as np
    data = pd.read_excel("Simulation Measurement_2nd dataset.xlsx", header=None, sheet_name="Measurements", usecols=[19,23], names=['Time', 'current1'], engine='openpyxl')
    data = data.sort_values('Time')
    segment_2_0_to_2_1 = data[(data['Time'] >= 5.0) & (data['Time'] <= 5.1)]

    data = data[(data['Time'] < 5.0) | (data['Time'] > 5.1)]
    import numpy as np
    #import matplotlib.pyplot as plt
    import pandas as pd

    # Create an empty DataFrame for storing the labels
    train_labels_df = pd.DataFrame(columns=['label'])

    train_time_series = []
    # Calculate the number of segments
    for i in range(len(data)):

      label = 'normal'
      # Append the label to the labels DataFrame
      train_labels_df = pd.concat([train_labels_df, pd.DataFrame({'label': [label]})], ignore_index=True)

      '''# Plotting the time series
      plt.plot(segment['Time'], segment['Current'])
      plt.title(labels_df['label'][i])
      plt.xlabel('Time')
      plt.ylabel('Currrent')
      plt.show()
      plt.clf()'''

    # Print the resulting DataFrame with labels
    # Create an empty DataFrame for storing the labels
    # Calculate the number of segments
    for i in range(len(segment_2_0_to_2_1)):
      label = 'oscillation'

      # Append the label to the labels DataFrame
      train_labels_df = pd.concat([train_labels_df, pd.DataFrame({'label': [label]})], ignore_index=True)

    train_time_series=pd.concat([data['current1'],segment_2_0_to_2_1['current1']])
    print(train_time_series)
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE
    train_time_series = train_time_series[1:].to_numpy().reshape(-1, 1)
    labels=train_labels_df['label'][1:].to_numpy().reshape(-1, 1)
    print(train_time_series.shape)
    print(labels.shape)
    print(np.count_nonzero(np.isnan(train_time_series)))
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the string labels to integer values
    train_labels_encoded = label_encoder.fit_transform(labels)
    oversampler = SMOTE(random_state=42, k_neighbors=2)

    # Perform oversampling on the train set
    train_time_series, train_labels_encoded = oversampler.fit_resample(train_time_series, train_labels_encoded)

    data = pd.read_excel("Simulation Measurements.xlsx", header=None, sheet_name="Measurements 1", usecols=[0, 4], names=['Time', 'current'])
    #data = data.sort_values('Time')
    segment_2_0_to_2_1 = data[(data['Time'] >= 2.0) & (data['Time'] <= 2.1)]

    data = data[(data['Time'] < 2.0) | (data['Time'] > 2.1)]
    import numpy as np
    #import matplotlib.pyplot as plt
    import pandas as pd
    # Create an empty DataFrame for storing the labels
    labels_df = pd.DataFrame(columns=['label'])
    time_series = []
    # Calculate the number of segments
    for i in range(len(data)):
      label = 'normal'
      # Append the label to the labels DataFrame
      labels_df = pd.concat([labels_df, pd.DataFrame({'label': [label]})], ignore_index=True)

      '''# Plotting the time series
      plt.plot(segment['Time'], segment['Current'])
      plt.title(labels_df['label'][i])
      plt.xlabel('Time')
      plt.ylabel('Currrent')
      plt.show()
      plt.clf()'''

    # Create an empty DataFrame for storing the labels
    # Calculate the number of segments
    for i in range(len(segment_2_0_to_2_1)):

      # Separate the current channels
      label = 'oscillation'
      # Append the label to the labels DataFrame
      labels_df = pd.concat([labels_df, pd.DataFrame({'label': [label]})], ignore_index=True)
    #print(labels_df)
    time_series=pd.concat([data['current'],segment_2_0_to_2_1['current']])
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE
    time_series=time_series.to_numpy().reshape(-1, 1)
    test_labels=labels_df['label'].to_numpy().reshape(-1,1)
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the string labels to integer values
    test_labels_encoded = label_encoder.fit_transform(test_labels)

    from sklearn.model_selection import train_test_split
    oversampler = SMOTE(random_state=42, k_neighbors=2)
    '''test_time_series=time_series
    test_labels_encoded=test_labels_encoded'''
    # Perform oversampling on the train set
    test_time_series, test_labels_encoded = oversampler.fit_resample(time_series, test_labels_encoded)

    # Calculate the distribution of labels in the train set
    train_label_counts = np.unique(train_labels_df['label'], return_counts=True)
    print("Train set label distribution:")
    for label, count in zip(train_label_counts[0], train_label_counts[1]):
        print(f"Label {label}: {count} samples")

    # Calculate the distribution of labels in the test set
    test_label_counts = np.unique(labels_df['label'], return_counts=True)
    print("\nTest set label distribution:")
    for label, count in zip(test_label_counts[0], test_label_counts[1]):
        print(f"Label {label}: {count} samples")

    '''pathTrain = ''
    pathTrain = pathTrain + traindirectory
    pathTrain = pathTrain + traindataset
    pathTest = ''
    pathTest = pathTest + testdirectory
    pathTest = pathTest + testdataset
    print('Loading datasets')
    nTestFeatures, nTestClasses, testdata, testlabels = parse_example.readChoirDat(pathTest)
    nTrainFeatures, nTrainClasses, traindata, trainlabels = parse_example.readChoirDat(pathTrain)'''
    traindata = np.asarray(train_time_series)
    traindata=traindata.reshape(-1,1)
    np.save("wind_train.npy",traindata)
    trainlabels = np.asarray(train_labels_encoded)
    np.save("wind_train_labels.npy",trainlabels)
    testdata = np.asarray(test_time_series)
    testdata =testdata .reshape(-1,1)
    np.save("wind_test.npy",testdata)
    testlabels = np.asarray(test_labels_encoded)
    np.save("wind_test_labels.npy",testlabels)
    print("train shape",traindata.shape)
    print(trainlabels.shape)
    print(testdata.shape)
    print(testlabels.shape)
    nTrainFeatures=5
    nTrainClasses=2
    return traindata, trainlabels, testdata, testlabels,nTrainFeatures,nTrainClasses