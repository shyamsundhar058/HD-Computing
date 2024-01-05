import Config
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import time
import sys
import createNormalBase
import math
import numpy as np
import random
import joblib
import parse_example
import KernelFunctions
sgn = KernelFunctions.sgn



def trainMulticlass(iterations,rate):
  weights = np.zeros((nTrainClasses,traindata.shape[1]))
  correct = 0
  t = 0
  accuracies = []
  while (correct / traindata.shape[0]) != 1:
    r = list(range(traindata.shape[0]))
    random.shuffle(r)
    correct = 0
    count = 0
    pred=[]
    for i in r:
      sample = traindata[i]
      answer = trainlabels[i]
      maxVal = -1
      guess = -1
      for m in range(nTrainClasses):
        val = kernel(weights[m],sample)
        if val > maxVal:
          maxVal = val
          guess = m 
      if guess != answer:
        weights[guess] = weights[guess] - rate*sample
        weights[answer] = weights[answer] + rate*sample
        
      else:
        pred.append(guess)
        correct += 1
      count += 1
    
    accuracy = 100*testMulticlass(weights)
    accuracies.append(accuracy)
    from sklearn.metrics import classification_report
    report = classification_report(trainlabels, pred)

    # Extract the recall value from the report
    recall = report['recall']

    # Print the recall value
    #print('Recall:', recall)
    print("Iteration: ",t,"Train Accuracy: ",correct / count,"Test Accuracy: ", accuracy, "recall",recall)
    t += 1
  print('Max Accuracy: ' + str(max(accuracies)))
def trainMulticlassBinary(iterations,rate):
  with open(file_path, "a") as file:
     file.write(str("multi_oscillation1") + "\n")
     file.close()

  weights = np.zeros((nTrainClasses,traindata.shape[1]))
  binaryWeights = np.copy(weights)
  correct = 0
  t = 0
  
  while (correct / traindata.shape[0]) != 1:
    pred=[]
    r = list(range(traindata.shape[0]))
    random.shuffle(r)
    correct = 0
    count = 0
    print("n",nTrainClasses)
    import time
    start=time.time()
    for i in r:
      sample = traindata[i]
      answer = trainlabels[i]
      maxVal = -1
      guess = -1
      
      for m in range(nTrainClasses):
        val = kernel(binaryWeights[m],sample)
        if val > maxVal:
          maxVal = val
          guess = m 
      #pred.append(guess)
      #print(answer)
      if guess != answer:
        weights[guess] = weights[guess] - rate*sample
        weights[answer] = weights[answer] + rate*sample
        binaryWeights = np.copy(weights)
        binaryWeights = KernelFunctions.binarizeAll(binaryWeights, 1, -1)
        pred.append(guess)
      else:
        correct += 1
        pred.append(guess)
      count += 1
    #pred.append(guess)
       
    from sklearn.metrics import classification_report
    #print(classification_report(trainlabels, pred))

    # Extract the recall value from the report
    #recall = report['recall']

    # Print the recall value
    print('train time',time.time()-start)
    with open(file_path, "a") as file:
      file.write(str("train time") + "\n")
      file.close()

    with open(file_path, "a") as file:
      file.write(str(time.time()-start) + "\n")
      file.close()

    #print('Recall:', recall)
    if t<=20:
      #print("Iteration: ",t,"Train Accuracy: ",correct / count,"Test Accuracy: ",100*testMulticlass(binaryWeights))
      t += 1
    else:
        break
def testMulticlass(weights):
  correct = 0
  pred=[]
  import time
  start=time.time()
  for i in range(testdata.shape[0]):
    sample = testdata[i]
    answer = testlabels[i]
    maxVal = -1
    for m in range(nTrainClasses):
      val = kernel(weights[m],sample)
      if val > maxVal:
        maxVal = val
        guess = m
    if guess == answer:
      correct += 1
    pred.append(guess)
  with open(file_path, "a") as file:
      file.write(str("test time") + "\n")
      file.close()

  print('test timing',time.time()-start)
  with open(file_path, "a") as file:
      file.write(str(time.time()-start) + "\n")
      file.close()

  from sklearn.metrics import classification_report
  print(classification_report(testlabels, pred))
  with open(file_path, "a") as file:
     file.write(str(classification_report(testlabels, pred)) + "\n")
     file.close()

  return correct / testdata.shape[0]


directory = Config.directory 
dataset = Config.dataset
kernel = KernelFunctions.kernel
'''traindata, trainlabels, testdata, testlabels,nTrainFeatures, nTrainClasses = KernelFunctions.load(directory,dataset)
np.save("traindata_Raw.npy",traindata) 
np.save("train_labels_Raw.npy",trainlabels)
np.save("testdata_Raw.npy",testdata)
np.save("test_labels_Raw.npy",testlabels)'''
init = 1
import time
if init == 1:
  D = KernelFunctions.D
  s1=time.time()
  traindata, trainlabels, testdata, testlabels,nTrainFeatures, nTrainClasses = KernelFunctions.load(directory,dataset)
  for i in [70]:
    nTrainFeatures=i
    nTrainClasses =2
    '''traindata=np.load("wind_train.npy") 
    trainlabels=np.load("wind_train_labels.npy")
    testdata=np.load("wind_test.npy")
    print(testdata)
    testlabels=np.load("wind_test_labels.npy")'''
    print("After load/n",traindata.shape)
    print(trainlabels.shape)
    print(testdata.shape)
    print(testlabels.shape)
    traindata = sklearn.preprocessing.normalize(traindata,norm='l2')
    testdata = sklearn.preprocessing.normalize(testdata,norm='l2') 

    mu = Config.mu
    sigma = Config.sigma #/ 20#1 / (math.sqrt(617)) #/ 24#1 #/ (1.4)
    if Config.sparse == 1:
      createNormalBase.createSparse(D, nTrainFeatures, mu, sigma, Config.s)
    else:
      createNormalBase.create(D,nTrainFeatures,mu,sigma)
    size = int(D)
    base = np.random.uniform(0,2*math.pi,size)
    start = time.time()
    traindata,trainlabels= KernelFunctions.encode(traindata,base,i, "train")
    #assert len(traindata)== len(trainlabels)
    file_path = "details.txt"
    with open(file_path, "a") as file:
      file.write(str(time.time() - start) + "\n")
      file.close()
    print("Encoding training time",time.time() - start)
    print("after encode train shape",traindata.shape)
    start = time.time()
    testdata,testlabels= KernelFunctions.encode(testdata,base,i,"test")
    file_path = "details.txt"
    with open(file_path, "a") as file:
      file.write(str(time.time() - start) + "\n")
      file.close()
    print('Encoding testing time',time.time() - start)
    print("after encode test shape",testdata.shape)
    if Config.sparse == 1:
      joblib.dump(traindata,open(str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'train.pkl',"wb"),compress=True)
      joblib.dump(testdata,open(str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'test.pkl','wb'),compress=True)
    else:
      joblib.dump(traindata,open(str(Config.dataset) + str(Config.D) + 'train.pkl',"wb"),compress=True)
      joblib.dump(testdata,open(str(Config.dataset) + str(Config.D) + 'test.pkl','wb'),compress=True)
    print(traindata.shape,trainlabels.shape)
    print(testdata.shape,testlabels.shape)
    if Config.binaryModel == 1:
      trainMulticlassBinary(260, Config.rate)
    else:
      trainMulticlass(260,Config.rate)
    file_path = "details.txt"
    with open(file_path, "a") as file:
      file.write(str("total program time") + "\n")
      file.close()
    file_path = "details.txt"
    with open(file_path, "a") as file:
      file.write(str(time.time() - s1) + "\n")
      file.close()
    print("tatal program time",time.time()-s1)


else:
  if Config.sparse == 1:
    traindata = joblib.load(str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'train.pkl')
    testdata = joblib.load(str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'test.pkl')
  else:
    traindata = joblib.load(str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'train.pkl')
    testdata = joblib.load(+ str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'test.pkl')

  if Config.binarize == 1:
    traindata = KernelFunctions.binarizeAll(traindata, 1, -1)
    testdata = KernelFunctions.binarizeAll(testdata, 1, -1)
  
  pass




