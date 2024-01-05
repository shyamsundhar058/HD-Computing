import sys
import time
import parse_example
import joblib
import numpy as np
import scipy.linalg as scilin
import Config

def encode(base,data,dim):
  encoding = np.zeros(dim)
  for i in range(0,len(base)):
    encoding = encoding + data[i]*base[i]
  return encoding

#return nFeatures, nClasses, X, y
#def readChoirDat(filename):
def main():
  if len(sys.argv) < 3:
    print('User error')
    print('Specifiy: features | dimensions')
    sys.exit(-1)
  create(sys.argv[1], sys.argv[2])


def create(feat,dim,mu,sigma):
  print("base",dim)
  start = time.time()  

  D = int(dim)
  feat = int(feat) 
  bases = list() # your ID vectors for encoding 
  base = np.zeros(D) # the base +1/-1 hypervector
  #for i in range(int(D/2)):
	#  base[i] = 1
  #for i in range(int(D/2),D):
	#  base[i] = -1

  for i in range(feat):
	  #bases.append(np.random.permutation(base));
    bases.append(np.random.normal(mu,sigma,D))
  print(len(bases))
  file_path = "details.txt"
  with open(file_path, "a") as file:
     file.write(str(len(bases)) + "\n")
     file.close()
  joblib.dump(np.asarray(bases),open("base.pkl","wb"),compress=True)
  end = time.time()
  print('total time: ' + str(end - start))

def createSparse(feat,dim,mu,sigma, s):
  print(dim)
  start = time.time()  

  D = int(dim*s)
  feat = int(feat) 
  bases = list() # your ID vectors for encoding 
  base = np.zeros(dim) # the base +1/-1 hypervector
  
  #for i in range(int(D/2)):
	#  base[i] = 1
  #for i in range(int(D/2),D):
	#  base[i] = -1
  for i in range(feat):
	  #bases.append(np.random.permutation(base));
    sparse = np.random.normal(mu, sigma, D)
    sparse.resize((dim,))
    sparse = np.roll(sparse, i%(dim-D+1))
    sparse[0:i%(dim-D+1)] = np.zeros(i%(dim-D+1))
    bases.append(sparse)
  joblib.dump(np.asarray(bases),open("base.pkl","wb"),compress=True)
  end = time.time()
  print('total time: ' + str(end - start))

if __name__ == "__main__":
	main()