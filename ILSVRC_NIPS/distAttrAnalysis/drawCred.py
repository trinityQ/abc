
import sys, os

#print(os.path.abspath('../'))
sys.path.append(os.path.abspath(os.path.join('../attributesGen/')))

import os
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

import seaborn as sns
from scipy import stats

import imageio

import PIL
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import argparse
import os.path
from os import listdir
from os.path import isfile, join
import re
import sys
import tarfile
import random

import numpy as np
from six.moves import urllib


import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
import cycler

banFol = "./distAttrDataStorePhysical1/AdverPatchBanana_30_1/"
contFol = "./distAttrDataStorePhysical1/ORIG1/"

controlEntropy = np.load(contFol+"flipProb.npy")
#controlEntropy = 1.0-controlEntropy
fgsmEntropy = np.load(banFol+"flipProb.npy")
#fgsmEntropy = 1.0-fgsmEntropy
fgsmNewLabel = np.load(banFol+"flipMaxIndex.npy")
#fgsmNewLabel = np.load("./FGSM2.0nSoftMaxLabelStore.npy")
fgsmOrigLabel = np.load(banFol+"actualLabel.npy")

sameFlag = np.load(banFol+"sameFlag.npy")

fgsmEntropy = fgsmEntropy[0]
controlEntropy = controlEntropy[0]
fgsmNewLabel = fgsmNewLabel[0]


controlFlip = controlEntropy
#notMnistFlip = np.load("./notmnist2.0nFlipProbabilityStore.npy")
#rotatedFlip = np.load("./mnistRotat2.0nFlipProbabilityStore.npy")
#fashionFlip = np.load("./fashion2.0nFlipProbabilityStore.npy")



print(controlEntropy.shape,fgsmEntropy.shape)
print("non zero count",np.count_nonzero(controlFlip),np.count_nonzero(controlEntropy),  np.count_nonzero(fgsmEntropy))



#####################################

credFlipFgsm = np.zeros(fgsmEntropy.shape)
credEntropyFgsm = np.zeros(fgsmEntropy.shape)
print(fgsmEntropy.shape)
print("**********************************************")
#print(controlEntropy)


print(fgsmEntropy.shape[0])
for i in range(fgsmEntropy.shape[0]):

    if sameFlag[i] == 1:
        print(i)
        continue

    print(i,fgsmEntropy[i])
    num = fgsmEntropy[i]
    count = 0.0
    for j in controlEntropy:
        if num <= j:
            count  = count + 1.0
    count = count/float(controlEntropy.shape[0])
    credEntropyFgsm[i]=count

print(credEntropyFgsm)

########################
bins = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

binEntropyFgsm=np.zeros(len(bins))

#make bins 

flagStore = np.zeros(credEntropyFgsm.shape[0])
for i in range(credEntropyFgsm.shape[0]):
    if sameFlag[i] == 1:
        print(i)
        continue

    if credEntropyFgsm[i] == 0.0:
        binEntropyFgsm[0] = binEntropyFgsm[0]+1
        flagStore[i]=0
        #print(0)

    for j in range(len(bins)-1):
        if  credEntropyFgsm[i]<= bins[j+1] and credEntropyFgsm[i] > bins[j]:
            binEntropyFgsm[j] = binEntropyFgsm[j] + 1
            flagStore[i]=j
            #print(j)
unique, counts = np.unique(flagStore, return_counts=True)

print(flagStore)


binAccuracySum=np.zeros(len(bins))

#fgsmNewLabel = np.load("./FGSM2.0nSoftMaxLabelStore.npy")
#fgsmOrigLabel = np.load("./FGSM2.0nActualLabelStore.npy")
for i in range(credEntropyFgsm.shape[0]):
    if fgsmNewLabel[i] == fgsmOrigLabel[i]:
        binIdx = int(flagStore[i])
        print(binIdx) 
        binAccuracySum[binIdx] = binAccuracySum[binIdx]+1

print(binAccuracySum/binEntropyFgsm)





print(binEntropyFgsm/(1000-np.sum(sameFlag)))


