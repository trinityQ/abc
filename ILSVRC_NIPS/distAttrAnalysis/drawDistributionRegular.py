
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


controlEntropy = np.load("./mnist2.0nEntropyStore.npy")
notMnistEntropy = np.load("./notmnist2.0nEntropyStore.npy")
rotatedEntropy = np.load("mnistRotat2.0nEntropyStore.npy")
fashionEntropy = np.load("./fashion2.0nEntropyStore.npy")



controlFlip = np.load("./mnist2.0nFlipProbabilityStore.npy")
controlFlip  = 1.0 - controlFlip
notMnistFlip = np.load("./notmnist2.0nFlipProbabilityStore.npy")
notMnistFlip = 1.0 - notMnistFlip
rotatedFlip = np.load("./mnistRotat2.0nFlipProbabilityStore.npy")
fashionFlip = np.load("./fashion2.0nFlipProbabilityStore.npy")
fashionFlip = 1.0 - fashionFlip





print(notMnistFlip,notMnistEntropy)


print(controlFlip.shape,notMnistFlip.shape,rotatedFlip.shape,controlEntropy.shape,notMnistEntropy.shape,rotatedEntropy.shape)
print("non zero count",np.count_nonzero(controlFlip),np.count_nonzero(controlEntropy),  np.count_nonzero(notMnistFlip),np.count_nonzero(notMnistEntropy),np.count_nonzero(rotatedFlip),np.count_nonzero(rotatedEntropy))
#exit()



########################
bins = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]

binFlipNot = np.zeros(len(bins))
binFlipFas = np.zeros(len(bins))
binFlipCont = np.zeros(len(bins))

for i in notMnistFlip:
    if i == 0.0:
        binFlipNot[0] = binFlipNot[0]+1
    for j in range(len(bins)-1):
        if i <= bins[j+1] and i > bins[j]:
            binFlipNot[j] = binFlipNot[j] + 1


for i in fashionFlip:
    if i == 0.0:
        binFlipFas[0] = binFlipFas[0]+1
    for j in range(len(bins)-1):
        if i <= bins[j+1] and i > bins[j]:
            binFlipFas[j] = binFlipFas[j] + 1

for i in controlFlip:
    if i == 0.0:
        binFlipCont[0] = binFlipCont[0]+1
    for j in range(len(bins)-1):
        if i <= bins[j+1] and i > bins[j]:
            binFlipCont[j] = binFlipCont[j] + 1




print(binFlipNot*100/np.sum(binFlipNot))
print(binFlipFas*100/np.sum(binFlipFas))
print(binFlipCont*100/np.sum(binFlipCont),np.sum(binFlipCont))



#print(binFlipCont[:1])
#exit()


for i in range(len(bins)):
    #print("("+str(bins[i])+","+")")
    print("("+str(bins[i])+","+str(np.sum((binFlipNot/np.sum(binFlipNot))[:i+1]))+")")
print("-----------------------------------")
for i in range(len(bins)):
    print("("+str(bins[i])+","+str(np.sum((binFlipFas/np.sum(binFlipCont))[:i+1]))+")")
print("-----------------------------------")

for i in range(len(bins)):
    print("("+str(bins[i])+","+str(np.sum((binFlipCont/np.sum(binFlipCont))[:i+1]))+")")



#print(binFlipRot)
exit()

#print(credFlipNot)

color = ['blue','orange']
#color = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']




label = ["Entropy","Flip"]
plt.hist([credEntropyNot,credFlipNot],bins=30,color=color,alpha=1,label=label,cumulative=0,density=1)
#plt.hist([pointsX1,pointsX3],bins=30,color=color,alpha=1,label=label,cumulative=0,density=1)
print("sunny")

plt.legend(prop={'size':10})
plt.show()


#exit()
#plot(points[2,0:50],pointsX0[2,0:50],"red")
#plot(pointsX0[2,0:50],"green")
#plot(pointsX1[2,0:50],"blue")
#plot(pointsX2[2,0:50],"black")
#plot(pointsX3[2,0:50],"yellow")
#plot(points[1,0:50],"green")
#plot(points[2,0:50],"blue")


