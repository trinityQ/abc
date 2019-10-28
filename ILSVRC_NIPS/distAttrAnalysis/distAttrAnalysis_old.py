
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

from InceptionModel.inception_utils import load_model, load_labels_vocabulary, make_predictions_and_gradients, top_label_id_and_score
from IntegratedGradients.integrated_gradients import integrated_gradients, random_baseline_integrated_gradients
from VisualizationLibrary.visualization_lib import Visualize, show_pil_image, pil_image
from imagenet import create_readable_names_for_imagenet_labels

MODEL_LOC='../attributesGen/InceptionModel/inception_v3_2016_08_28_frozen.pb'
LABELS_LOC='../attributesGen/InceptionModel/imagenet_slim_labels.txt'



slim = tf.contrib.slim
tensorflow_master = ""
checkpoint_path   = "../preReq/inception-v3/inception_v3.ckpt"
max_epsilon       = 20.0
image_width       = 299
image_height      = 299
#batch_size        = 1
batch_size        = 50
model_dir = '../preReq/imageNetMapping'


eps = 2.0 * max_epsilon / 255.0
batch_shape = (batch_size, image_height, image_width, 3)
num_classes = 1001


categories = pd.read_csv("../preReq/nips-2017-adversarial-learning-development-set/categories.csv")
image_classes = pd.read_csv("../preReq/nips-2017-adversarial-learning-development-set/images.csv")


nameV3 = create_readable_names_for_imagenet_labels()


def plot(imattr):
    print("In plotting function")
    x = np.array(imattr).flatten()

    parameters = norm.fit(x)
    print("parameters",parameters)
    #exit()

    y = np.linspace(-0.01,0.01,10000)

    fittedPdf = norm.pdf(y,loc=parameters[0],scale=parameters[1]/50.0)

    normalPdf = norm.pdf(y)

    plt.plot(y,fittedPdf,"red") 
    plt.plot(y,normalPdf,"blue") 
    plt.hist(x,bins=1000)
    plt.show()
    #sns.distplot(x)


def getFilteredData(imattra, imdata, block, sign):
    print(imattra)
    print(imattra.shape,imdata.shape,block,sign)


    plot(imattra)
    exit()
    #absAttr = (np.sort(imattr.flatten()))
    #absAttr = absAttr[::-1]

    imattr = np.copy(imattra)
    #print(imattra.flatten())
    #imattr = np.absolute(imattra)
    #print(imattr.flatten())

    #print(imattr.flatten())
    #imattr = np.multiply(-1,imattr)
    #print(imattr.flatten())
    #imattr = np.absolute(imattr)

    absAttr = (np.sort(imattr.flatten()))
    absAttr = absAttr[::-1]

    #absAttr = (np.sort(list(filter(lambda x: x > 0, imattr.flatten()))))[::-1]

    #if len(absAttr):
        # If there's no value after filtering above,
        # there's nothing to drop. Hence, the above check.
        #dropPercentile = np.percentile(absAttr, 99.0)
        #print(dropPercentile)
        #absAttr = np.array(list(filter(lambda x: x < dropPercentile, absAttr)))
        #pass

    #print(dropPercentile,"absAttr",absAttr)
    toKeepCnt = int((block / 100.0) * len(absAttr)) - 1
    if toKeepCnt <= 0:
        pivot = sign
    else:
        pivot = absAttr[toKeepCnt]

    print(toKeepCnt,"toKeepCnt","pivot",pivot)
    #print("pivot",pivot)
    
    filteredData = np.copy(imdata)
    for i in range(imdata.shape[0]):
        for j in range(imdata.shape[1]):
            #for k in range(imdata.shape[2]):
                #if imattr[i][j][k] > pivot:
                    #filteredData[i][j][k]=1
            if imattr[i][j][0] >= pivot or imattr[i][j][1] >= pivot or imattr[i][j][2] >= pivot:
                ##filteredData[i][j][0] = random.uniform(-1, 1)
                ##filteredData[i][j][1] = random.uniform(-1, 1)
                ##filteredData[i][j][2] = random.uniform(-1, 1)
                filteredData[i][j][0] = 0
                filteredData[i][j][1] = 0
                filteredData[i][j][2] = 0


    #imattrFilter =  np.array([[0 if x > pivot  else 1 for x in row] for row in imattr])

    #print(imattrFilter)
    #filteredData = np.multiply(imattrFilter, imdata)
    #filteredData = np.multiply(1.0, imdata)
    #print(filteredData.shape)
    return filteredData







class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join( 
          model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    self.uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      self.uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    self.node_id_to_uid = {}
    self.uid_to_node_id = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        self.node_id_to_uid[target_class] = target_class_string[1:-2]
        #print(target_class_string[1:-2])
        self.uid_to_node_id[target_class_string[1:-2]] = target_class

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in self.node_id_to_uid.items():
      if val not in self.uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = self.uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

  def uid_to_string(self,uid):
      if uid not in self.uid_to_human:
          return ''
      return self.uid_to_human[uid]

  def uid_to_id(self,uid):
      if uid not in self.uid_to_node_id:
          return ''
      return self.uid_to_node_id[uid]



labelPath = '../dataStore/originalILSVRC/labels.npy'
imagePathOrig = sys.argv[1]
attrPath = sys.argv[2]
analysisStore = sys.argv[3] 
imageStore = sys.argv[4]



#imagePathOrig = '../dataStore/attackILSVRC/FGSM2'
#attrPath = '../dataStore/attackILSVRCAttr/FGSM2/'
#labelPath = '../dataStore/originalILSVRC/labels.npy'
#analysisStore = "./attrFlipDataStore/FGSM2ILSVRC_New.npy"
#imageStore = "./attrFlipDataStore/FGSM2/"

labelStore = np.load(labelPath)

def load_images(img_path,batch_shape):
    print("In load imagesI")
    images = np.zeros(batch_shape)
    labels = []
    path = []
    idx = 0
    batch_size = batch_shape[0]
    for i in range(1000):
        fullPath = os.path.join(img_path,str(i)+".JPEG")
        print(fullPath,labelStore[i])
        images[idx, :, :, :] = np.load(fullPath+".npy")
        labels.append(labelStore[i])
        path.append(str(i)+".JPEG")
        idx+=1
        if idx == batch_size:
            yield labels, images, path
            labels = []
            path = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield labels, images, path






# Load the Inception model for attributions.
attr_sess, attr_graph = load_model(MODEL_LOC)
inception_predictions_and_gradients = make_predictions_and_gradients(attr_sess, attr_graph)
attr_labels = load_labels_vocabulary(LABELS_LOC)
image_iterator = load_images(imagePathOrig, batch_shape)
attr_iterator = load_images(attrPath, batch_shape)



#percent = [1,2,4,5,10,20,30,40,50,60,70,80,90,95,96,98,99]
#percent = [0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,10,15,20,25]
percent = [0.0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40]
#percent = [0.5,1,2,4,5,10,15,20,25,30,40,50,60,70,75,80,85,90,95,96,98,99]
print(len(percent))

counterSame = 0
counterPert = 0
counterSamePos = np.zeros(len(percent));
counterPertPos = np.zeros(len(percent));
counterSameNeg = np.zeros(len(percent));
counterPertNeg = np.zeros(len(percent));
scoreStore = np.zeros((3,1000,len(percent),2))



for i in range(20):
    labels, images , path= next(image_iterator)
    _, attrs , path= next(attr_iterator)
    for j in range(0,batch_size):
        topLabel, score = top_label_id_and_score(images[j], inception_predictions_and_gradients)

        attributions = attrs[j]
        img = (images[j]+1.0)/2.0*255.0
        #img = np.uint8((images[j]+1.0)/2.0*255.0)


        if topLabel == labels[j]:
            counterSame += 1
        else:
            counterPert += 1

        count = i*batch_size+j;
        for k in range(len(percent)):
	    #mask = np.array(Visualize(
    			#attributions, img,
    			#clip_above_percentile=100,
    			#clip_below_percentile=percent[k],
    			#morphological_cleanup=False,
    			#outlines=True,
    			#overlay=False))

            #fileName = './temp/o'+str(k)
            #a = np.uint8(mask)
            #imgo = Image.fromarray(a)
            #print(fileName)
            #imgo.save(fileName,"JPEG")


            #positive mask: keep the mask
            posImage = getFilteredData(attributions,images[j],percent[k],1) 
            #fileName = './temp/pm'+str(k)
            #a = np.uint8((posImage+1.0)/2.0*255.0)
            #imgo = Image.fromarray(a)
            #print(fileName)
            #imgo.save(fileName,"JPEG")

            fileName = imageStore+str(count)+"_"+str(percent[k])+"pos.JPEG"
            a = np.uint8((posImage+1.0)/2.0*255.0)
            imgo = Image.fromarray(a)
            #print(fileName)
            #imgo.save(fileName,"JPEG")

            #negative mask: remove the mask
            negImage = posImage
            #fileName = './temp/nm'+str(k)
            #a = np.uint8((negImage+1.0)/2.0*255.0)
            #imgo = Image.fromarray(a)
            #print(fileName)
            #imgo.save(fileName,"JPEG")

            scorePosStore = np.zeros((1000,len(percent)))
            scoreNegStore = np.zeros((1000,len(percent)))


            count = i*batch_size+j;

            #check prediction positive
            topLabelPos, scorePos = top_label_id_and_score(posImage, inception_predictions_and_gradients)
            scoreStore[0][count][k][0] = topLabelPos
            scoreStore[0][count][k][1] = scorePos


            #check predictio negative
            topLabelNeg = topLabelPos
            scoreNeg = scorePos
            scoreStore[1][count][k][0] = topLabelNeg
            scoreStore[1][count][k][1] = scoreNeg

            scoreStore[2][count][k][0] = topLabel
            scoreStore[2][count][k][1] = score

            print(count,":",percent[k]," topLabel",topLabel,"score",score,"labels",labels[j],"posLabel",topLabelPos,"scorePos",scorePos,"negLabel",topLabelNeg,"scoreNeg",scoreNeg)


            if topLabel == labels[j]:
                #we have a correct prediction no flipping should happen
                if topLabelPos == topLabel:
                    counterSamePos[k] += 1
            
                if topLabelNeg == topLabel:
                    counterSameNeg[k] += 1
            else:
                #we have perturbed image a flipping should be observed
                if topLabelPos !=  topLabel:
                    counterPertPos[k] += 1

                if topLabelNeg != topLabel:
                    counterPertNeg[k] += 1




print("counterSame",counterSame,"counterPert",counterPert)

for i in range(len(percent)):
    print(percent[i],"% counterSamePos",counterSamePos[i]," percent ",counterSamePos[i]*100.0/counterSame)
    print(percent[i],"% counterSameNeg",counterSameNeg[i]," percent ",counterSameNeg[i]*100.0/counterSame)
    print(percent[i],"% counterPertPos",counterPertPos[i]," percent ",counterPertPos[i]*100.0/counterPert)
    print(percent[i],"% counterPertNeg",counterPertNeg[i]," percent ",counterPertNeg[i]*100.0/counterPert)



print counterPert
for i in range(len(percent)):
    s = str(percent[i])+","+str(counterPertPos[i])+","+str(counterPertPos[i]*100.0/counterPert)
    print s




#print(scoreStore)
 
np.save(analysisStore,scoreStore)
#scoreStore = np.load(analysisStore)
#print(scoreStore)

        #exit()




#print(scoreStore)
 
np.save(analysisStore,scoreStore)
#scoreStore = np.load(analysisStore)
#print(scoreStore)

        #exit()


