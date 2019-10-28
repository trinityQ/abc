import os
#from cleverhans.attacks import FastGradientMethod,CarliniWagnerL2, FastFeatureAdversaries, LBFGS, SPSA
from io import BytesIO
#import IPython.display
import numpy as np
import pandas as pd
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
import imageio
#from cleverhans.model import Model

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

MODEL_LOC='./InceptionModel/inception_v3_2016_08_28_frozen.pb'
LABELS_LOC='./InceptionModel/imagenet_slim_labels.txt'



slim = tf.contrib.slim
tensorflow_master = ""
#checkpoint_path   = "../input/inception-v3/inception_v3.ckpt"
max_epsilon       = 20.0
image_width       = 299
image_height      = 299
#batch_size        = 1
batch_size        = 50
model_dir = '../preReq/imageNetMapping'
train_dir_list = []


eps = 2.0 * max_epsilon / 255.0
batch_shape = (batch_size, image_height, image_width, 3)
num_classes = 1001



categories = pd.read_csv("../preReq/nips-2017-adversarial-learning-development-set/categories.csv")
image_classes = pd.read_csv("../preReq/nips-2017-adversarial-learning-development-set/images.csv")

#print("categories ",categories)
#print("image_class",image_classes)

nameV3 = create_readable_names_for_imagenet_labels()

#exit()



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


print(sys.argv[1]," ",sys.argv[2])

imagePathOrig = sys.argv[1]
attrStore = sys.argv[2]
labelPath = sys.argv[3] #'../dataStoreNew/originalILSVRC/labels.npy'

#imagePathOrig = '../dataStore/attackILSVRC/FGSM5'
#attrStore = '../dataStore/attackILSVRCAttr/FGSM5/'
#labelPath = '../dataStoreNew/originalILSVRC/labels.npy'

os.system("mkdir "+attrStore)
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


for i in range(20):
    labels, images , path= next(image_iterator)
    for j in range(0,batch_size):
        topLabel, score = top_label_id_and_score(images[j], inception_predictions_and_gradients)
        print("topLabel",topLabel,"labels",labels[j])

        img = np.copy(images[j])
        attributions = random_baseline_integrated_gradients(
                        img,
                        topLabel,
                        inception_predictions_and_gradients,
                        steps=50,
                        num_random_trials=10) 
                   
                    
        #print("attributions",attributions.shape,"max",np.max(attributions))
        fileName = attrStore+path[j]
        print(fileName)
        #print("attr shape",attributions.shape)
        #print(attributions)
        #print("_______________")
        np.save(fileName,attributions)
        #attributions = np.load(fileName+".npy")
        #print(s)

        img = np.uint8((img+1.0)/2.0*255.0) 
        npAttrOutlinePert = np.array(Visualize(
    			attributions, img,
    			clip_above_percentile=99,
    			clip_below_percentile=10,
    			morphological_cleanup=False,
    			outlines=True,
    			overlay=True))

        a = np.uint8(npAttrOutlinePert)
        img = Image.fromarray(a)
        print(fileName)
        img.save(fileName,"JPEG")


