
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

from InceptionModel.inception_utils import load_model, load_labels_vocabulary, make_predictions_and_gradients, top_label_id_and_score,top_label_id_and_layer
from IntegratedGradients.integrated_gradients import integrated_gradients, random_baseline_integrated_gradients
from VisualizationLibrary.visualization_lib import Visualize, show_pil_image, pil_image
from imagenet import create_readable_names_for_imagenet_labels

MODEL_LOC='../attributesGen/InceptionModel/inception_v3_2016_08_28_frozen.pb'
LABELS_LOC='../attributesGen/InceptionModel/imagenet_slim_labels.txt'



matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')
#matplotlib.rcParams[key] = eval(config._sections['matplotlib.rcParams'][key]) 
#matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler(color=hexcolor)


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

filterCutOff = 0.9
#power = 10
#samples = 2
samples = 100
e = math.e


nameV3 = create_readable_names_for_imagenet_labels()


def best_fit_distribution(data, bins=1000, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        st.cauchy,st.dweibull,st.gennorm
        #st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine, #cauchy
        #st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk, #dweibull
        #st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon, #gennorm
        #st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        #st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        #st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        #st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        #st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,#nct
        #st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        #st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)
                print(params)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)





def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)



    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


def plot(imattr):
    print("In plotting function")
    x = np.array(imattr).flatten()
    data = pd.DataFrame(x)
    params = st.gennorm.fit(x)

    #parameters = norm.fit(x)
    print("parameters",params)

    y = np.linspace(-0.01,0.01,50000)

    pdf = make_pdf(st.gennorm, params)
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    #data.plot(kind='hist', bins=1000, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)


    fittedPdf = st.gennorm.pdf(y,beta=params[0],loc=params[1],scale=params[2])
    print(st.gennorm.cdf(0.00001,beta=params[0],loc=params[1],scale=params[2]))


    #exit()

    #normalPdf = st.gennorm.pdf(y,*params)

    plt.plot(y,fittedPdf,"red") 
    #plt.plot(y,normalPdf,"blue") 
    plt.hist(x,bins=1000,normed=False,color="green")
    plt.hist(x,bins=1000,normed=True,color="red")
    plt.show()
    exit()
    #sns.distplot(x)








def plot1(imattr):
    print("In plotting function")
    x = np.array(imattr).flatten()



    data = pd.DataFrame(x)

    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=1000, normed=True, alpha=0.5)#, color=plt.rcParams['axes.color_cycle'][1])
    # Save plot limits
    dataYLim = ax.get_ylim()

    # Find best fit distribution
    best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
    best_dist = getattr(st, best_fit_name)

    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(u'El Nino sea temp.\n All Fitted Distributions')
    ax.set_xlabel(u'Temp C')
    ax.set_ylabel('Frequency')

    # Make PDF with best params 
    pdf = make_pdf(best_dist, best_fit_params)

    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data.plot(kind='hist', bins=1000, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)

    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)

    ax.set_title(u'El Nino sea temp. with best fit distribution \n' + dist_str)
    ax.set_xlabel(u'Temp. (C)')
    ax.set_ylabel('Frequency')
    plt.show()
    
    exit()



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



def randomFilter(x0,x1,x2):
    #generate random numbers
    s = np.random.uniform(0,1)
    #print(s)

    #div = e**(1.0/(1.0-filterCutOff))
    #print(e,1.0/div)
    #x0 = e**(1.0/(1.0-cd0))/div - e/div
    #x1 = e**(1.0/(1.0-cd1))/div - e/div
    #x2 = e**(1.0/(1.0-cd2))/div - e/div

    if x0 > s or x1 > s or x2 > s:
        #print(cd0,x0,cd1,x1,cd2,x2)
        #print("returning 1")
        return 1, s
    else:
        #print("returning 0")
        return 0 , s


    #print(cd0,x0,cd1,x1,cd2,x2)





def getFilteredData(imattra, imdata, power, sign):
    #print(imattra)
    print(imattra.shape,imdata.shape,power,sign)


    imattr = np.copy(imattra)
    x = np.array(imattr).flatten()
    params = st.gennorm.fit(x)
    filteredData = np.copy(imdata)

    filteredDataNp = np.zeros((samples,filteredData.shape[0],filteredData.shape[1],filteredData.shape[2]))
    filteredDataPlaceHolder = np.zeros((samples,filteredData.shape[0],filteredData.shape[1]))

    for i in range(samples):
        filteredDataNp[i] = filteredData
    
    count = 0
    for i in range(imdata.shape[0]):
        for j in range(imdata.shape[1]):
            cd0 = st.gennorm.cdf(np.abs(imattr[i][j][0]),beta=params[0],loc=params[1],scale=params[2])
            cd1 = st.gennorm.cdf(np.abs(imattr[i][j][1]),beta=params[0],loc=params[1],scale=params[2])
            cd2 = st.gennorm.cdf(np.abs(imattr[i][j][2]),beta=params[0],loc=params[1],scale=params[2])
            x0 = cd0**power;
            x1 = cd1**power;
            x2 = cd2**power;



            for l in range(samples):
                filterFlag, s = randomFilter(x0,x1,x2) 
                if filterFlag == 1:

                    #print(cd0,cd1,cd2,x0,x1,x2,s)
                    filteredDataPlaceHolder[l][i][j] = 1
                    count = count + 1

            #for l in range(samples):
                #if cd0 > 0.99 or cd1 > 0.99 or cd2 > 0.99:
                    #filteredDataNp[l][i][j][0]=0
                    #filteredDataNp[l][i][j][1]=0
                    #filteredDataNp[l][i][j][2]=0
 


    tot = imdata.shape[0]*imdata.shape[1]*samples

    #exit()


    #previous paper has alredy hinted to an optimal removal strategy we want to remove 0.1% only!

    optimalRemoval = 4.0
    currentRemoval = count*100.0/tot

    ratio = optimalRemoval/currentRemoval

    updatedCount = 0
    for l in range(samples):
        for i in range(imdata.shape[0]):
            for j in range(imdata.shape[1]):
                if filteredDataPlaceHolder[l][i][j] == 1:
                    s = np.random.uniform(0,1)
                    if s <= ratio:
                        filteredDataNp[l][i][j][0]=0
                        filteredDataNp[l][i][j][1]=0
                        filteredDataNp[l][i][j][2]=0
                        updatedCount = updatedCount+1

    print("count is",count, " vs ", tot, " percent ",currentRemoval ," updatedCount ",updatedCount*100.0/tot)
    #exit()

    return filteredDataNp







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



imagePathOrig = sys.argv[1]
labelPath = sys.argv[2] 
attrPath = sys.argv[3]
analysisStore = sys.argv[4] 
imageStore = sys.argv[5]
origFlag = int(sys.argv[6])



#imagePathOrig = '../dataStore/attackILSVRC/FGSM2'
#labelPath = '../dataStoreNew/originalILSVRC/labels.npy'
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



#power = [8,10,15]
power = [75]
#power = [2,5,8,10,15]
print(len(power))

counterSame = 0
counterPert = 0


crossEntropyStore = np.zeros((len(power),1000))
countStore = np.zeros((len(power),1000))
sameFlag = np.zeros(1000)


flipProbStore = np.zeros((len(power),1000))
entropyStore = np.zeros((len(power),1000))
flipMaxIndexStore = np.zeros((len(power),1000))

softMaxLabelStore = np.zeros(1000)
softMaxConfStore = np.zeros(1000)
actualLabelStore = np.zeros(1000)





print(sameFlag.shape[0])


for i in range(20):
    labels, images , path= next(image_iterator)
    _, attrs , path= next(attr_iterator)
    for j in range(0,batch_size):
        count = i*batch_size+j;

        topLabel, px = top_label_id_and_layer(images[j], inception_predictions_and_gradients)
        topLabel, softMaxScore = top_label_id_and_score(images[j], inception_predictions_and_gradients)

        softMaxLabelStore[count]=topLabel
        softMaxConfStore[count]=softMaxScore
        actualLabelStore[count]=labels[j]
        
        attributions = attrs[j]
        img = (images[j]+1.0)/2.0*255.0
        #img = np.uint8((images[j]+1.0)/2.0*255.0)

        if topLabel == labels[j]:
            counterSame += 1
            sameFlag[count] = 1
        else:
            counterPert += 1

        for k in range(len(power)):
            filteredImages = getFilteredData(attributions,images[j],power[k],1)
            qx = np.zeros((1,1001))

            flipMaxIndex = np.zeros(1001)

            #flipMaxIndexStore = np.zeros((len(power),1000))

            countChange = 0
            countSame = 0
            for s in range(samples):
                #fileName = './temp/'+str(count)+"_"+str(k)+"_"+str(s)+".JPEG"
                #a = np.uint8((filteredImages[s]+1.0)/2.0*255.0)
                #imgo = Image.fromarray(a)
                #print(fileName)
                #imgo.save(fileName,"JPEG")
        
                topLabelTemp, layerTemp = top_label_id_and_layer(filteredImages[s], inception_predictions_and_gradients)

                print("topLabel ",topLabel, " topLabelTemp:",topLabelTemp, " actualLabel:",labels[j])
                qx = qx+(np.array(layerTemp)/float(samples))


                flipMaxIndex[int(topLabelTemp)] = flipMaxIndex[int(topLabelTemp)]+1

                if topLabel == labels[j]:
                    if topLabelTemp == topLabel:
                        countSame = countSame + 1
                else:
                    if topLabelTemp != topLabel:
                        countChange = countChange + 1

            maxIndex = np.argmax(flipMaxIndex)
            flipMaxIndexStore[k][count]=maxIndex 

            #Calculate crossentropy
            print(px.shape)
            crossEntropy = 0
            entropy = 0
            for p in range(px.shape[1]):
                #print(p,crossEntropy,entropy,px[0][p],qx[0][p])
                if px[0][p] < sys.float_info.min:
                    px[0][p] = sys.float_info.min
                if qx[0][p] < sys.float_info.min:
                    qx[0][p] = sys.float_info.min


                crossEntropy = crossEntropy - px[0][p]*math.log(qx[0][p],2)
                entropy = entropy - qx[0][p]*math.log(qx[0][p],2)


            flipProbStore[k][count] = 1.0 - flipMaxIndex[maxIndex]/samples
            #flipProbStore[k][count] = 1 - qx[0][topLabel];   
            #flipProbStore[k][count] = 1.0 - flipMaxIndex[topLabel]/samples;   
            crossEntropyStore[k][count] = crossEntropy
            entropyStore[k][count] = entropy

            print("crossEntropy ",crossEntropy," entropy",entropy," flip prob:",flipProbStore[k][count])
            print("countChange ",countChange, "countSame ",countSame)
            if sameFlag[count] == 1:
                countStore[k][count]=countSame
            else:
                countStore[k][count]=countChange
                

#crossEntropyStore = np.zeros((len(power),1000))
#countStore = np.zeros((len(power),1000))
#sameFlag = np.zeros(1000)


#flipProbStore = np.zeros((len(power),1000))
#entropyStore = np.zeros((len(power),1000))

#flipMaxIndexStore = np.zeros((len(power),1000))
#softMaxLabelStore = np.zeros(1000)
#softMaxConfStore = np.zeros(1000)
#actualLabelStore = np.zeros(1000)





np.save(imageStore+"crossentropy",crossEntropyStore)
np.save(imageStore+"count",countStore)
np.save(imageStore+"sameFlag",sameFlag)
np.save(imageStore+"flipProb",flipProbStore)
np.save(imageStore+"entropy",entropyStore)

np.save(imageStore+"flipMaxIndex",flipMaxIndexStore)
np.save(imageStore+"softMaxLabel",softMaxLabelStore)
np.save(imageStore+"softMaxConf",softMaxConfStore)
np.save(imageStore+"actualLabel",actualLabelStore)



#print stuff

if origFlag == 1:
    for k in range(len(power)):
        for i in range(1000):
            if sameFlag[i] == 1:
                print(power[k]," count:",countStore[k][i]," entropy:",entropyStore[k][i]," crossentropy:",crossEntropyStore[k][i])

else:
    for k in range(len(power)):
        for i in range(1000):
            if sameFlag[i] == 0:
                print(power[k]," count:",countStore[k][i]," entropy:",entropyStore[k][i]," crossentropy:",crossEntropyStore[k][i])

 







