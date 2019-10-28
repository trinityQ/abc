
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

from InceptionModel.inception_utils import load_model, load_labels_vocabulary, make_predictions_and_gradients, top_label_id_and_score
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


def plot(imattr,imattr1,color):
    print("In plotting function")
    x0 = np.array(imattr).flatten()
    x1 = np.array(imattr1).flatten()
    #x2 = np.array(imattr[2]).flatten()
    x = x0
    data = pd.DataFrame(x)
    params = st.gennorm.fit(x)

    #parameters = norm.fit(x)
    print("parameters",params)

    y = np.linspace(-0.01,0.01,10)

    pdf = make_pdf(st.gennorm, params)
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    #data.plot(kind='hist', bins=1000, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)


    fittedPdf = st.gennorm.pdf(y,beta=params[0],loc=params[1],scale=params[2])
    print(st.gennorm.cdf(0.00001,beta=params[0],loc=params[1],scale=params[2]))


    #exit()

    #normalPdf = st.gennorm.pdf(y,*params)

    #plt.plot(y,fittedPdf,"red") 
    #plt.plot(y,normalPdf,"blue") 
    #plt.hist(x,bins=1000,normed=False,color="green")
    #plt.hist(x0,bins=10,normed=False,color="red",alpha=0.5)
    plt.hist([x0,x1],bins=10,normed=False,alpha=1)
    plt.show()
    #plt.hist(x2,bins=10,normed=False,color="green",alpha=0.5)
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
    data.plot(kind='hist', bins=10, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)

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

    y = np.linspace(-0,20,10000)

    fittedPdf = norm.pdf(y,loc=parameters[0],scale=parameters[1]/50.0)

    normalPdf = norm.pdf(y)

    plt.plot(y,fittedPdf,"red") 
    plt.plot(y,normalPdf,"blue") 
    plt.hist(x,bins=1000)
    plt.show()
    #sns.distplot(x)



#points = np.load("./distAttrDataStore/ORIGILSVRC_New.npyentropy.npy")
#pointsX0 = np.load("./distAttrDataStore/FGSM2ILSVRC_New.npyentropy.npy")
#pointsX1 = np.load("./distAttrDataStore/FGSM10ILSVRC_New.npyentropy.npy")
#pointsX2 = np.load("./distAttrDataStore/CWILSVRC_New.npyentropy.npy")
#pointsX3 = np.load("./distAttrDataStore/DFILSVRC_New.npyentropy.npy")
#pointsX4 = np.load("./distAttrDataStore/PGDESPILSVRC_New.npyentropy.npy")
#pointsX5 = np.load("./distAttrDataStore/DIFGSMILSVRC_New.npyentropy.npy")
#pointsX6 = np.load("./distAttrDataStore/MDIFGSMILSVRC_New.npyentropy.npy")
#pointsX7 = np.load("./distAttrDataStore/MIFGSMILSVRC_New.npyentropy.npy")
#pointsX8 = np.load("./distAttrDataStore/LavanILSVRC.npyentropy.npy")






origFol = "./distAttrDataStorePhysical1/ORIG/"
FGSM2Fol = "./distAttrDataStorePhysical1/Lavan/"
FGSM5Fol = "./distAttrDataStorePhysical1/AdverPatchToaster_25/"
CWFol = "./distAttrDataStorePhysical1/AdverPatchBanana_25/"
DFFol = "./distAttrDataStorePhysical1/AdverPatchBanana_30/"
PGDFol = "./distAttrDataStorePhysical1/AdverPatchBanana_35/"



files = [origFol,FGSM2Fol,FGSM5Fol,CWFol,DFFol,PGDFol]

#points = np.load("./distAttrDataStore/ORIGILSVRC_New.npyentropy.npy")
#print(points.shape)
#print(points[0,0:50])

color = ['blue','orange','green','red','purple','brown']
#color = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']




points1 = np.zeros((0))
points2 = np.zeros((0))
points3 = np.zeros((0))
points4 = np.zeros((0))
points5 = np.zeros((0))
points6 = np.zeros((0))

allPoints = [points1,points2,points3,points4,points5,points6]

powerIdx = 0


for i in range(len(files)):
    sameFlag = np.load(files[i]+"sameFlag.npy")
    flipProb = np.load(files[i]+"flipProb.npy")
    #flipProb = 1.0-flipProb
    entropy = np.load(files[i]+"entropy.npy")
    crossentropy = np.load(files[i]+"crossentropy.npy")
    count = np.load(files[i]+"count.npy")

    #print(sameFlag)

    for j in range(sameFlag.size):
        if sameFlag[j] == 1 and i == 0 and j < 100: #this is the original
            allPoints[i] = np.append(allPoints[i],flipProb[0][j])
            #allPoints[i] = np.append(allPoints[i],crossentropy[powerIdx][j])
        elif sameFlag[j] == 0 and i > 0 and j < 100:
            allPoints[i] = np.append(allPoints[i],flipProb[0][j])
            #allPoints[i] = np.append(allPoints[i],crossentropy[powerIdx][j])
            #allPoints[i].append(crossentropy[powerIdx][j])
        #elif sameFlag[j] == 0 and i > 1 and j < 100:
            #allPoints[i] = np.append(allPoints[i],flipProb[0][j])
        else:
            pass

print(allPoints)



#exit()

for i in range(len(allPoints)):
    allPoints[i] = 1-allPoints[i]



bins = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
#bins = [0.0,0.001,0.0025,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]

binPoints = np.zeros((len(allPoints),len(bins)))

for p in range(len(allPoints)):
    for i in allPoints[p]:
        if i == 0.0:
            binPoints[p][0] = binPoints[p][0]+1
        for j in range(len(bins)-1):
            if i <= bins[j+1] and i > bins[j]:
                binPoints[p][j] = binPoints[p][j] + 1


#print(binPoints)    

print("-----------------------------------")
for p in range(len(allPoints)):
    for i in range(len(bins)):
    #print("("+str(bins[i])+","+")")
        print("("+str(bins[i])+","+str(np.sum((binPoints[p]/np.sum(binPoints[p]))[:i+1]))+")")
    print("-----------------------------------")

exit()



#plt.hist([points[0,0:50],pointsX5[0,0:50]],bins=10,normed=False,alpha=1,label=['a','b'])
#plt.hist([points[2,0:50],pointsX0[2,0:50],pointsX1[2,0:50],pointsX2[2,0:50],pointsX3[2,0:50],pointsX4[2,0:50],pointsX5[2,0:50]],bins=10,normed=False,alpha=1,label=['Original','FGSM2','MDIFGSM','CW','PGDESP','DF','MIFGSM'])
#plt.hist([points[1,0:50],pointsX0[1,0:50],pointsX1[1,0:50],pointsX2[1,0:50],pointsX3[1,0:50],pointsX4[1,0:50],pointsX5[1,0:50]],bins=10,normed=False,alpha=1,label=['Original','FGSM2','MDIFGSM','CW','PGDESP','DF','MIFGSM'])
#plt.hist([points[0,0:50],pointsX0[0,0:50],pointsX1[0,0:50],pointsX2[0,0:50],pointsX3[0,0:50],pointsX4[0,0:50],pointsX5[0,0:50],pointsX6[0,0:50],pointsX7[0,0:50],pointsX8[0,0:50]],bins=10,normed=False,color=color,alpha=1,label=['Original','FGSM2','FGSM10','CW','DF','PGDESP','DIFGSM','MDIFGSM','MIFGSM','Lavan'])

#plt.hist([points[0,0:50],pointsX0[0,0:50],pointsX1[0,0:50],pointsX2[0,0:50],pointsX3[0,0:50],pointsX4[0,0:50],pointsX5[0,0:50],pointsX6[0,0:50],pointsX7[0,0:50]],bins=10,color=color,alpha=1,label=['Original','FGSM2','FGSM10','CW','DF','PGDESP','DIFGSM','MDIFGSM','MIFGSM','Lavan'],cumulative=1,density=1)


label = ["Original","FGSM2","FGSM5","CW","DF","PGD"]
plt.hist([allPoints[0],allPoints[1],allPoints[2],allPoints[3],allPoints[4],allPoints[5]],bins=10,color=color,alpha=1,label=label,cumulative=1,density=1)
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


