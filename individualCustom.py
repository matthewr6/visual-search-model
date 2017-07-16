import traceback
import sys
import cPickle
import Model1
import scipy.misc
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import ModelOptions1 as opt
import json
import os


reload(opt)
reload(Model1)


# Build filters
s1filters = Model1.buildS1filters()
protsfile = open('imgprots.dat', 'rb')
imgprots = cPickle.load(protsfile)
with open('gdrivesets/prots/objprots_smallerscales.dat', 'rb') as f:
    objprots = cPickle.load(f)
with open('gdrivesets/prots/objprots_smallerscales.dat', 'rb') as f: # correct file?
    imgC2b = cPickle.load(f)

# img = scipy.misc.imread('gdrivesets/scenes/5and2/setsize{}_{}.png'.format(12, 36), mode='I')
datatype = 'blackandwhite'
size = 3
idx = 20
name = 'setsize{}_{}.png'.format(size, idx)

filename = 'gdrivesets/scenes/{}/{}'.format(datatype, name)
print '{} beginning'.format(name)
img = scipy.misc.imread(filename, mode='I')
S1outputs = Model1.runS1layer(img, s1filters)
C1outputs = Model1.runC1layer(S1outputs)
print 'before s2b'
S2boutputs = Model1.runS2blayer(C1outputs, imgprots)
print 'after s2b'
targetIndex = 0
feedback = Model1.feedbackSignal(objprots, targetIndex, imgC2b)
lipmap = Model1.topdownModulation(S2boutputs,feedback)
protID = np.argmax(feedback)
with open('sample_unmodified_lip.dat', 'wb') as f:
    cPickle.dump(lipmap, f, protocol=-1)

### vis stuff

numCols = 5
numRows = 12

whichgraph = 'a'


if 'a' in whichgraph:
    fig,ax = plt.subplots(nrows = numRows, ncols = numCols)
    plt.gray()  # show the filtered result in grayscale


    for i in xrange(numRows):
        ax[i,0].imshow(img)

    i = 0
    for scale in S1outputs:
        sif, minV, maxV = Model1.imgDynamicRange(np.mean(scale, axis = 2))
        ax[i,1].imshow(sif)
        i += 1

    i = 0
    for scale in C1outputs:
        cif, minV, maxV = Model1.imgDynamicRange(np.mean(scale, axis = 2))
        ax[i,2].imshow(cif)
        i += 1

    i = 0
    for scale in S2boutputs:
        #s2b, minV, maxV = Model1.imgDynamicRange(np.mean(scale, axis = 2))
        s2b, minV, maxV = Model1.imgDynamicRange(scale[:,:,protID])
        ax[i,3].imshow(s2b)
        i += 1

    i = 0
    for scale in lipmap:
        #lm, minV, maxV = Model1.imgDynamicRange(np.mean(scale, axis = 2))
        lm, minV, maxV = Model1.imgDynamicRange(scale[:,:,protID])  
        ax[i,4].imshow(lm)
        i += 1

    ax[0,0].set_title('Original')
    ax[0,1].set_title('S1')
    ax[0,2].set_title('C1')
    ax[0,3].set_title('S2b')
    ax[0,4].set_title('LIP')

plt.show()