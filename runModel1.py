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


reload(opt)
reload(Model1) 

beginning = 372
change = 10


# Build filters
s1filters = Model1.buildS1filters()
print 'Loaded s1 filters'
protsfile = open('imgprots.dat', 'rb')
imgprots = cPickle.load(protsfile)#[beginning:beginning+change]
print 'Loading objprots filters'
protsfile = open('objprotsCorrect.dat', 'rb')
objprots = cPickle.load(protsfile)
for idx, _ in enumerate(objprots):
    objprots[idx] = objprots[idx]#[beginning:beginning+change]
# objprots = objprots[0:-1] # NOTE THIS IS HACK because objprots was generated from a folder with 41 instead of 40 images. Getting rid of the last img.
print 'Objprots shape:', len(objprots), objprots[0].shape
protsfile = open('naturalImgC2b.dat', 'rb')
imgC2b = cPickle.load(protsfile)
print 'imgC2b: ', len(imgC2b)
imgC2b = imgC2b[0:-1]
with open('S3prots.dat', 'rb') as f:
    s3prots = cPickle.load(f)[:-1]
#num_objs x num_scales x n x n x prototypes

# Model1.buildS3Prots(1720,s1filters,imgprots)

#num_scales x n x n x prototypes

objNames = Model1.getObjNames()

# #objects
hat = 0
butterfly=13
binoculars = 8
tuba = 31
ant = 3
camera = 14
statue = 12
fan = 16
phonograph = 36
piano = 37
spiral = 4
lobster = 22
accordion = 1
turtle = 38

targetIndex = hat
stimnum = 8
location = (0, 1) # ZERO INDEXED

scaleSize = 8
protID = 0
print 'Obj names', objNames[targetIndex]

def check_bounds(x, y):
    wh = 256/3.0
    bounds = [
        location[0] * wh,
        (location[0]+1) * wh,
        location[1] * wh,
        (location[1]+1) * wh
    ]
    print x, y, bounds
    return x >= bounds[0] and x <= bounds[1] and y >= bounds[2] and y <= bounds[3]

img = scipy.misc.imread('example.png')
# img = scipy.misc.imread('stimuli/1.array{}.ot.png'.format(stimnum))
S1outputs = Model1.runS1layer(img, s1filters)
#sif, minV, maxV = Model1.imgDynamicRange(np.mean(S1outputs[scaleSize], axis = 2))
#print 'Sif: ', sif.shape, 'Max: ', maxV, 'Min: ', minV

C1outputs = Model1.runC1layer(S1outputs)
#cif, minV, maxV = Model1.imgDynamicRange(np.mean(C1outputs[scaleSize], axis = 2))
#print 'Cif: ', cif.shape, 'Max: ', maxV, 'Min: ', minV

S2boutputs = Model1.runS2blayer(C1outputs, imgprots)
#s2b, minV, maxV = Model1.imgDynamicRange(S2boutputs[scaleSize][:,:,protID])
#print 's2b: ', s2b.shape, 'Max: ', maxV, 'Min: ', minV

# C2boutputs = Model1.runC1layer(S2boutputs)
feedback = Model1.feedbackSignal(objprots, targetIndex, imgC2b)
print 'feedback info: ', feedback.shape
lipmap = Model1.topdownModulation(S2boutputs,feedback)
protID = np.argmax(feedback)
print feedback[protID], np.mean(feedback)
print 'lipmap shape: ', len(lipmap), lipmap[0].shape
#lm, minV, maxV = Model1.imgDynamicRange(lipmap[scaleSize][:,:,protID])
#print 'lipmap: ', lm.shape, 'Max: ', maxV, 'Min: ', minV

priorityMap = Model1.priorityMap(lipmap,[256,256])

# i = 0
# found = False
# while i < 5 and not found:
#     priorityMap, fx, fy = Model1.inhibitionOfReturn(priorityMap)
#     found = check_bounds(fx, fy)
#     i += 1

# print i, found

# S2boutputs = Model1.prio_modulation(priorityMap, S2boutputs[:3]) # only doing first three scales
# inhibitions = 1
# for i in xrange(inhibitions):
#     priorityMap = Model1.inhibitionOfReturn(priorityMap)

t = Model1.runS3layer(S2boutputs, s3prots)
# t2 = Model1.runC3layer(t)

# print t2
# priorityMap = Model1.inhibitionOfReturn(priorityMap)

print 'Feedback signal shape: ', feedback.shape

numCols = 5
numRows = 12

whichgraph = 'ab'


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

if 'b' in whichgraph:

    plt.figure()
    plt.gray()
    pmap, minV, maxV = Model1.imgDynamicRange(priorityMap)
    dims = pmap.shape
    print dims
    pmap = Model1.scale(priorityMap)
    for i in xrange(dims[0]):
        for j in xrange(dims[0]):
            tmp = pmap[i,j]
            pmap[i,j]= np.exp(np.exp(tmp))
            # pmap[i,j]= np.exp(np.exp(np.exp(tmp)))
    plt.imshow(gaussian_filter(pmap, sigma=3))

if 'c' in whichgraph:

    fig,ax = plt.subplots(nrows = numRows, ncols = change)
    plt.gray()  # show the filtered result in grayscale

    for i in xrange(change):
        for j, scale in enumerate(S2boutputs):
            s2b, minV, maxV = Model1.imgDynamicRange(scale[:,:,i])
            ax[j,i].imshow(s2b)


if 'd' in whichgraph:

    fig,ax = plt.subplots(nrows = numRows, ncols = change)
    plt.gray()  # show the filtered result in grayscale
    plt.axis('off')

    for i in xrange(change):
        for j, scale in enumerate(lipmap):
            s2b, minV, maxV = Model1.imgDynamicRange(scale[:,:,i])
            ax[j,i].imshow(s2b)


plt.show()

# protsfile = open('naturalImgC2b.dat', 'wb')
# try:
#   prots = Model1.buildObjProts(s1filters, imgprots)
#   cPickle.dump(prots, protsfile, protocol = -1)
# except: # Exception as e:
#   tb = traceback.format_exc()
#   print tb
