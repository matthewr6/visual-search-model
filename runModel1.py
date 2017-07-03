import traceback
import sys
import cPickle
import Model1
import scipy.misc
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import ModelOptions1 as opt


reload(opt)
reload(Model1) 


# Build filters
s1filters = Model1.buildS1filters()
print 'Loaded s1 filters'
protsfile = open('imgprots.dat', 'rb')
imgprots = cPickle.load(protsfile)
print 'Loading objprots filters'
protsfile = open('objprotsCorrect.dat', 'rb')
objprots = cPickle.load(protsfile)
# objprots = objprots[0:-1] # NOTE THIS IS HACK because objprots was generated from a folder with 41 instead of 40 images. Getting rid of the last img.
print 'Objprots shape:', len(objprots), objprots[0].shape
protsfile = open('naturalImgC2b.dat', 'rb')
imgC2b = cPickle.load(protsfile)
print 'imgC2b: ', len(imgC2b)
imgC2b = imgC2b[0:-1]
#num_objs x num_scales x n x n x prototypes

#num_scales x n x n x prototypes

objNames = Model1.getObjNames()

# #objects
hat=0
butterfly=13
binoculars = 8

targetIndex = hat
scaleSize = 8
protID = 150
print 'Obj names', objNames[targetIndex]

img = scipy.misc.imread('example.png')
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
lipmap = Model1.topdownModulation(S2boutputs,feedback) 
print 'lipmap shape: ', len(lipmap), lipmap[0].shape
#lm, minV, maxV = Model1.imgDynamicRange(lipmap[scaleSize][:,:,protID])
#print 'lipmap: ', lm.shape, 'Max: ', maxV, 'Min: ', minV

priorityMap = Model1.priorityMap(lipmap,[256,256])

print 'Feedback signal shape: ', feedback.shape

numCols = 5
numRows = 12


# whichgraph = raw_input('a or b:  ')
whichgraph = 'b'


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
    		pmap[i,j]= np.exp(np.exp(np.exp(tmp)))
    plt.imshow(pmap)


plt.show()

# protsfile = open('naturalImgC2b.dat', 'wb')
# try:
# 	prots = Model1.buildObjProts(s1filters, imgprots)
# 	cPickle.dump(prots, protsfile, protocol = -1)
# except: # Exception as e:
# 	tb = traceback.format_exc()
# 	print tb