import ModelOptions1 as opt
import cPickle
import random
import pdb
import numpy as np
import scipy.misc as sm
import scipy.signal
from scipy import stats
from scipy import spatial
import math
import scipy.ndimage.filters as snf
import os
import time

reload(opt)

#DEBUG
np.seterr(all='raise') #Floating point errors generate an actual exception rather than a warning


def buildS1filters():
	"""
	This function returns a list of list of S1 (Gabor) filters. Each Gabor
	filter is a square 2D array. The inner
	lists run over orientations (4 orientations per scale), and the outer list
	runs over scales (all the S1 RF sizes defined in the options). We use
	exactly the same equation as in Serre et al. PNAS 2007 (SI text).
	"""
	
	print "Building S1 filters"
	filts=[];
	for RFSIZE in opt.S1RFSIZES:
		filtsthissize=[]
		for o in [45, 90, 135, 180]:
			theta = np.radians(o) 
			#print "RF SIZE:", RFSIZE, "orientation: ", theta / math.pi, "* pi"
			x, y = np.mgrid[0:RFSIZE, 0:RFSIZE] - RFSIZE/2
			sigma = 0.0036 * RFSIZE * RFSIZE +0.35 * RFSIZE + 0.18
			lmbda = sigma / 0.8 
			gamma = 0.3
			x2 = x * np.cos(theta) + y * np.sin(theta)
			y2 = -x * np.sin(theta) + y * np.cos(theta)
			myfilt = (np.exp(-(x2*x2 + gamma * gamma * y2 * y2) / (2 * sigma * sigma))
					* np.cos(2*math.pi*x2 / lmbda))
			#print type(myfilt[0,0])
			myfilt[np.sqrt(x**2 + y**2) > (RFSIZE/2)] = 0.0
			# Normalized like in Minjoon Kouh's code
			myfilt = myfilt - np.mean(myfilt)
			myfilt = myfilt / np.sqrt(np.sum(myfilt**2))
			filtsthissize.append(myfilt.astype('float'))
		filts.append(filtsthissize)
	return filts

def runC1layer(S1outputs):
	"""
	Used for both C1, C2b and C3 layers.
	Input: A stack of 4 2D maps (one per orientation) for each scale (there are 12 scales)
	Output: The output of a C1 cell of scale S and orientation theta us the maximum of S1 cells 
			of identical orientation and scale within the RF of this C1 cell. 
			Note only one RF is used: 9 x 9
			Therefore output is 4 2D maps (one per orientation) for each scale.
			The scales are not merged UNLIKE HMAX.
	"""
	
	# print "Run C1 layer"
  
	output = []
	for k in range(0, len(S1outputs)):
		wdt,hgt,numOrient = S1outputs[k].shape
		#print 'C1 input shape: ', S1outputs[k].shape
		out = []
		for p in range(numOrient):
			img = S1outputs[k][:,:,p]
			img = img[::2,::2] #my interpretation of page5 column2, paragraph2 "positioned over every other column... "
			result = snf.maximum_filter(img, size= opt.C1RFSIZE)
			#print 'C1 output shape: ', result.shape
			out.append(result)
		output.append(np.dstack(out[:]))

	# print 'C1 layer shape: ', len(output), output[0].shape
	return output
	

def runS1layer(imgin, s1f):
	'''
	Input: n x n img
	Output: 4D arrays, 12 (one per scale) 4 (one per orientation) 2D maps 
	'''
	# print "Running S1 layer"
	img = imgin.astype(float)
	# print 'Input shape: ', img.shape
	output=[]
	imgsq = img**2
	cpt=0
	# Each element in s1f is the set of filters (of various orientations) for a
	# particular scale. We also use the index of this scale for debugging
	# purposes in an assertion.

	for scaleidx, fthisscale in enumerate(s1f):
		# We assume that at any given scale, all the filters have the same RF size,
		# and so the RF size is simply the x-size of the filter at the 1st orientation
		# (note that all RFs are assumed square).
		#sprint 'Shape of S1 filter: ', fthisscale[0].shape 
		RFSIZE = fthisscale[0].shape[0]
		assert RFSIZE == opt.S1RFSIZES[scaleidx]
		stride = int(np.round(RFSIZE/4.0))
		# print 'Stride is: ', stride, ' and output shape is: ', img.shape[0]/stride
		outputsAllOrient = []
		# The output of every S1 neuron is divided by the
		# Euclidan norm (root-sum-squares) of its inputs; also, we take the
		# absolute value.
		# As seen in J. Mutch's hmin and Riesenhuber-Serre-Bileschi code.
		# Perhaps a SIGMA in the denominator would be good here?...
		# Though it might need to be adjusted for filter size...
		tmp = snf.uniform_filter(imgsq, RFSIZE)*RFSIZE*RFSIZE
		#tmp = tmp[::stride,::stride] #subsample according to Page5 Miconi et al
		#print 'Size of tmp: ', tmp.shape
		tmp[tmp<0]=0.0
		normim = np.sqrt(tmp) + 1e-9 + opt.SIGMAS1
		assert np.min(normim>0)
		for o in range(0,4):
			# fft convolution; note that in the case of S1 filters, reversing
			# the filters seems to have no effect, so convolution =
			# cross-correlation (...?)
			tmp = np.fft.irfft2(np.fft.rfft2(img) * np.fft.rfft2(fthisscale[o], img.shape))
			# Using the fft convolution requires the following (fun fact: -N/2 != -(N/2) ...)
			tmp =np.roll(np.roll(tmp,-(RFSIZE/2), axis=1),-(RFSIZE/2), axis=0)
			# Normalization
			tmp  /= (normim)
			# crop the border of conv output
			fin = np.abs(tmp[RFSIZE/2:-RFSIZE/2, RFSIZE/2:-RFSIZE/2])
			fin = tmp[::stride,::stride] #perform striding according to page 5 of Miconi et al (might be a more efficient way to do this)
			
			# assert np.max(fin) < 1
			#print 'Output shape on S1 filter: ', fin.shape
			outputsAllOrient.append(fin)
		# We stack together the orientation maps of all 4 orientations into one single
		# 3D array, for each scale/RF size.
		output.append(np.dstack(outputsAllOrient[:]));
		cpt += 1

		
	return output

def extractS3Vector(output):
	numScales = range(len(output))[0:3]
	selectedScale = random.choice(numScales)
	print 'S3 Selected SCALE is: ', selectedScale
	Cchoice = output[selectedScale]
	assert Cchoice.shape[0]> 1 
	shapeX = Cchoice.shape[0] 
	shapeY = Cchoice.shape[1] 
	finished = 0
	numTries = 0
	while(finished == 0):
		posx = random.randrange(shapeX)
		posy = random.randrange(shapeY)
		prot = Cchoice[posx,posy,:]
		threshold = np.sum(prot)
		print 'S3 Threshold is: ', threshold
		if(threshold > 0):
			print 'Prot.flat shape: ', len(prot.flat)
			prot = prot / np.linalg.norm(prot.flat)
			finished = 1
			print 'Size of prot in S3: ', prot.shape, 'after number of tries: ', numTries 
			return prot
		numTries += 1
	


def extract3DPatch(output, nbkeptweights):
	'''
	Input: An 3D array 12 (scales) 4(orientations) 2D maps from C1 layer 
	Output: 4 9x9 stack from a random scale where 100 randomly selected values from this patch were kept 
	unmodified and everything else was set to zero 
	'''
	
	RFsize = opt.C1RFSIZE
	selectedScale = random.choice(range(len(output)))
	print 'Selected scale: ', selectedScale
	Cchoice = 	output[selectedScale]
	print 'Cchoice shape: ', Cchoice.shape
	print 'Shape of X: ', Cchoice.shape[0], ' shape of Y: ', Cchoice.shape[1], 'and RFsize: ', RFsize
	assert Cchoice.shape[0] - RFsize > RFsize 
	shapeX = Cchoice.shape[0] - RFsize
	shapeY = Cchoice.shape[1] - RFsize


	posx = random.randrange(shapeX)
	posy = random.randrange(shapeY)
	prot = Cchoice[posx:posx+RFsize,posy:posy+RFsize,:]
	#print 'Size of prot:', prot.size
	permutedweights = np.random.permutation(prot.size)
	#print 'permuteed weights: ', permutedweights
	keptweights = permutedweights[:nbkeptweights]
	zeroedweights = permutedweights[nbkeptweights:]

	# Here, for normalization, we just need to compute the norm of
	# a single patch, so we don't need to go through the
	# uniform_filter of the squared input thing...
	# Of course only use kept weights in the normalization
	prot = prot / np.linalg.norm(prot.flat[keptweights])
	# Set non-kept weights to -1 - *after* normalizing...
	prot.flat[zeroedweights] = -1
	return prot

def myNormCrossCorr(stack, prot):
	""" This helper function performs a 3D cross-correlation between a 3D stack
	of 2D maps (stack) and a 3D prototype (prot). These have the same depth,
	and we exclude the edges, so only one 2D map is produced. 

	This is done by summing shifted versions of the input stack, multiplied by
	the appropriated value of the prototype.
	
	This method is much faster than anything else (including FFT convolution)
	due to the fact that the majority of weights are set to -1 (i.e. "ignore"),
	following Serre (see extractCpatch()).
	
	For normalization, we also sum shifted versions of the input stack,
	squared. We then divide the output of the cross-correlation with the square
	root of this map. Again, this is equivalent to multiplying the prots by the
	inputs at each point, then dividing the result by the norm of the inputs,
	following Kouh's method.

	"""

	assert prot.shape[2] == stack.shape[2]
	NBPROTS = prot.shape[2]
	RFSIZE = prot.shape[0] # Assuming square RFs, always
	zerothres = RFSIZE*RFSIZE * (-1)
	#cpt = 0 # For debugging
	XSIZE = stack.shape[0]
	YSIZE = stack.shape[1]
	norm = np.zeros((XSIZE-RFSIZE+1, YSIZE-RFSIZE+1))
	o2 = np.zeros((XSIZE-RFSIZE+1, YSIZE-RFSIZE+1))
	pi = np.zeros((XSIZE-RFSIZE+1, YSIZE-RFSIZE+1))
	for k in range(NBPROTS):
	# If all the weights in that slice of the filter are set to -1, don't bother (note
	# that this will be the case for >90% of slices in S3):
		if np.sum(prot[:,:,k]) > zerothres:
			for i in range(RFSIZE):
				for j in range(RFSIZE):
					if prot[i,j,k] > 0: #> 1e-7:
						#cpt += 1
						norm += stack[i:i+1+XSIZE-RFSIZE, j:j+1+YSIZE-RFSIZE, k] ** 2
						pi += prot[i,j,k] ** 2
						o2  +=  stack[i:i+1+XSIZE-RFSIZE, j:j+1+YSIZE-RFSIZE, k] * prot[i,j,k]
	return o2 / (((np.sqrt(norm + 1e-9))*(np.sqrt(pi + 1e-9))) + opt.SIGMAS)



def runS2blayer(C1outputs, prots):
	# print 'Running S2b layer' 
	output=[]
	# For each scale, extract the stack of input C layers of that scale...
	for  scaleNum, Cthisscale in enumerate(C1outputs):
		# print '------------------------------'
		# print 'Working on scale: ', scaleNum
	# If the C input maps are too small, as in, smaller than the S filter,
	# then there's no point in computing the S output; we return a depth-column 
	# of 0s instead
	# Note that we're assuming all the prototypes to have the same siz
		if prots[0].shape[0] >= Cthisscale.shape[0]:
			# print 'Cinput map too small!'
			outputthisscale = [0] * len(prots)
			output.append(np.dstack(outputthisscale[:]))
			continue

		outputthisscale=[]
		for nprot, thisprot in enumerate(prots): # thisprot is 9x9x4
			# Filter cross-correlation !
			#print 'Cross corr step: ', nprot, prots[nprot].shape, ' Scale:', Cthisscale.shape
			tmp = myNormCrossCorr(Cthisscale, thisprot)         
			outputthisscale.append(tmp)
			assert np.max(tmp) < 1
		output.append(np.dstack(outputthisscale[:]))
	# print 'S2b layer shape: ', len(output), output[0].shape
	return output

# def runS3layer(S2boutputs, prots):
# 	print 'Running S3 layer' 
# 	# Only check 3 smallest scales, pg 9, 1st paragraph
# 	# print len(S2boutputs)
# 	S2bsmall = S2boutputs[0:3]
# 	output=[] # want to end up 40 x 43
# 	# For each scale, extract the stack of input C layers of that scale...
# 	for  scaleNum, Cthisscale in enumerate(S2bsmall):
# 		print '------------------------------'
# 		print 'Working on scale: ', scaleNum
# 	# If the C input maps are too small, as in, smaller than the S filter,
# 	# then there's no point in computing the S output; we return a depth-column 
# 	# of 0s instead
# 	# Note that we're assuming all the prototypes to have the same siz
# 		print Cthisscale.shape
# 		print len(prots)
# 		print len(prots[0]) # prots is 40 x 43 x 600
# 		# if prots[0].shape[0] >= Cthisscale.shape[0]:
# 		# 	print 'Cinput map too small!'
# 		# 	outputthisscale = [0] * len(prots)
# 		# 	output.append(np.dstack(outputthisscale[:]))
# 		# 	continue

# 		outputthisscale=[]
# 		for nprot, thisprot in enumerate(prots): # going over objects
# 			# thisprot is 43 x 600
# 			# Cthisscale.shape is n x n x 600
# 			# Filter cross-correlation !
# 			print 'Cross corr step: ', nprot, prots[nprot].shape, ' Scale:', Cthisscale.shape
# 			tmp = myNormCrossCorr(Cthisscale, thisprot)
# 			outputthisscale.append(tmp)
# 			assert np.max(tmp) < 1
# 		output.append(np.dstack(outputthisscale[:]))
# 	print 'S3 layer shape: ', len(output), output[0].shape
# 	return output

def scale_maxes(scales, prio_map):
	relative_focus = np.argmax(prio_map)
	fy = math.floor(relative_focus/256)
	fx = relative_focus % 256

	final = [] # want to be 600 long
	for scale in scales:
		# scale.shape is n x n x 600
		scale_set = [] # will end up being 600 x 1
		sfx = int(fx * scale.shape[0]/256.0)
		sfy = int(fy * scale.shape[1]/256.0)
		for prot_idx in xrange(scale.shape[2]):
			# scale_set.append(np.amax(scale[:,:,prot_idx]))
			scale_set.append(scale[sfy,sfx,prot_idx]) # order of x/y?
		final.append(scale_set)
	# final should currently be 3 x 600
	return np.mean(final, axis=0) # 600-length, 1D

def avg_spearman(a, b):
	spearman_mat = stats.spearmanr(a, b, axis=1)
	lower = np.tril(spearman_mat, k=-1)
	value_count = np.count_nonzero(lower)
	# data[data == 0] = np.nan
	# means = np.nanmean(data[:, 1:], axis=1)
	return np.sum(lower)/value_count

def comparison(a, b):
	return stats.spearmanr(a, b)[0]
	# return spatial.distance.euclidean(a, b)
	# return np.dot(a, b)/np.linalg.norm(a)

def runS3layer(S2boutputs, prots, prio_map):
	print 'Running S3 layer'
	S2bsmall = S2boutputs[:3] # 3 x n x n x 600
	final_output = [] # want to end up with 40 x 43
	# S2bsmall is an array 3 of numpy arrays that are n x n x 600
	maxes = scale_maxes(S2bsmall, prio_map)
	# import matplotlib.pyplot as plt
	# plt.plot(np.arange(len(maxes)), maxes)
	for prot_idx, thisprot in enumerate(prots):
		# thisprot is 43 x 600
		subprots = []
		for points in thisprot:
			subprots.append(comparison(points, maxes))
		final_output.append(subprots) # append a 43-length array of floats

	return final_output # 40 long, and each item is 43 long

def runC3layer(S3outputs):
	print "Running  C3 group (global max of S3 inputs)"

	summed = [np.sum(a) for a in S3outputs]

	print summed

	print 'Max activation for object: ', np.argmax(summed)
	print 'Min activation for object: ', np.argmin(summed)

	return np.argmax(summed) 

def getC2bAverage(objprots):
	'''
		prots shape: 40 objs x 12 scales x (n x n x 600)
	'''
	"""
	objprots = objprots[0:-1] # NOTE THIS IS HACK because objprots was generated from a folder with 41 instead of 40 images. Getting rid of the last img.
	averages = [np.zeros(scale.shape) for scale in objprots[0]] #since scales are same for all obj, we  can use objprots[0] as a representative for the rest
	for obj in objprots:
		#print 'obj' , obj
		for scale_id in xrange(len(obj)):
			averages[scale_id] += obj[scale_id]
	averages = [avg/len(objprots) for avg in averages]
	return averages
	"""
	return np.mean(objprots,axis=0)


def feedbackSignal(objprots, targetIndx, imgC2b): #F(o,P), Eq 4
	# c2b = np.zeros((len(objprots),objprots[0][0].shape[2])) #creates a num_obj x num_prot matrix
	# for obj_id in xrange(len(objprots)):
	# 	obj = objprots[obj_id]
	# 	#compute max for a given scale
	# 	max_acts = [np.max(scale.reshape(scale.shape[0]*scale.shape[1],scale.shape[2]),axis=0) for scale in obj] 
	# 	#compute max over the 12 scales and store as a row in our c2b matrix
	# 	c2b[obj_id] = np.max(np.asarray(max_acts),axis=0)

	C2bavg = getC2bAverage(objprots)	#changed from objprots
	# print 'C2bavg shape', C2bavg.shape
	# print 'Target c2b shape', len(objprots), objprots[0].shape
	feedback = objprots[targetIndx]/C2bavg
	# print 'objprots[target].shape', objprots[targetIndx].shape
#	feedback = ((feedback - np.min(feedback))/np.max(feedback))+1
	feedback = feedback - np.min(feedback)
	feedback = feedback / np.max(feedback)
	# feedback *= 4.0 # 1 to 10.  try scaling exponentially or quadratically next
	feedback += 1.0
	# feedback = feedback ** 2.0
	# print 'Feedback after normalization: ', feedback, np.min(feedback), np.max(feedback)
	return feedback

def scale(arr):
	arr *= 1.0/np.amax(arr)
	arr += 0.5
	return arr

def imgDynamicRange(inmap):
	minVal = np.min(inmap)
	maxVal = np.max(inmap)
	if maxVal == 0:
		return [inmap, 0, 0]
	normalized = (inmap - minVal)/maxVal
	return [normalized, minVal,maxVal]


def topdownModulation(S2boutputs,feedback): #LIP MAP
	#s2boutputs dimension: numScales x n x n x numProts
	lipMap = []
	for scale in xrange(len(S2boutputs)):
		S2bsum = np.sum(S2boutputs[scale], axis = 2)
		S2bsum = S2bsum[:,:,np.newaxis]
		lip = (S2boutputs[scale] * feedback)/(S2bsum + opt.STRNORMLIP)
		lipMap.append(lip)
	return lipMap

# def priorityMap(lipMap):
# 	prMap = []
# 	wdt, hgt = lipMap[0].shape
# 	for scale in xrange(len(lipMap)):
# 		if(scale > 0):
# 			prMap.append(sm.imresize(lipMap[scale], (wdt,hgt), mode='F', interp='bicubic'))
# 		else:
# 			prMap.append(lipMap[scale])
# 	result = np.mean(prMap, axis = 0)

# 	return result
def computeFinalStride(scale):
	RFSIZE = opt.S1RFSIZES[scale]
	stride = int(np.round(RFSIZE/4.0))
	return (2*stride)


def get_relatived_value(arr, rel_idx):
	forward_bias, idx = math.modf(rel_idx * len(arr))
	if idx+1 == len(arr):
		return arr[-1]
	first_bias = 1 - forward_bias
	v1 = np.array(arr[int(idx)]) * first_bias
	v2 = np.array(arr[int(idx)+1]) * forward_bias
	return (v1 + v2).tolist()

def rel_2d_value(mat, rel_x, rel_y):
	return get_relatived_value(get_relatived_value(mat, rel_x), rel_y)

def corresponding_points(ox, oy, stride, size):
	# x values begin at z + xw and go from xs to xs+s
	# y values begin at z + yw and go from ys to ys+s
	# z = 0, s = stride, w = int(np.round(stride/4.0))?
	points = []
	# w = int(np.round(stride/4.0))
	diff = int(np.round(stride * 0.75))
	x = ox * diff
	for i in range(stride):
		# print x
		y = oy * diff
		for j in range(stride):
			if x < size and y < size:
				points.append((x, y))
			y += 1
		x += 1
	return points

def priorityMap(lipMap,originalImgSize): #Eq 6 sum over scales
	#originalImgSize is the size of the original image, e.g., 256x256
	# use wdt, hgt
	wdt, hgt, numProts = lipMap[0].shape
	priorityMap = np.zeros(originalImgSize)
	# priorityMap = np.zeros([wdt, hgt]) # v2

	pointsUsed = np.zeros(originalImgSize)

	# v3
	for scale in xrange(len(lipMap)): # iterating over images
		lip_S = np.sum(lipMap[scale],axis=2)
		dims = lip_S.shape
		# stride = computeFinalStride(scale)
		stride = int(np.round(opt.S1RFSIZES[scale]))
		for i in xrange(dims[0]): # iterate over pixels of LIP (smaller than image.)
			for j in xrange(dims[1]):
				for x, y in corresponding_points(i, j, stride, 256):
					priorityMap[x, y] += lip_S[i, j]
					pointsUsed[x, y] += 1

	# with np.errstate(divide='ignore', invalid='ignore'):
	priorityMap = np.divide(priorityMap, pointsUsed)
	return priorityMap


def buildImageProts(numProts, s1filters): 
	print 'Building ', numProts, 'protoypes from natural images'
	imgfiles = os.listdir(opt.IMAGESFORPROTS)
	prots = []
	for n in range(numProts):
		selectedImg = random.choice(range(len(imgfiles)))
		print '----------------------------------------------------'
		
		imgfile = imgfiles[selectedImg]
		print 'Prot number', n, 'select image: ', selectedImg, imgfile
		
		if(imgfile == '._.DS_Store' or imgfile == '.DS_Store'):
			selectedImg = random.choice(range(len(imgfiles)))
			imgfile = imgfiles[selectedImg]
		img = sm.imread(opt.IMAGESFORPROTS+'/'+imgfile)
		S1outputs = runS1layer(img, s1filters)
		C1outputs = runC1layer(S1outputs)
		prots.append(extract3DPatch(C1outputs, nbkeptweights = opt.NBKEPTWEIGHTS))
	return prots

def buildObjProts(s1filters, imgProts, resize=False, full=False): #computing C2b
	print 'Building object protoypes' 
	imgfiles = os.listdir(opt.IMAGESFOROBJPROTS) #changed IMAGESFOROBJPROTS to IMAGESFORPROTS
	print imgfiles

	prots = [0 for i in range(len(imgfiles)-1)]
	if full:
		prots = [0 for i in range(len(imgfiles))]
	print 'Prots length: ', len(prots)
	for n in range(len(imgfiles)):
		print '----------------------------------------------------'
		print 'Working on object number', n, ' ', imgfiles[n]
		imgfile = imgfiles[n]
		if(imgfile == '.DS_Store' or imgfile == '._.DS_Store' or imgfile == '._1.normal.png' ):
			continue
		tmp = imgfile.strip().split('.')
		#added:
		# tmp1 = tmp[0].split('_')
		# add finish
		# pnum = (int(tmp1[1])) -1
		pnum = int(tmp[0]) - 1
		print 'pnum: ', pnum		

		img = sm.imread(opt.IMAGESFOROBJPROTS+'/'+imgfile, mode='I') # changed IMAGESFOROBJPROTS to get 250 nat images c2b vals
		if resize:
			img = sm.imresize(img, (64, 64))
		
		t = time.time()
		S1outputs = runS1layer(img, s1filters)
		C1outputs = runC1layer(S1outputs)

		S2boutputs = runS2blayer(C1outputs, imgProts)
		#compute max for a given scale
		max_acts = [np.max(scale.reshape(scale.shape[0]*scale.shape[1],scale.shape[2]),axis=0) for scale in S2boutputs]
		C2boutputs = np.max(np.asarray(max_acts),axis=0) #glabal maximums
		#C2boutputs = runC1layer(S2boutputs)
		prots[pnum] = C2boutputs
		timeF = (time.time()-t)
		print "Time elapsed: ", timeF, " Estimated time of completion: ", timeF*(len(imgfiles)-(n+1))
	return prots

def getObjNames():
	print 'Get obj names' 
	imgfiles = os.listdir(opt.IMAGESFOROBJPROTS)
	#print imgfiles
	prots = [0 for i in range(len(imgfiles))]
	
	for n in range(len(imgfiles)):
		imgfile = imgfiles[n]
		if(imgfile == '.DS_Store' or imgfile == '._.DS_Store' or imgfile == '._1.normal.png' ):
			continue
		tmp = imgfile.strip().split('.')
		pnum = (int(tmp[0])) -1
		#print 'pnum: ', pnum		
		prots[pnum] = imgfile
	return prots

	

def buildS3Prots(numprots, s1filters, imgProts, resize=False):
	print 'Building S3 prots'
	imgfiles = os.listdir(opt.IMAGESFOROBJPROTS)
	print 'Numfiles is: ', len(imgfiles), 'Using files: ', (len(imgfiles)-1) 
	numProtsPerObj = numprots/(len(imgfiles)-1)
	prots = [[] for i in range(len(imgfiles))]
	for i in range (len(imgfiles)):
		print '----------------------------------------------------'
		imgfile = imgfiles[i]
		print 'Working on object number', i, ' ', imgfile
		
		if(imgfile == '.DS_Store' or imgfile == '._.DS_Store' or imgfile == '._1.normal.png' ):
			continue
		tmp = imgfile.strip().split('.')
		pnum = (int(tmp[0])) -1
		print 'pnum: ', pnum		

		img = sm.imread(opt.IMAGESFOROBJPROTS+'/'+imgfile)
		if resize:
			img = sm.imresize(img, (64, 64))
		
		for n in range(numProtsPerObj):
			print 'Prot number', n, 'select image: ',  imgfile
			S1outputs = runS1layer(img, s1filters)
			C1outputs = runC1layer(S1outputs)
			S2boutputs = runS2blayer(C1outputs, imgProts)
			e = extractS3Vector(S2boutputs)
			prots[pnum].append(e)
		print 'major percent', float(i)/len(imgfiles)
	return prots

def gauss_2d(focus_x, focus_y, sigma):
	# inverTED vs inverSE?  inverTED may not be the same as inverSE
	# dims = [256, 256]
	# grid = np.empty(dims)
	# for i in xrange(dims[0]):
	# 	for j in xrange(dims[1]):
	# 		grid[i, j] = stats.norm.pdf(i, focus_y, sigma) * stats.norm.pdf(j, focus_x, sigma)
	# return grid
	grid_y, grid_x = np.mgrid[:256, :256]
	return stats.norm.pdf(grid_x, focus_x, sigma) * stats.norm.pdf(grid_y, focus_y, sigma)
	
def focus_location(prio):
	relative_focus = np.argmax(prio)
	y = math.floor(relative_focus/256)
	x = relative_focus % 256
	return (x, y)

def inhibitionOfReturn(prio):
	relative_focus = np.argmax(prio)
	img_size = 256.0*256.0
	focus_y = math.floor(relative_focus/256)
	focus_x = relative_focus % 256
	# k = 0.2 paper used this
	# sigma = 16.667 paper used this
	# print 'before gauss'
	g = (1.0 - opt.GAUSSFACTOR*gauss_2d(focus_x, focus_y, opt.IORSIGMA))
	# print 'after gauss'
	# print g
	# print prio
	# print prio * g
	return prio * g, focus_x, focus_y

def prio_modulation(prio, s2boutputs):
	# prio = (prio - np.min(prio))/np.max(prio)
	copy = np.copy(s2boutputs)
	ret = []
	for idx, scale in enumerate(copy):
		rs = sm.imresize(prio, scale.shape[:2])
		new = np.copy(copy[idx])
		for i in xrange(600):
			new[:,:,i] *= rs
		# copy[idx].fill(1.0)
		ret.append(new)
	return ret

def crop_s2boutputs(s2boutputs, prio):
	relative_focus = np.argmax(prio)
	img_size = 256.0*256.0
	fy = math.floor(relative_focus/256)
	fx = relative_focus % 256

	cropped = []

	for idx, scale in enumerate(s2boutputs):
		sfx = int(fx * scale.shape[0]/256.0)
		sfy = int(fy * scale.shape[1]/256.0)
		window_radius = int(math.ceil((scale.shape[0]/3.0)/2.0)) # assuming square.

		x_size, y_size = scale.shape[:2]

		if x_size - sfx < window_radius:
			sfx = x_size - window_radius
		if sfx < window_radius:
			sfx = window_radius

		if y_size - sfy < window_radius:
			sfy = y_size - window_radius
		if sfy < window_radius:
			sfy = window_radius

		right_bound = min(sfx + window_radius, x_size)
		left_bound = max(sfx - window_radius, 0)
		top_bound = min(sfy + window_radius, y_size)
		bottom_bound = max(sfy - window_radius, 0)
		print bottom_bound, top_bound, left_bound, right_bound
		cropped.append(scale[bottom_bound:top_bound, left_bound:right_bound])

	return cropped