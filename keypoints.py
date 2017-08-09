import sys
import cv2
import numpy as np

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

def detect_peaks(image, s=2):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(s,s)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    # eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    # detected_peaks = local_max ^ eroded_background

    # return detected_peaks
    return local_max


img = cv2.imread(sys.argv[1])

# s1 = 5

# s2 = 3

# b1 = cv2.GaussianBlur(img,(s1,s1),5)

# b2 = cv2.GaussianBlur(img,(s2,s2),0)

# diff =  b1 - b2

# sift = cv2.xfeatures2d.SIFT_create()
orbset = 20
orb = cv2.ORB_create(edgeThreshold=orbset, patchSize=orbset)
kp = orb.detect(img, None)

kp, des = orb.compute(img, kp)

img = cv2.drawKeypoints(img, kp, img, color=(0,255,0))

# cv2.imshow('b1', b1)
# cv2.imshow('b2', b2)
# cv2.imshow('diff', diff)
cv2.imshow('img', img)
# cv2.imshow('img', detect_peaks(diff))

# print np.unique(diff)
cv2.waitKey(0)
# cv2.destroyAllWindows()