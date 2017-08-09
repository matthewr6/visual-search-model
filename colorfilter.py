import sys
import cv2
import numpy as np

import scipy.ndimage.filters as snf

sizes = [
    64,
    43,
    32,
    26,
    22,
    19
]

# def runPoolingSizes(S1outputs):
#     output = []
#         wdt,hgt,numOrient = S1outputs[k].shape
#         out = []
#         for p in range(numOrient):
#             img = S1outputs[k][:,:,p]
#             img = img[::2,::2] #my interpretation of page5 column2, paragraph2 "positioned over every other column... "
#             result = snf.maximum_filter(img, size= opt.C1RFSIZE)
#             #print 'C1 output shape: ', result.shape
#             out.append(result)
#         output.append(np.dstack(out[:]))

#     # print 'C1 layer shape: ', len(output), output[0].shape
#     return output

# r vs g, b vs y

def intensity(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def makeNormalizedColorChannels(image, thresholdRatio=10.):
    """
        Creates a version of the (3-channel color) input image in which each of
        the (4) channels is normalized.  Implements color opponencies as per 
        Itti et al. (1998).
        Arguments:
            image           : input image (3 color channels)
            thresholdRatio  : the threshold below which to set all color values
                                to zero.
        Returns:
            an output image with four normalized color channels for red, green,
            blue and yellow.
    """
    intens = intensity(image)
    threshold = intens.max() / thresholdRatio
    # logger.debug("Threshold: %d", threshold)
    r,g,b = cv2.split(image)
    cv2.threshold(src=r, dst=r, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
    cv2.threshold(src=g, dst=g, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
    cv2.threshold(src=b, dst=b, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
    R = r - (g + b) / 2
    G = g - (r + b) / 2
    B = b - (g + r) / 2
    Y = (r + g) / 2 - cv2.absdiff(r,g) / 2 - b

    # Negative values are set to zero.
    cv2.threshold(src=R, dst=R, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
    cv2.threshold(src=G, dst=G, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
    cv2.threshold(src=B, dst=B, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
    cv2.threshold(src=Y, dst=Y, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)

    image = cv2.merge((R,G,B,Y))
    return image

def rg(image):
    r,g,_,__ = cv2.split(image)
    return cv2.absdiff(r,g)

def by(image):
    _,__,b,y = cv2.split(image)
    return cv2.absdiff(b,y)

def compareScales(input_img, s1, s2):
    if s2 < s1:
        s1, s2 = s2, s1
    b_img, s_img = (input_img, input_img)
    for i in range(s1):
        b_img = cv2.pyrDown(b_img)
    for i in range(s2):
        s_img = cv2.pyrDown(s_img)
    for i in range(s2 - s1):
        s_img = cv2.pyrUp(s_img)
    s_img = cv2.resize(s_img, b_img.shape[:2][::-1])
    return np.abs(b_img + s_img)

img = makeNormalizedColorChannels(cv2.imread(sys.argv[1]))

# cv2.imshow('rg', rg(img))
# cv2.imshow('by', by(img))
# both = rg(img) + by(img)
# both = np.maximum(rg(img), by(img))
# cv2.imshow('both', snf.maximum_filter(both, size=15))
# img = cv2.imread(sys.argv[1])
s1 = 3
s2 = 6
cv2.imshow('rg', compareScales(rg(img), s1, s2))
cv2.imshow('by', compareScales(by(img), s1, s2))
cv2.waitKey(0)

#http://www.nature.com/nature/journal/v388/n6637/abs/388068a0.html?foxtrotcallback=true