import traceback
import sys
import cPickle
import Model1
import scipy.misc
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import time
import sys
# import matplotlib.pyplot as plt
import ModelOptions1 as opt


reload(opt)
reload(Model1) 

beginning = 372
change = 10


# Build filters
s1filters = Model1.buildS1filters()
print 'Loaded s1 filters'
with open('imgprots.dat', 'rb') as f:
    imgprots = cPickle.load(f)

# objprots = Model1.buildObjProts(s1filters, imgprots, resize=True)
# with open('resizedobjprots.dat', 'wb') as f:
#     cPickle.dump(objprots, f, protocol=-1)

s3prots = Model1.buildS3Prots(1720, s1filters, imgprots)
with open('resizeds3prots.dat', 'wb') as f:
    cPickle.dump(s3prots, f, protocol=-1)