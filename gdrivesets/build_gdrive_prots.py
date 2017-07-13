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


# Build filters
s1filters = Model1.buildS1filters()
print 'Loaded s1 filters'
with open('../imgprots.dat', 'rb') as pf:
    imgprots = cPickle.load(pf)

objprots = Model1.buildObjProts(s1filters, imgprots, resize=True, full=True)
with open('prots/objprots_25.dat', 'wb') as f:
    cPickle.dump(objprots, f, protocol=-1)

# s3prots = Model1.buildS3Prots(2 * 43, s1filters, imgprots)
# with open('gdrivesets/prots/s3prots_25.dat', 'wb') as f:
#     cPickle.dump(s3prots, f, protocol=-1)

# protsfile = open('objprotsCorrect.dat', 'rb')
# protsfile = open('naturalImgC2b.dat', 'rb')
# imgC2b = cPickle.load(protsfile)
# print 'imgC2b: ', len(imgC2b)
# imgC2b = imgC2b[0:-1]
# with open('S3prots.dat', 'rb') as f:
#     s3prots = cPickle.load(f)[:-1]

# protsfile = open('naturalImgC2b.dat', 'wb')
# try:
#   prots = Model1.buildObjProts(s1filters, imgprots)
#   cPickle.dump(prots, protsfile, protocol = -1)
# except: # Exception as e:
#   tb = traceback.format_exc()
#   print tb