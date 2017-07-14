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

# with open('sample_unmodified_prio.dat', 'rb') as f:
#     priorityMap = cPickle.load(f)

with open('sample_unmodified_lip.dat', 'rb') as f:
    lipmap = cPickle.load(f)

priorityMap = Model1.priorityMap(lipmap,[256,256])

inhibitions = 8
displays = [priorityMap]
for i in xrange(inhibitions):
    displays.append(Model1.inhibitionOfReturn(displays[-1])[0])

fig,ax = plt.subplots(nrows = len(displays), ncols = 2)
plt.gray()

for idx, pmap in enumerate(displays):
    pmap = np.exp(np.exp(Model1.scale(pmap)))
    ax[idx, 0].imshow(gaussian_filter(pmap, sigma=3))

    pmap = np.exp(pmap)
    ax[idx, 1].imshow(gaussian_filter(pmap, sigma=3))

plt.show()