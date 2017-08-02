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

def save_priomap(prio, name):
    pmap = np.exp(np.exp(np.exp(Model1.scale(prio))))
    # plt.imshow(gaussian_filter(pmap, sigma=3), cmap='hot')
    plt.imsave('{}.png'.format(name), gaussian_filter(pmap, sigma=3), format='png', cmap='hot')

with open('sample_unmodified_prio.dat', 'rb') as f:
    priorityMap = cPickle.load(f)


with open('sample_unmodified_lip.dat', 'rb') as f:
    lipmap = cPickle.load(f)

priorityMap = Model1.priorityMap(lipmap,[256,256])
# print np.mean(priorityMap)

save_priomap(priorityMap, 'exhat_color')

inhibitions = 4
displays = [priorityMap]
for i in xrange(inhibitions):
    n = Model1.inhibitionOfReturn(displays[-1])
    print n[1:]
    displays.append(n[0])

fig,ax = plt.subplots(nrows = len(displays), ncols = 2)
# plt.gray()
# plt.pcolor(gaussian_filter(np.exp(np.exp(np.exp(priorityMap))), sigma=3), cmap='hot')
for idx, pmap in enumerate(displays):
    pmap = np.exp(np.exp(Model1.scale(pmap)))
    ax[idx, 0].imshow(gaussian_filter(pmap, sigma=3), cmap='hot')

    pmap = np.exp(pmap)
    ax[idx, 1].imshow(gaussian_filter(pmap, sigma=3), cmap='hot')

plt.show()