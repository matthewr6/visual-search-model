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
with open('gdrivesets/prots/objprots.dat', 'rb') as f:
    objprots = cPickle.load(f)
# for idx, _ in enumerate(objprots):
#     objprots[idx] = objprots[idx]
with open('gdrivesets/prots/objprots.dat', 'rb') as f: # correct file?
    imgC2b = cPickle.load(f)
# imgC2b = imgC2b[0:-1]

# predicted x and y, real x and y
box_radius = (256/5.0)/2.0
targetIndex = 0
def check_bounds(px, py, rx, ry):
    bounds = [
        rx - box_radius,
        rx + box_radius,
        ry - box_radius,
        ry + box_radius
    ]
    print px, py, bounds
    return px >= bounds[0] and px <= bounds[1] and py >= bounds[2] and py <= bounds[3]

# img = scipy.misc.imread('gdrivesets/scenes/5and2/setsize{}_{}.png'.format(12, 36), mode='I')
datatype = '5and2'
with open('gdrivesets/jsondata/{}.json'.format(datatype), 'rb') as f:
    dataset = json.load(f)

if os.path.isfile('gdrivesets/fixationdata/{}_final.txt'.format(datatype)):
    with open('gdrivesets/fixationdata/{}_final.txt'.format(datatype), 'rb') as f:
        already_run = [a.split(' :: ')[0] for a in f.read().split('\n')]
else:
    already_run = []

fixations_allowed = 20
with open('gdrivesets/fixationdata/{}_final.txt'.format(datatype), 'ab') as f:
    for name, position in dataset.iteritems():

        filename = 'gdrivesets/scenes/{}/{}'.format(datatype, name)
        if not os.path.isfile(filename) or name in already_run:
            continue
        print '{} beginning'.format(name)
        img = scipy.misc.imread(filename, mode='I')
        S1outputs = Model1.runS1layer(img, s1filters)
        C1outputs = Model1.runC1layer(S1outputs)
        print 'before s2b'
        S2boutputs = Model1.runS2blayer(C1outputs, imgprots)
        print 'after s2b'
        feedback = Model1.feedbackSignal(objprots, targetIndex, imgC2b)
        lipmap = Model1.topdownModulation(S2boutputs,feedback)
        protID = np.argmax(feedback)

        priorityMap = Model1.priorityMap(lipmap,[256,256])
        print 'priority map created'
        i = 0
        found = False
        while i < fixations_allowed and not found:
            fx, fy = Model1.focus_location(priorityMap)
            found = check_bounds(fx, fy, position[0], position[1])
            if not found:
                priorityMap, _, _ = Model1.inhibitionOfReturn(priorityMap)
            i += 1
        f.write('{} :: {} :: {}\n'.format(name, i, found))
        print '{} completed'.format(name)