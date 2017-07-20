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
import matplotlib.pyplot as plt


reload(opt)
reload(Model1)

scenes = {
    '5and2': {
        'scenes': [(18, 2), (12, 79), (18, 86)],
        'targetIdx': 0
    },
    'blackandwhite': {
        'scenes': [(6, 25), (18, 29), (18, 65)],
        'targetIdx': 3
    },
    'conjunction': {
        'scenes': [(18, 78), (12, 57), (12, 13)],
        'targetIdx': 3
    },
}

# Build filters
s1filters = Model1.buildS1filters()
protsfile = open('imgprots.dat', 'rb')
imgprots = cPickle.load(protsfile)
with open('gdrivesets/prots/objprots.dat', 'rb') as f:
    objprots = cPickle.load(f)

box_radius = (256/5.0)/2.0
def check_bounds(px, py, rx, ry):
    bounds = [
        rx - box_radius,
        rx + box_radius,
        ry - box_radius,
        ry + box_radius
    ]
    return px >= bounds[0] and px <= bounds[1] and py >= bounds[2] and py <= bounds[3]

dataset = {}
for settype in scenes:
    with open('gdrivesets/scenejson/{}.json'.format(settype), 'rb') as f:
        dataset[settype] = json.load(f)

plt.gray()

def save_priomap(prio, name, itr):
    pmap = np.exp(np.exp(np.exp(Model1.scale(prio))))
    plt.imshow(gaussian_filter(pmap, sigma=3))
    plt.savefig('gdrivesets/gifsrc/{}_{}.png'.format(name, itr))


for scenetype in scenes:
    for scene in scenes[scenetype]['scenes']:
        root_name = 'setsize{}_{}'.format(scene[0], scene[1])
        name = '{}.png'.format(root_name)
        position = dataset[scenetype][name]
        filename = 'gdrivesets/scenes/{}/{}'.format(scenetype, name)
        # if not os.path.isfile(filename) or name in already_run:
        #     continue
        print '{} beginning'.format(name)
        fixations_allowed = scene[0]
        img = scipy.misc.imread(filename, mode='I')
        S1outputs = Model1.runS1layer(img, s1filters)
        C1outputs = Model1.runC1layer(S1outputs)
        print '  before s2b'
        S2boutputs = Model1.runS2blayer(C1outputs, imgprots)
        print '  after s2b'
        feedback = Model1.feedbackSignal(objprots, scenes[scenetype]['targetIdx'], objprots)
        lipmap = Model1.topdownModulation(S2boutputs,feedback)
        protID = np.argmax(feedback)

        priorityMap = Model1.priorityMap(lipmap,[256,256])
        print '  priority map created'
        i = 0
        found = False

        save_priomap(priorityMap, root_name, i)

        while i < fixations_allowed and not found:
            fx, fy = Model1.focus_location(priorityMap)
            found = check_bounds(fx, fy, position[0], position[1])
            if not found:
                priorityMap, _, _ = Model1.inhibitionOfReturn(priorityMap)
            i += 1
            if not found:
                save_priomap(priorityMap, root_name, i)
        print '  {}'.format(i)
        print '{} completed'.format(name)