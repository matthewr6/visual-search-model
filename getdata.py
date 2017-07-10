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

beginning = 372
change = 10


# Build filters
s1filters = Model1.buildS1filters()
print 'Loaded s1 filters'
protsfile = open('imgprots.dat', 'rb')
imgprots = cPickle.load(protsfile)#[beginning:beginning+change]
print 'Loading objprots filters'
protsfile = open('objprotsCorrect.dat', 'rb')
objprots = cPickle.load(protsfile)
for idx, _ in enumerate(objprots):
    objprots[idx] = objprots[idx]#[beginning:beginning+change]
# objprots = objprots[0:-1] # NOTE THIS IS HACK because objprots was generated from a folder with 41 instead of 40 images. Getting rid of the last img.
print 'Objprots shape:', len(objprots), objprots[0].shape
protsfile = open('naturalImgC2b.dat', 'rb')
imgC2b = cPickle.load(protsfile)
print 'imgC2b: ', len(imgC2b)
imgC2b = imgC2b[0:-1]
with open('S3prots.dat', 'rb') as f:
    s3prots = cPickle.load(f)[:-1]
#num_objs x num_scales x n x n x prototypes

# Model1.buildS3Prots(1720,s1filters,imgprots)

#num_scales x n x n x prototypes

objNames = Model1.getObjNames()

# #objects
hat = 0
butterfly=13
binoculars = 8
tuba = 31
ant = 3
camera = 14
statue = 12
fan = 16
phonograph = 36
piano = 37
spiral = 4
lobster = 22
accordion = 1
turtle = 38
boat = 33
dolphin = 26
cannon = 15
croc = 23
brain = 10
phone = 17
emu = 30
dragonfly = 27
tree = 9

# [(2, True), (4, True), (2, True), (2, True), (5, False), (5, False), (5, False), (5, True), (1, True)] - last may not be right
# [(2, True), (4, True), (4, True), (2, True), (5, False), (5, False), (5, False), (4, True), (2, True)] - x10

# [(3, True), (4, True), (2, True), (2, True), (5, False), (5, False), (5, False), (4, True), (2, True)] - quad
# target_test_set = [
#     (1, butterfly, (1, 1)),
#     (2, fan, (0, 1)),
#     (3, piano, (1, 0)),
#     (4, turtle, (2, 0)),
#     (5, cannon, (1, 2)),
#     (6, statue, (2, 2)),
#     (7, boat, (1, 1)),
#     (8, binoculars, (2, 0)),
#     (9, phonograph, (0, 1))
# ]

# [(5, True), (1, True), (5, True), (4, True), (5, False), (5, False), (2, True), (2, True), (3, True)]
# [(5, False), (1, True), (5, True), (4, True), (5, False), (5, False), (1, True), (2, True), (3, True)]

# [(5, False), (1, True), (5, True), (4, True), (5, False), (5, False), (1, True), (2, True), (3, True)]
# target_test_set = [
#     (1, binoculars, (2, 2)),
#     (2, statue, (0, 1)),
#     (3, hat, (2, 0)),
#     (4, ant, (2, 1)),
#     (5, dolphin, (0, 2)),
#     (6, ant, (0, 2)),
#     (7, accordion, (0, 1)),
#     (8, hat, (1, 0)),
#     (9, turtle, (1, 0))
# ]

#  [(1, True), (4, True), (1, True), (1, True), (2, True), (1, True), (3, True), (5, False), (5, True)] - quad
# target_test_set = [
#     (1, hat, (2, 0)),
#     (2, phonograph, (1, 2)),
#     (3, spiral, (1, 1)),
#     (4, accordion, (0, 0)),
#     (5, piano, (2, 0)),
#     (6, hat, (1, 2)),
#     (7, statue, (1, 0)),
#     (8, dolphin, (2, 1)),
#     (9, binoculars, (2, 0))
# ]

# [(2, True), (2, True), (4, True), (5, True), (2, True), (5, False), (4, True), (5, True), (2, True)] - quad
# target_test_set = [
#     (1, croc, (1, 0)),
#     (2, hat, (2, 0)),
#     (3, dolphin, (2, 0)),
#     (4, brain, (1, 0)),
#     (5, turtle, (1, 1)),
#     (6, phone, (2, 1)),
#     (7, hat, (2, 2)),
#     (8, emu, (1, 2)),
#     (9, piano, (0, 2))
# ]

# [(5, False), (5, True), (3, True), (2, True), (2, True), (5, False), (2, True), (1, True), (5, True)] - quad
target_test_set = [
    (1, tuba, (0, 2)),
    (2, ant, (2, 1)),
    (3, tree, (0, 0)),
    (4, hat, (1, 2)),
    (5, hat, (2, 2)),
    (6, dragonfly, (1, 1)),
    (7, spiral, (0, 2)),
    (8, spiral, (0, 2)),
    (9, brain, (0, 0))
]



fin = []

def check_bounds(loc, x, y):
    wh = 256/3.0
    bounds = [
        loc[0] * wh,
        (loc[0]+1) * wh,
        loc[1] * wh,
        (loc[1]+1) * wh
    ]
    return x >= bounds[0] and x <= bounds[1] and y >= bounds[2] and y <= bounds[3]

for stimnum, targetIndex, location in target_test_set:
    img = scipy.misc.imread('stimuli/1.array{}.ot.png'.format(stimnum))
    S1outputs = Model1.runS1layer(img, s1filters)
    C1outputs = Model1.runC1layer(S1outputs)
    S2boutputs = Model1.runS2blayer(C1outputs, imgprots)
    feedback = Model1.feedbackSignal(objprots, targetIndex, imgC2b)
    lipmap = Model1.topdownModulation(S2boutputs,feedback)

    priorityMap = Model1.priorityMap(lipmap,[256,256])

    i = 0
    found = False
    while i < 5 and not found:
        print i, 'start'
        priorityMap, fx, fy = Model1.inhibitionOfReturn(priorityMap)
        found = check_bounds(location, fx, fy)
        print 'end'
        i += 1

    fin.append((i, found))

print fin