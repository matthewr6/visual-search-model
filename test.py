import numpy as np
import math
import cPickle
import json

with open('S3prots.dat', 'rb') as f:
    s3prots = cPickle.load(f)[:-1]

for idx, thing in enumerate(s3prots):
    # thing is 43 x 600
    s3prots[idx] = np.mean(thing)

print s3prots

import matplotlib.pyplot as plt

plt.plot(np.arange(len(s3prots)), s3prots)
plt.show()