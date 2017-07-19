import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt

basename = os.path.basename(sys.argv[1]).split('.')[0]
data = []
with open(sys.argv[1], 'rb') as f:
    data = json.load(f)

sizes = [('3', (0, 0)), ('6', (0, 1)), ('12', (1, 0)), ('18', (1, 1))]

def pseudocdf(arr):
    ret = [(0, 0)]
    for idx, v in enumerate(arr):
        if len(ret) and ret[-1][0] == v * 250.0:
            ret[-1] = (v*250.0, float(idx)/len(arr))
        else:
            ret.append((v*250.0, float(idx)/len(arr)))
    return ret

fig, ax = plt.subplots(nrows=2, ncols=2)

for size, pos in sizes:
    d = data[size]
    d.sort()
    ax[pos[0], pos[1]].plot(*zip(*pseudocdf(d)))
    ax[pos[0], pos[1]].title.set_text('Set size {}'.format(size))
    ax[pos[0], pos[1]].set_xlim(left=0, right=5000)
    ax[pos[0], pos[1]].set_ylim(bottom=0, top=1)
    ax[pos[0], pos[1]].set_xlabel('Search Time (adjusted to ms)')
    ax[pos[0], pos[1]].set_ylabel('Percent')

plt.suptitle('Model Fixation Time CDF')

plt.subplots_adjust(hspace=0.5)

plt.savefig('graphs/{}_model.png'.format(basename))