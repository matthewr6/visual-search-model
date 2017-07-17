import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt

# we care about setsize (3), rt (7)
basename = os.path.basename(sys.argv[1]).split('.')[0]
data = []
with open(sys.argv[1], 'rb') as f:
    data = json.load(f)

sizes = [('3', (0, 0)), ('6', (0, 1)), ('12', (1, 0)), ('18', (1, 1))]

# data = data['18']

# data.sort()

def pseudocdf(arr):
    return [(v, float(idx)/len(arr)) for idx, v in enumerate(arr)]

fig, ax = plt.subplots(nrows=2, ncols=2)

for size, pos in sizes:
    d = data[size]
    d.sort()
    ax[pos[0], pos[1]].plot(*zip(*pseudocdf(d)))
    ax[pos[0], pos[1]].title.set_text(size)
    for tick in ax[pos[0], pos[1]].get_xticklabels():
        tick.set_rotation(90)
        tick.set_fontsize(8)

# plt.plot(np.arange(len(data)), pseudocdf(data))
# plt.plot(*zip(*pseudocdf(data)))
# plt.xlabel('Milliseconds')
# plt.ylabel('CDF')

plt.savefig('graphs/{}_human.png'.format(basename))