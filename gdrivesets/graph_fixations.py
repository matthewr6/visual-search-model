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

data = data['3w']

data.sort()

def pseudocdf(arr):
    return [(v, float(idx)/len(arr)) for idx, v in enumerate(arr)]

# plt.plot(np.arange(len(data)), pseudocdf(data))
plt.plot(*zip(*pseudocdf(data)))
plt.xlabel('Milliseconds')
plt.ylabel('CDF')

plt.show()