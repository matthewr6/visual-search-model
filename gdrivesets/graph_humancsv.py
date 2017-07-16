import os
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt

# we care about setsize (3), rt (7)
basename = os.path.basename(sys.argv[1]).split('.')[0]
data = []
with open(sys.argv[1], 'rb') as f:
    reader = csv.reader(f)
    firstline = True
    for row in reader:
        if not firstline:
            data.append((int(row[3]), int(row[7])))
        firstline = False

# filter only the setsize=12
data = [d[1] for d in data if d[0] == 12]

data.sort()

def pseudocdf(arr):
    return [(v, float(idx)/len(arr)) for idx, v in enumerate(arr)]

# plt.plot(np.arange(len(data)), pseudocdf(data))
plt.plot(*zip(*pseudocdf(data)))
plt.xlabel('Milliseconds')
plt.ylabel('CDF')

plt.show()