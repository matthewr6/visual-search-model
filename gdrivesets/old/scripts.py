import json
import numpy as np
from scipy import stats

x = [3,6,12,18]

with open('humandata/5and2.json', 'rb') as f:
    data = json.load(f)

# for k in hdata:
#     print (np.mean(hdata[k]) - )/float(k)

# d = [np.mean(hdata[k]) for k in hdata]
# d.sort()

# print d
points = []
for thing in data:
    for t in data[thing]:
        points.append((int(thing), t))

print stats.linregress(*zip(*points))