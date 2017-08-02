import json
import numpy as np

types = ['5and2', 'blackandwhite', 'conjunction']

d = []
for t in types:
    with open('humandata/{}.json'.format(t), 'rb') as f:
        j = json.load(f)
        for s in j:
            d.append(len(j[s]))

print np.mean(d)
print np.std(d)