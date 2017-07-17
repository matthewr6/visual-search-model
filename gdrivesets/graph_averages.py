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

setsizes = ['3', '6', '12', '18']
means = []
errors = []

for size in setsizes:
    means.append(np.mean(data[size]))
    errors.append(np.std(data[size]))

print means
print errors
plt.bar(np.arange(len(setsizes)), means, 0.5, yerr=errors, color='r')
# plt.errorbar(np.arange(len(setsizes)), means, yerr=errors)
plt.show()