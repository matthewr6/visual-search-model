import sys
import json

import numpy as np
import matplotlib.pyplot as plt

sizes = ['3', '6', '12', '18']
model_means = []
model_errors = []
human_means = []
human_errors = []

if sys.argv[1] not in ['5and2', 'blackandwhite', 'conjunction']:
    raise Exception('bad')

with open('fixationjson/{}_final.json'.format(sys.argv[1]), 'rb') as f:
    model_data = json.load(f)

with open('humandata/{}.json'.format(sys.argv[1]), 'rb') as f:
    human_data = json.load(f)

for size in sizes:
    model_means.append(np.mean(np.array(model_data[size]) * 250.0))
    model_errors.append(np.std(np.array(model_data[size]) * 250.0))
    human_means.append(np.mean(human_data[size]))
    human_errors.append(np.std(human_data[size]))

N = len(sizes)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, model_means, width, color='#226666', yerr=model_errors, ecolor='k')

rects2 = ax.bar(ind + width, human_means, width, color='#003333', yerr=human_errors, ecolor='k')

# add some text for labels, title and axes ticks
ax.set_ylabel('Search Time (ms)')
ax.set_xlabel('Set size')
scene_types = {
    '5and2': 'Spatial Configuration Search',
    'blackandwhite': 'Feature Search',
    'conjunction': 'Conjunction Search',
}
plt.suptitle('Model vs Human {}'.format(scene_types[sys.argv[1]]))
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(sizes)

ax.legend((rects1[0], rects2[0]), ('Model', 'Human'), loc=2)

ax.set_ylim(bottom=0, top=5000)


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

# plt.show()
plt.savefig('graphs/{}_compare.png'.format(sys.argv[1]))