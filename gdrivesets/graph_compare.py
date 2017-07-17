import sys
import json

import numpy as np
import matplotlib.pyplot as plt

# N = 5
# men_means = (20, 35, 30, 35, 27)
# men_std = (2, 3, 4, 1, 2)

sizes = ['3', '6', '12', '18']
model_means = []
model_errors = []
human_means = []
human_errors = []

with open('fixationjson/{}_final.json'.format(sys.argv[1]), 'rb') as f:
    model_data = json.load(f)

with open('humandata/{}.json'.format(sys.argv[1]), 'rb') as f:
    human_data = json.load(f)

for size in sizes:
    # model_means.append(np.mean(np.array(model_data[size]) * 250.0))
    # model_errors.append(np.std(np.array(model_data[size]) * 250.0))
    model_means.append(np.mean(np.array([d for d in model_data[size] if d != 20]) * 250.0))
    model_errors.append(np.std(np.array([d for d in model_data[size] if d != 20]) * 250.0))
    human_means.append(np.mean(human_data[size]))
    human_errors.append(np.std(human_data[size]))

N = len(sizes)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, model_means, width, color='r', yerr=model_errors)

women_means = (25, 32, 34, 20, 25)
women_std = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, human_means, width, color='y', yerr=human_errors)

# add some text for labels, title and axes ticks
ax.set_ylabel('RT')
# ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(sizes)

ax.legend((rects1[0], rects2[0]), ('Model', 'Human'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# plt.show()
plt.savefig('graphs/{}_compare_foundonly.png'.format(sys.argv[1]))