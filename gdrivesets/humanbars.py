import sys
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set(style="white")

top = 2500

fs = 18

plt.rc('font', size=fs)          # controls default text sizes
plt.rc('axes', titlesize=fs)     # fontsize of the axes title
plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
plt.rc('legend', fontsize=fs)    # legend fontsize
plt.rc('figure', titlesize=fs)  # fontsize of the figure title

# dark blue #0145AC
# light blue #82C7A5

# light blue #E9EDEE
# darker blue #1A9988
# orange #EB5600

c1 = '#177148'
c2 = '#42A97A'
# c3 = '#EB5600'

cs = [c1, c2]

sizes = ['3', '6', '12', '18']

data = {}
datatypes = ['5and2', 'blackandwhite']
for datatype in datatypes:
    with open('humandata/{}.json'.format(datatype), 'rb') as f:
        d = json.load(f)
        data[datatype] = {}
        for k in d:
            data[datatype][k] = {
                'mean': np.mean(d[k]),
                'error': np.std(d[k])
            }

N = len(sizes)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
ax.set_ylabel('Search Time (ms)')
ax.set_xlabel('Set size')
scene_types = {
    '5and2': 'Spatial Configuration (human)',
    'blackandwhite': 'Color Popout (human)',
}
ax.set_xticks(ind + width)
ax.set_xticklabels(sizes)
ax.set_ylim(bottom=0, top=top)

rects = []
for idx, dtype in enumerate(data):
    # d = data[size]
    d = data[dtype]
    ms = [d[a]['mean'] for a in sizes]
    es = [d[a]['error'] for a in sizes]
    print dtype, ms
    b = ax.bar(ind + (width*idx), ms, width, color=cs[idx], yerr=es, ecolor='k', edgecolor='white', linewidth='2')
    rects.append(b[0])
# rects1 = ax.bar(ind, , width)
# rects2 = ax.bar(ind + width, human_means, width, color=hcolor, yerr=human_errors, ecolor='k')
# rects3 = ax.bar(ind + width + width, human_means, width, color=hcolor, yerr=human_errors, ecolor='k')


# ax.legend((rects1[0], rects2[0]), ('Model', 'Human'), loc=2)
labels = ['Color Popout (human)', 'Spatial Configuration (human)']
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(rects, labels, loc=2)

plt.savefig('graphs/bar/human.png'.format(datatype), bbox_inches='tight')
