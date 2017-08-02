import sys
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# sns.set(style="white")

labels = ['Color popout', 'Conjunction', 'Spatial configuration', 'Spatial orientation', 'Shape popout', 'Orientation popout']
datatypes = ['blackandwhite', 'conjunction', '5and2', '5and2_orientation', 'popout', '5and2_popout']
lines = []

compsets = {
    'high': {
        'labels': labels[:3],
        'types': datatypes[:3]
    },
    'med': {
        'labels': labels[2:4],
        'types': datatypes[2:4]
    },
    'low': {
        'labels': ['Spatial search',] + labels[-2:],
        'types': ['5and2',] + datatypes[-2:]
    },
}

solid = '-'
dashed = '--'
dotted = ':'
dashdot = '-.'

lstyles = {
    'blackandwhite': dashed,
    'conjunction': dashed,
    '5and2': solid,
    '5and2_orientation': solid,
    'popout': dashdot,
    '5and2_popout': dashdot
}

for datatype in datatypes:

    sizes = ['3', '6', '12', '18']
    linebounds = [int(a) for a in sizes]

    with open('fixationjson/{}_final.json'.format(datatype), 'rb') as f:
        data = json.load(f)

    points = []
    for size in sizes:
        for point in data[size]:
            points.append((int(size), point))

    unzipped = zip(*points)
    m, b = np.polyfit(unzipped[0], unzipped[1], 1)
    l = plt.plot(linebounds, m*np.array(linebounds) + b, linestyle=lstyles[datatype], linewidth=2)
    lines.append(l[0])

plt.suptitle('Model Performance Regression')
legend = plt.legend(lines, labels, loc=2)
plt.xlim(left=3, right=18)
plt.ylim(bottom=0, top=14)
plt.xticks([3,6,12,18])
plt.xlabel('Set size')
plt.ylabel('Fixation count')
plt.savefig('graphs/regressioncompare.png')
plt.clf()

for settype in compsets:
    for datatype in compsets[settype]['types']:
        sizes = ['3', '6', '12', '18']
        linebounds = [int(a) for a in sizes]
        print compsets[settype]
        with open('fixationjson/{}_final.json'.format(datatype), 'rb') as f:
            data = json.load(f)

        points = []
        for size in sizes:
            for point in data[size]:
                points.append((int(size), point))

        unzipped = zip(*points)
        m, b = np.polyfit(unzipped[0], unzipped[1], 1)
        l = plt.plot(linebounds, m*np.array(linebounds) + b, linewidth=2)
        lines.append(l[0])

    plt.suptitle('Model Performance Regression')
    legend = plt.legend(lines, compsets[settype]['labels'], loc=2)
    plt.xlim(left=3, right=18)
    plt.ylim(bottom=0, top=14)
    plt.xticks([3,6,12,18])
    plt.xlabel('Set size')
    plt.ylabel('Fixation count')
    plt.savefig('graphs/regression_{}.png'.format(settype))
    plt.clf()