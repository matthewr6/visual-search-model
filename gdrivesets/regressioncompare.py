import sys
import json

import numpy as np
import matplotlib.pyplot as plt

# dark blue #0145AC
# light blue #82C7A5

sizes = ['3', '6', '12', '18']
model_means = []
model_errors = []
human_means = []
human_errors = []

# if sys.argv[1] not in ['5and2', 'blackandwhite', 'conjunction']:
#     raise Exception('bad')

for datatype in ['5and2', 'blackandwhite', 'conjunction']:

    sizes = ['3', '6', '12', '18']
    linebounds = [int(a) for a in sizes]

    with open('fixationjson/{}_final.json'.format(datatype), 'rb') as f:
        model_data = json.load(f)

    with open('humandata/{}.json'.format(datatype), 'rb') as f:
        human_data = json.load(f)

    model_points = []
    human_points = []
    for size in sizes:
        for point in model_data[size]:
            p = point * 250.0
            model_points.append((int(size), p))
        for point in human_data[size]:
            human_points.append((int(size), point))

    fig, ax = plt.subplots()
    unzipped = zip(*model_points)
    # scatter1 = ax.scatter(*unzipped)
    m, b = np.polyfit(unzipped[0], unzipped[1], 1)
    plt.plot(linebounds, m*np.array(linebounds) + b)

    unzipped = zip(*human_points)
    m, b = np.polyfit(unzipped[0], unzipped[1], 1)
    plt.plot(linebounds, m*np.array(linebounds) + b)

    # scatter2 = ax.scatter(*zip(*human_points))

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Search Time (ms)')
    ax.set_xlabel('Set size')
    scene_types = {
        '5and2': 'Spatial Configuration',
        'blackandwhite': 'Color Popout',
        'conjunction': 'Conjunction',
    }
    plt.suptitle('Model vs Human {}'.format(scene_types[datatype]))
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(sizes)

    # ax.legend((scatter1, scatter2), ('Model', 'Human'), loc=2)

    ax.set_ylim(bottom=0, top=5000)

    # autolabel(rects1)
    # autolabel(rects2)

    # plt.show()
    plt.savefig('graphs/{}_regression.png'.format(datatype))