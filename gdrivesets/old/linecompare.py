import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt

# if sys.argv[1] not in ['5and2', 'blackandwhite', 'conjunction']:
#     raise Exception('bad')

for datatype in ['5and2', 'blackandwhite', 'conjunction']:
    with open('fixationjson/{}_final.json'.format(datatype), 'rb') as f:
        model_data = json.load(f)

    with open('humandata/{}.json'.format(datatype), 'rb') as f:
        human_data = json.load(f)

    sizes = [('3', (0, 0)), ('6', (0, 1)), ('12', (1, 0)), ('18', (1, 1))]

    def hpseudocdf(arr):
        return [(v, float(idx)/len(arr)) for idx, v in enumerate(arr)]

    def mpseudocdf(arr):
        ret = [(0, 0)]
        for idx, v in enumerate(arr):
            if len(ret) and ret[-1][0] == v * 250.0:
                ret[-1] = (v*250.0, float(idx)/len(arr))
            else:
                ret.append((v*250.0, float(idx)/len(arr)))
        return ret

    fig, ax = plt.subplots(nrows=2, ncols=2)

    for size, pos in sizes:
        d1 = model_data[size]
        d1.sort()
        d2 = human_data[size]
        d2.sort()
        m, = ax[pos[0], pos[1]].plot(*zip(*mpseudocdf(d1)), color='#0145AC')
        h, = ax[pos[0], pos[1]].plot(*zip(*hpseudocdf(d2)), color='#82C7A5')
        ax[pos[0], pos[1]].title.set_text('Set size {}'.format(size))
        ax[pos[0], pos[1]].set_xlim(left=0, right=5000)
        ax[pos[0], pos[1]].set_ylim(bottom=0, top=1)
        ax[pos[0], pos[1]].set_xlabel('Search Time (ms)')
        ax[pos[0], pos[1]].set_ylabel('Percent')
        fig.legend((m, h), ('Model', 'Human'), 'lower center')

    plt.suptitle('Fixation Time CDFs')

    plt.subplots_adjust(hspace=0.5, bottom=0.2, wspace=0.25)

    plt.savefig('graphs/{}_cdf_compare.png'.format(datatype))