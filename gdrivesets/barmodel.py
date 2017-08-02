import sys
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set(style="white")

rxn_scale = 70.
rxn_yint = 558.6475

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
# use spatial #61707D

modelpalette = {
    '5and2': '#6B99C4',
    '5and2_orientation': '#45739F',
    '5and2_popout': '#2E5B87',
    'blackandwhite': '#1C436A',
    'popout': '#08233D',
    'conjunction': '#005199',
}
humanpalette = {
    '5and2': '#42A97A',
    'blackandwhite': '#177148',
    'conjunction': '#034125'
}

hcolor = '#90D7FF'
mcolor = '#1A9988'
secondarycolor = '#61707D'

sizes = ['3', '6', '12', '18']
model_means = []
model_errors = []

# if sys.argv[1] not in ['5and2', 'blackandwhite', 'conjunction']:
#     raise Exception('bad')

with open('stats2.json', 'rb') as f:
    stats = json.load(f)

def build_significances(data, p=0.01):
    r = []
    for s in sizes:
        if data[s] < p:
            r.append('*')
        else:
            r.append('n.s.')
    return r

with open('fixationjson/5and2_final.json', 'rb') as f:
    baseline_data = json.load(f)

for datatype in ['popout', '5and2_popout', '5and2_orientation']:

    sizes = ['3', '6', '12', '18']
    model_means = []
    model_errors = []
    baseline_means = []
    baseline_errors = []
    with open('fixationjson/{}_final.json'.format(datatype), 'rb') as f:
        model_data = json.load(f)

    for size in sizes:
        model_means.append(np.mean((np.array(model_data[size]) * rxn_scale) + rxn_yint))
        model_errors.append(np.std((np.array(model_data[size]) * rxn_scale) + rxn_yint))

        baseline_means.append(np.mean((np.array(baseline_data[size]) * rxn_scale) + rxn_yint))
        baseline_errors.append(np.std((np.array(baseline_data[size]) * rxn_scale) + rxn_yint))

    N = len(sizes)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    ax.set_xlabel('Set size')
    ax.set_ylim(bottom=0, top=top)
    scene_types = {
        '5and2': 'Spatial Configuration',
        'blackandwhite': 'Color Popout',
        'conjunction': 'Conjunction',
        'popout': 'Shape Popout',
        '5and2_popout': 'Orientation Popout',
        '5and2_orientation': 'Spatial Orientation',
    }
    # plt.suptitle('Spatial vs {}'.format(scene_types[datatype]))
    ax.set_xticks(ind + width)
    ax.set_xticklabels(sizes)
    ax.set_ylabel('Search Time (ms)')

    rects2 = ax.bar(ind + width, baseline_means, width, color=modelpalette['5and2'], yerr=baseline_errors, ecolor='k', edgecolor='white', linewidth='2')
    rects1 = ax.bar(ind, np.zeros(len(model_means)), width)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend((rects2[0],), ('Spatial',), loc=2)
    plt.savefig('graphs/bar/spatial.png'.format(datatype))

    rects1 = ax.bar(ind, model_means, width, color=modelpalette[datatype], yerr=model_errors, ecolor='k', edgecolor='white', linewidth='2')
    ax.legend((rects1[0], rects2[0]), (scene_types[datatype], 'Spatial'), loc=2)

    def autolabel(rects, text):
        """
        Attach a text label above each bar displaying its height
        """
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            above = 1950
            if text[idx] == '*':
                ax.text(rect.get_x() + rect.get_width(), above, text[idx], ha='center', va='bottom', fontsize=18)
            else:
                ax.text(rect.get_x() + rect.get_width(), above, text[idx], ha='center', va='bottom')
            m = 0.5
            x1 = rect.get_x() + m
            x2 = rect.get_x() + 2*rect.get_width() - m
            h = 50
            y = above - 100
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')

    autolabel(rects1, build_significances(stats[datatype]))
    # autolabel(rects2)

    # plt.show()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig('graphs/bar/{}_modelbar.png'.format(datatype), bbox_inches='tight')