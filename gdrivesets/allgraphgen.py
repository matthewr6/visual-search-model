import sys
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set(style="white")

rxn_scale = 70.
rxn_yint = 558.6475

linebounds = [3, 18]

top = 2500

fs = 18

plt.rc('font', size=fs)          # controls default text sizes
plt.rc('axes', titlesize=fs)     # fontsize of the axes title
plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
plt.rc('legend', fontsize=fs)    # legend fontsize
plt.rc('figure', titlesize=fs)  # fontsize of the figure title

modeltypes = ['5and2', '5and2_orientation', '5and2_popout', 'blackandwhite', 'popout']
humantypes = ['5and2', 'blackandwhite']
names = {
    '5and2': 'Spatial Configuration',
    '5and2_orientation': 'Spatial Orientation',
    '5and2_popout': 'Orientation popout',
    'blackandwhite': 'Color popout',
    'popout': 'Shape popout'
}
mdata = {}
for mt in modeltypes:
    with open('fixationjson/{}_final.json'.format(mt), 'rb') as f:
        mdata[mt] = json.load(f)
hdata = {}
for ht in humantypes:
    with open('humandata/{}.json'.format(ht), 'rb') as f:
        hdata[ht] = json.load(f)
humanspatial = []
for size in hdata['5and2']:
    for v in hdata['5and2'][size]:
        humanspatial.append((int(size), v))
modelpalette = {
    '5and2': ['#6B99C4', 'white'],
    '5and2_orientation': ['#45739F','white'],
    '5and2_popout': ['#2E5B87','white'],
    'blackandwhite': ['#1C436A','white'],
    'popout': ['#08233D','white']
}
humanpalette = {
    '5and2': ['#42A97A','white'],
    'blackandwhite': ['#177148','white']
}

graphs = [
    ('m5and2', 'h5and2'),
    ('m5and2_popout', 'mpopout'),
    ('mblackandwhite', 'hblackandwhite'),
    ('m5and2_orientation', 'm5and2')
    # ('m5and2_orientation',)
]
with_humanslope = [1, 3]
sizes = ['3', '6', '12', '18']
N = len(sizes)
ind = np.arange(N)

def build_graphable(m_or_h, data):
    means = []
    errors = []
    for size in sizes:
        if m_or_h == 'm':
            means.append(np.mean((np.array(data[size]) * rxn_scale) + rxn_yint))
            errors.append(np.std((np.array(data[size]) * rxn_scale) + rxn_yint))
        else:
            means.append(np.mean(data[size]))
            errors.append(np.std(data[size]))
    return means, errors

def build_legend(ns):
    l = []
    for n in ns:
        mh = n[0]
        n = n[1:]
        new = names[n]
        if mh == 'h':
            new += ' (human)'
        else:
            new += ' (model)'
        l.append(new)
    return l

for gidx, graphset in enumerate(graphs):
    print graphset
    fig, ax = plt.subplots()
    ax.set_ylabel('Search Time (ms)')
    ax.set_xlabel('Set size')
    if len(graphset) == 2:
        width = 0.35
        ax.set_xticks(ind + width)
    elif len(graphset) == 3:
        width = 0.25
        ax.set_xticks(ind + (width*1.5))
    else:
        width = 0.45
        ax.set_xticks(ind + width/2.0)
    ax.set_xticklabels(sizes)
    ax.set_ylim(bottom=0, top=top)
    bars = []
    for idx, types in enumerate(graphset):
        m_or_h = types[0]
        t = types[1:]
        if m_or_h == 'm':
            d = mdata[t]
        else:
            d = hdata[t]
        means, errors = build_graphable(m_or_h, d)
        if m_or_h == 'm':
            color = modelpalette[t][0]
        else:
            color = humanpalette[t][0]
        print color
        bar = ax.bar(ind + (width*idx), means, width, color=color, yerr=errors, ecolor='k', edgecolor='white', linewidth='2')
        bars.append(bar)
    # if gidx in with_humanslope:
    #     uz = zip(*humanspatial)
    #     m, b = np.polyfit(uz[0], uz[1], 1)
    #     line = ax.plot(linebounds, m*np.array(linebounds) + b)
    ax.legend(bars, build_legend(graphset), loc=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tick_params(axis='both', top='off', right='off')
    plt.savefig('graphs/bars2/{}.png'.format(gidx), bbox_inches='tight')