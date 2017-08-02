import json
import numpy as np
from scipy import stats

comparisons = [
    ['5and2', 'blackandwhite'],
    ['conjunction', 'blackandwhite'],
    ['5and2', 'conjunction'],

    ['5and2', '5and2_orientation'],
    ['5and2', '5and2_popout'],
    ['5and2', 'popout'],
    ['5and2_popout', 'popout'],
]

def zscore(s1, e1, s2, e2):
    return (s1 - s2)/np.sqrt(((e1 * s1)**2) + ((e2 * s2)**2))

sizes = ['3', '6', '12', '18']
def regression(data):
    d = []
    for s in sizes:
        for t in data[s]:
            d.append((int(s), t))
    m, b, r, p, e = stats.linregress(*zip(*d))
    return (m, e)

def pvalue(z, two_sided=True):
    p = stats.norm.sf(abs(z))
    if two_sided:
        p *= 2.0
    return p

# pretty sure this is the wrong test lol
# since I have two sets I want to compare and four means in each set...

for c in comparisons:
    print c
    with open('fixationjson/{}_final.json'.format(c[0]), 'rb') as f:
        d1 = json.load(f)

    with open('fixationjson/{}_final.json'.format(c[1]), 'rb') as f:
        d2 = json.load(f)

    # for k in d1:
        # m, b, r, p, e = stats.linregress(d1[k], d2[k])
    m1, e1 = regression(d1)
    m2, e2 = regression(d2)

    # print m1, m2
    # print e1, e2

    print pvalue(zscore(m1, e1, m2, e2))
    print ''
        # print k, stats.ttest_ind(d1[k], d2[k])[1]