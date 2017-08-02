import sys
import json

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

datatypes = ['blackandwhite', 'conjunction', '5and2', '5and2_orientation', 'popout', '5and2_popout']
labels = ['Feature', 'Conjunction', 'Spatial', 'Spatial orientation', 'Popout', 'Spatial popout']

for idx, datatype in enumerate(datatypes):

    sizes = ['3', '6', '12', '18']

    with open('fixationjson/{}_final.json'.format(datatype), 'rb') as f:
        data = json.load(f)

    points = []
    for size in sizes:
        for point in data[size]:
            points.append((int(size), point))
    # slope, intercept, r_value, p_value, std_err
    m, b, r, p, e = stats.linregress(*zip(*points))

    print '{} & ${}$ & ${}$\\\\ \\hline'.format(labels[idx], round(m, 4), round(e, 4))
    print p
    print ''