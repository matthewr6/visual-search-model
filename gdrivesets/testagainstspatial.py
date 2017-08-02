from scipy import stats
import numpy as np
import json

datatypes = ['popout', '5and2_popout', '5and2_orientation']

statsdata = {}

def zscore(x, mu, sigma):
    return (x - mu)/float(sigma)

def twoz(d1, d2):
    num = (np.mean(d1) - np.mean(d2))
    denom = np.sqrt((np.var(d1)/len(d1)) + (np.var(d2)/len(d2)))
    print num, denom
    return num/denom

slope = 70.
yint = 558.6475

with open('fixationjson/5and2_final.json', 'rb') as f:
    base_data = json.load(f)

for datatype in datatypes:
    with open('fixationjson/{}_final.json'.format(datatype), 'rb') as f:
        model_data = json.load(f)

    sizes = [('3'), ('6'), ('12'), ('18')]

    statsdata[datatype] = {}

    for size in sizes:
        statsdata[datatype][size] = stats.ttest_ind(model_data[size], base_data[size], equal_var=False)[1]

with open('stats2.json', 'wb') as f:
    json.dump(statsdata, f, indent=4)