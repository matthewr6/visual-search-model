from scipy import stats
import numpy as np
import json

datatypes = ['5and2', 'conjunction', 'blackandwhite']

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

for datatype in datatypes:
    with open('fixationjson/{}_final.json'.format(datatype), 'rb') as f:
        model_data = json.load(f)

    with open('humandata/{}.json'.format(datatype), 'rb') as f:
        human_data = json.load(f)

    sizes = [('3'), ('6'), ('12'), ('18')]

    statsdata[datatype] = {}

    for size in sizes:
        statsdata[datatype][size] = stats.ttest_ind((np.array(model_data[size])*slope) + yint, human_data[size], equal_var=False)[1]

with open('stats.json', 'wb') as f:
    json.dump(statsdata, f, indent=4)