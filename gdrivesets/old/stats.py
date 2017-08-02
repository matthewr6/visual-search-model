from scipy import stats
import numpy as np
import json

datatypes = ['5and2', 'conjunction', 'blackandwhite']
datatypes2 = ['5and2_orientation', 'popout', '5and2_popout']

statsdata = {}

for datatype in datatypes:
    with open('fixationjson/{}_final.json'.format(datatype), 'rb') as f:
        model_data = json.load(f)

    with open('humandata/{}.json'.format(datatype), 'rb') as f:
        human_data = json.load(f)

    sizes = [('3'), ('6'), ('12'), ('18')]

    statsdata[datatype] = {}

    print datatype
    for size in sizes:
        md = (np.array(model_data[size]) * 70.) + 558.6475
        hd = human_data[size]
        mu_m = np.mean(md)
        sig_m =  np.std(md)
        mu_h = np.mean(hd)
        sig_h = np.std(hd)
        print '{} & {} & {} & {} & {}\\\\ \\hline'.format(size, round(mu_m, 2), round(sig_m, 2), round(mu_h, 2), round(sig_h, 2))
        statsdata[datatype][size] = {
            'mu_m': mu_m,
            'sig_m': sig_m,
            'mu_h': mu_h,
            'sig_h': sig_h
        }
    print ''

for datatype in datatypes2:
    with open('fixationjson/{}_final.json'.format(datatype), 'rb') as f:
        model_data = json.load(f)

    sizes = [('3'), ('6'), ('12'), ('18')]

    statsdata[datatype] = {}

    print datatype
    for size in sizes:
        md = (np.array(model_data[size]) * 70.) + 558.6475
        mu_m = np.mean(md)
        sig_m =  np.std(md)
        print '{} & {} & {}\\\\ \\hline'.format(size, round(mu_m, 2), round(sig_m, 2))
        statsdata[datatype][size] = {
            'mu_m': mu_m,
            'sig_m': sig_m,
            'mu_h': mu_h,
            'sig_h': sig_h
        }
    print ''

# with open('stats.json', 'wb') as f:
    # json.dump(statsdata, f, indent=4)