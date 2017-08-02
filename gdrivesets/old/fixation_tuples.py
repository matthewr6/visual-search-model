import os
import sys
import json

basename = os.path.basename(sys.argv[1]).split('.')[0]

with open(sys.argv[1], 'rb') as f:
    o_data = f.read().split('\n')[:-1]

# dict_data = {}
list_data = []
for row in o_data:
    row = row.split(' :: ')
    found = row[2] == 'True'
    # if not found:
    #     continue
    fixations = int(row[1])
    setsize = int(row[0].split('_')[0][7:])
    trialnum = int(row[0].split('_')[1][:-4])
    tup = (setsize, trialnum, fixations)
    list_data.append(tup)

with open('tuples/{}.json'.format(basename), 'wb') as f:
    json.dump(list_data, f, indent=4)