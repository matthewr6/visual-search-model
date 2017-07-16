import os
import sys
import json

basename = os.path.basename(sys.argv[1]).split('.')[0]

with open(sys.argv[1], 'rb') as f:
    o_data = f.read().split('\n')[:-1]

dict_data = {}
# list_data = []
for row in o_data:
    row = row.split(' :: ')
    fixations = int(row[1])
    setsize = row[0].split('_')[0][7:]
    if setsize not in dict_data:
        dict_data[setsize] = []
    dict_data[setsize].append(fixations)
    # list_data.append(int(row[1]))

with open('fixationjson/{}.json'.format(basename), 'wb') as f:
    json.dump(dict_data, f, indent=4)