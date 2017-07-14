import os
import sys
import json

basename = os.path.basename(sys.argv[1]).split('.')[0]

with open(sys.argv[1], 'rb') as f:
    o_data = f.read().split('\n')[:-1]

dict_data = {}
list_data = []
for row in o_data:
    row = row.split(' :: ')
    list_data.append(int(row[1]))

with open('fixationjson/{}.json'.format(basename), 'wb') as f:
    json.dump(list_data, f, indent=4)