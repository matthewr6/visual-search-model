import os
import sys
import json

desired_columns = [3, 7]

basename = os.path.basename(sys.argv[1]).split('.')[0]
with open(sys.argv[1], 'rb') as f:
    data = f.read().split('\n')[1:-1]
    # data = f.read().split('\t\t')[1:-1]

for idx, d in enumerate(data):
    # data[idx] = d.split('\t')
    data[idx] = d.split(',')

parsed_data = {}

for item in data:
    if item[3] not in parsed_data:
        parsed_data[item[3]] = []
    parsed_data[item[3]].append(int(item[7]))

with open('humandata/{}.json'.format(basename), 'wb') as f:
    json.dump(parsed_data, f, indent=4)