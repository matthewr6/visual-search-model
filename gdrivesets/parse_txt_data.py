import sys

# the 5s and 2s are 14 by 16... so give radius of 25?

with open(sys.argv[1], 'rb') as f:
    data = f.read().split('\n')[:-1]

# trial, setsize, x, y
# setsize<x>_<y>.png
keep = [1,3,5,6]
floats = [2,3]
nameformat = 'setsize{}_{}.png'
final = {}
for idx, row in enumerate(data):
    split = row.split(' ')
    final[nameformat.format(split[3], split[1])] = (float(split[5]), float(split[6]))

# print final.keys()
print final[nameformat.format(12, 1)]