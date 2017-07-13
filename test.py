import numpy as np
# import math
# import cPickle
# import json
from scipy import stats

# with open('S3prots.dat', 'rb') as f:
#     s3prots = cPickle.load(f)[:-1]

# for idx, thing in enumerate(s3prots):
#     # thing is 43 x 600
#     s3prots[idx] = np.mean(thing)

# print s3prots

# import matplotlib.pyplot as plt

# plt.plot(np.arange(len(s3prots)), s3prots)
# plt.show()

# import time
# i = 1
# with open('test.txt', 'ab') as f:
#     while True:
#         f.write(str(i))
#         i += 1
#         time.sleep(1)

sigma = 15
def gauss_2d(focus_x, focus_y):
    # inverTED vs inverSE?  inverTED may not be the same as inverSE
    dims = [256, 256]
    grid = np.empty(dims)
    print 'a'
    for i in xrange(dims[0]):
        for j in xrange(dims[1]):
            grid[i, j] = stats.norm.pdf(i, focus_y, sigma) * stats.norm.pdf(j, focus_x, sigma)
    print 'b'
    return grid

# o = gauss_2d(25, 25)
# result = vectest([[7,2,3],[4,5,6],[7,8,9]])
# print result

# grid = np.indices((10,10)).T.swapaxes(0, 1)

def fast_gauss_2d(focus_x, focus_y):
    grid_y, grid_x = np.mgrid[:256, :256]
    return stats.norm.pdf(grid_x, focus_x, sigma) * stats.norm.pdf(grid_y, focus_y, sigma)
# print grid_a * grid_b

# print gauss_2d(2, 2)