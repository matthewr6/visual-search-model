import numpy as np
import math

# def get_relatived_value(arr, rel_idx):
#     forward_bias, idx = math.modf(rel_idx)
#     if idx+1 == len(arr):
#         return arr[-1]
#     first_bias = 1 - forward_bias
#     v1 = np.array(arr[int(idx)]) * first_bias
#     v2 = np.array(arr[int(idx)+1]) * forward_bias
#     print v1 + v2
#     return (v1 + v2).tolist()

# def rel_2d_value(mat, rel_x, rel_y):
#     return get_relatived_value(get_relatived_value(mat, rel_x), rel_y)

# mat = [
#     [1, 2, 3],
#     [2, 3, 4],
#     [3, 4, 5],
# ]
# print rel_2d_value(mat, 0.5, 1.5)

def corresponding_points(ox, oy, stride, size):
    # x values begin at z + xw and go from xs to xs+s
    # y values begin at z + yw and go from ys to ys+s
    # z = 0, s = stride, w = int(np.round(stride/4.0))?
    points = []
    # w = int(np.round(stride/4.0))
    w = 0
    x = ox * (w + stride)
    for i in range(stride):
        y = oy * (w + stride)
        for j in range(stride):
            if x < size and y < size:
                points.append((x, y))
            y += 1
        x += 1
    return points

print corresponding_points(5, 5, 8, 256)