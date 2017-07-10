import matplotlib.pyplot as plt
import numpy as np

a = [(3, True), (4, True), (2, True), (2, True), (5, False), (5, False), (5, False), (4, True), (2, True)]
b =  [(5, False), (1, True), (5, True), (4, True), (5, False), (5, False), (1, True), (2, True), (3, True)]
c = [(1, True), (4, True), (1, True), (1, True), (2, True), (1, True), (3, True), (5, False), (5, True)]
d = [(2, True), (2, True), (4, True), (5, True), (2, True), (5, False), (4, True), (5, True), (2, True)]
e = [(5, False), (5, True), (3, True), (2, True), (2, True), (5, False), (2, True), (1, True), (5, True)]

data = a + b + c + d + e

# data = [d[0] for d in data]
data = [d[0] for d in data if d[1]]

data.sort()

x_values = np.unique(data)

d2 = []
for x in x_values:
    d2.append(data.count(x))

print d2

total = float(sum(d2))

graph_data = []
curtotal = 0
for d in d2:
    curtotal += d/total
    graph_data.append(curtotal)

plt.plot(np.arange(len(graph_data)), graph_data)
plt.scatter(np.arange(len(graph_data)), graph_data)
plt.show()