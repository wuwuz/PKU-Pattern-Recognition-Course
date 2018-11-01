import numpy as np
import matplotlib.pyplot as plt

def parzen_estimation(X):

    print('parzen window = ? ')
    window_size = np.float32(input())

    x_axis = []
    y_axis = []

    le = 0
    ri = 0

    for i in range(0, 200000):
        cur = np.float32(i)
        x_axis.append(np.float32(i))
        while (le <= n) and (cur - X[le] > window_size * 0.5) : le += 1
        while (ri <= n) and (X[ri] - cur <= window_size * 0.5) : ri += 1
        sum = ri - le
        y_axis.append(np.float32(sum * 1.0 / window_size / n))

    plt.plot(x_axis, y_axis)
    plt.xlabel('window size = ' + str(window_size))
    plt.show()

def kn_estimation(X):

    print('k = ?')
    k = int(input())

    x_axis = []
    y_axis = []

    le = 0
    ri = k

    for i in range(0, 200000):
        cur = np.float32(i)
        x_axis.append(np.float32(i))
        while (le < n) and (abs(cur - X[le]) > abs(cur - X[ri])) :
            le += 1
            ri += 1
        sum = ri - le
        window_size = X[ri - 1] - X[le]
        y_axis.append(np.float32(sum * 1.0 / window_size / n))

    plt.plot(x_axis, y_axis)
    plt.xlabel('k = ' + str(k))
    plt.show()

input_file = open('A.txt', 'r', encoding='utf-8')

X = []
n = 0
for line in input_file.readlines():
    cur_line = line.strip()
    x = float(cur_line)
    X.append(x)
    n += 1

X.sort()
#print(X[0])
#print(X[-1])

parzen_estimation(X)
kn_estimation(X)

exit(0)

