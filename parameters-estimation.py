import numpy as np
import matplotlib.pyplot as plt

input = open('A.txt', 'r', encoding='utf-8')

X = []
n = 0

for line in input.readlines():
    cur_line = line.strip()
    X.append(float(x))
    n += 1

X.sort()
print(X[0])
print(X[-1])

exit(0)

x = np.arange(0.1, 200000)
y = np.exp(-np.power(np.log(x)-theta, 2) / 2.0 / sigma / sigma) / sigma / x / np.sqrt(2 * np.pi)

plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.ylabel('p(x)')

t = np.arange(-70000, 130000)
z = np.exp(-np.power(t - avg, 2) / 2.0 / var / var) / np.sqrt(2 * np.pi) / var
plt.subplot(2, 1, 2)
plt.plot(t, z)
plt.ylabel('q(x)')

plt.xlabel('x')
plt.show()


