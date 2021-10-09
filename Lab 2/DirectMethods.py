#!/usr/bin/env python3
import numpy as np

def Gauss(a : np.array, f : np.array):
    n = f.shape[0]
    #Прямой ход алгоритма
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            f[j] = f[j] - f[i]*(a[j][i] / a[i][i])
            a[j] = a[j] - a[i]*(a[j][i] / a[i][i])
    #Обратный ход алгоритма
    x = f
    for i in range(n - 1, -1, -1):
        for k in range(n - 1, i, -1):
            x[i] -= f[k]*a[i][k]
        x[i] /= a[i][i]
    return x


def main():
    a = np.array([[1, 1, 1], [1, -1, 1], [2, -1, 3]])
    f = np.array([6, 2, 9])
    print(Gauss(a, f))

if __name__ == '__main__':
    main()