#!/usr/bin/env python3
import numpy as np

def Jacobi(a : np.array, f : np.array, epsilon) :
    n = f.shape[0]
    if a.shape[0] != n or a.shape[1] != n:
        raise RuntimeError('Matrix and/or vector dimensions do not correspond')
    S = sum(sum(abs(a))) - sum(abs(np.diag(a)));
    for d in np.diag(a):
        if abs(d) < S:
            raise ValueError('No diagonal domination, iterations would not converge')

    x = np.ones(n)
    xs = np.zeros(n)
    delta = np.copy(x)
    while (max(delta) > epsilon):
        for i in range(0, n):
            j = range(0, i)
            k = range(i + 1, n)
            xs[i] = (f[i] / a[i][i]) - sum(a[i][j] * x[j] / a[i][i]) - sum(a[i][k] * x[k] / a[i][i])
            delta = abs(x - xs)
            x = np.copy(xs)
            print(x)
            print(max(delta))
    return x

def main():
    epsilon = 10e-4
    a = np.array([[9.0, 1, 1], [1, -12, 1], [2, -1, 64]])
    f = np.array([6.0, 2, 9])
    print(Jacobi(a, f, epsilon))

if __name__ == '__main__':
    main()