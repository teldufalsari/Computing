#!/usr/bin/env python3
import numpy as np

def Gauss(a : np.array, f : np.array):
    if np.linalg.det(a) == 0:
        raise ValueError('The coefficient matrix is singular')
    n = f.shape[0]
    #Прямой ход алгоритма
    for i in range(0, n - 1):
        #Если попался нулевой ведущий элемент, его нужно заменить
        #Ненулевой ведущий должен найтись, так как определитель не нулевой
        if a[i][i] == 0:
            for k in range(i + 1, n-1):
                if a[k][i] != 0:
                    a[[i, k]] = a[[k, i]]
                    f[[i, k]] = f[[k, i]]
                    break
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

def GaussSel(a : np.array, f : np.array):
    if np.linalg.det(a) == 0:
        raise ValueError('The coefficient matrix is singular')
    n = f.shape[0]
    #Прямой ход алгоритма
    for i in range(0, n):
        #Выбираем наибольший элемент текущего столбца
        k = np.argmax(abs(a[i:,i])) + i #Лютый костыль конечно, но что поделать
        a[[k, i]] = a [[i, k]]
        f[[k, i]] = f [[i, k]]
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
    a = np.array([[1.0, 1, 1], [1, -1, 1], [2, -1, 3]])
    f = np.array([6.0, 2, 9])
    print(GaussSel(a, f))

if __name__ == '__main__':
    main()