#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Просто линейная функция
def linear(x, a, b):
        return a * x + b

# Производная вперёд, первый порядок
def der1for(f0, fplus, step):
    return (fplus - f0) / step

# Производная назад, первый порядок
def der1aft(fminus, f0, step):
    return (f0 - fminus) / step

# Первая производная во внутренней точке отрезка, второй порядок
def der1ord2(fminus, fplus, step):
    return (fplus - fminus) / (2 * step)

# Вторая производная во внутренней точке отрезка, второй порядок
def der2ord2(fminus, f0, fplus, step):
    return ((fplus + fminus) - (f0 + f0)) / (step * step)

# Проектирует функцию func на сетку x_array
def tabulate(func, x_array):
    y = []
    for i in x_array:
        y.append(func(i))
    return y

# Вычисляет методом первого порядка производную сеточной функции x_arr -> y_arr
# возвращает производную в виде вектора значений
def calculateFirstDer1(x_arr, y_arr):
         N = len(x_arr) - 1
         step =  (x_arr[N] - x_arr[0]) / N
         yder_arr  = []
         yder_arr.append(der1for(y_arr[0], y_arr[1], step))
         for i in range(1, N + 1):
             yder_arr.append(der1aft(y_arr[i - 1], y_arr[i], step))
         return yder_arr

# Вычисляет методом второго порядка производную сеточной функции x_arr -> y_arr
def calculateFirstDer2(x_arr, y_arr):
    N = len(x_arr) - 1
    step =  (x_arr[N] - x_arr[0]) / N
    yder_arr = []
    yder_arr.append((-3 * y_arr[0] + 4 * y_arr[1] - y_arr[2]) / (2 * step))
    for i in range(1, N):
        yder_arr.append(der1ord2(y_arr[i - 1], y_arr[i + 1], step))
    yder_arr.append((3 * y_arr[N] - 4 * y_arr[N - 1] + y_arr[N - 2]) / (2 * step))
    return yder_arr

# Вычисляет вторую производную сеточной функции x_arr -> y_arr
def calculateSecDer2(x_arr, y_arr):
    N = len(x_arr) - 1
    step =  (x_arr[N] - x_arr[0]) / N
    yder_arr = []
    yder_arr.append((2*y_arr[0] - 5 * y_arr[1] + 4 * y_arr[2] - y_arr[3]) / (step * step))
    for i in range(1, N):
        yder_arr.append(der2ord2(y_arr[i - 1], y_arr[i], y_arr[i + 1], step))
    yder_arr.append((2*y_arr[N] - 5 * y_arr[N - 1] + 4 * y_arr[N - 2] - y_arr[N - 3]) / (step * step))
    return yder_arr

# Для функции func на отрезке [lowbound, upbound] строит сеточную функцию с шагом 1/N
# Вычисляет производные сеточной функции всеми методами и возвращяет их в виде векторов значений
def calculateDerivatives(func, lowbound, upbound, N):
    x = np.linspace(lowbound, upbound, N + 1)
    y = tabulate(func, x)
    dy = calculateFirstDer1(x, y)
    dy2 = calculateFirstDer2(x, y)
    d2y = calculateSecDer2(x, y)
    return x, dy, dy2, d2y

# Вычисляет максимум отклонения сеточной функции x_arr -> y_arr от функции func на сетке x_arr
def calculateError(x_arr, y_arr, func):
    err = []
    for i in range(0, len(x_arr)):
        err.append(np.abs(y_arr[i] - func(x_arr[i])))
    return np.max(err)

# Строит графики ошибок для функции f на отрезке [lb, ub]
def plotErrors(f, df, ddf, lb, ub):
    # Массивы значений для построения графиков зависимости ошибок от N
    err_dy_log = []
    err_dy2_log = []
    err_d2y_log = []
    N_arr = range(10, 1000, 20)
    N_log = []

    # Для каждого N найдём отклонение каждой численно найденной производной от найденной аналитически
    for N in N_arr:
        x, dy, dy2, d2y = calculateDerivatives(f, lb, ub, N)
        err_dy_log.append(np.log10(calculateError(x, dy, df)))
        err_dy2_log.append(np.log10(calculateError(x, dy2, df)))
        err_d2y_log.append(np.log10(calculateError(x, d2y, ddf)))
        N_log.append(np.log10(1/N))
    
    # Вычислим значения коэффициента наклона прямой методом наименьших квадратов
    popt, pcov = curve_fit(linear, N_log, err_dy_log, p0 = [0, -1])
    n11 = popt[0]
    popt, pcov = curve_fit(linear, N_log, err_dy2_log, p0 = [0, -2])
    n12 = popt[0]
    popt, pcov = curve_fit(linear, N_log, err_d2y_log, p0 = [0, -2])
    n22 = popt[0]

    # Построим графики
    plt.grid(True)
    plt.ylabel('lg(max|err|)')
    plt.xlabel('lg(h)')
    plt.text(-2, -0.5, rf'$ n_{11} = {n11} $')
    plt.text(-2, -0.7, rf'$ n_{12} = {n12} $')
    plt.text(-2, -0.9, rf'$ n_{22} = {n22} $')
    plt.plot(N_log, err_dy_log, label = r'$ \frac{df}{dx}$, первый порядок')
    plt.plot(N_log, err_dy2_log, label = r'$\frac{df}{dx}$, второй порядок')
    plt.plot(N_log, err_d2y_log, label = r'$\frac{d^2f}{dx^2}$, второй порядок')
    plt.legend()
    plt.show()


def main():
    # Выберем функцию для исследования
    def f(x):
        return np.cos(x)

    def df(x):
        return -np.sin(x)
    
    def ddf(x):
        return -np.cos(x)

    lb = -4
    ub = 4
    plotErrors(f, df, ddf, lb, ub)

if __name__ == '__main__':
    main()