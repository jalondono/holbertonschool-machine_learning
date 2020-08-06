#!/usr/bin/env python3

import numpy as np
expectation = __import__('6-expectation').expectation

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi = np.array([0.75, 0.075, 0.075, 0.1])
    m = np.array([[28, 42], [4, 23], [65, 31], [21, 65]])
    S = np.array([[[70, 5], [5, 70]], [[15, 10], [10, 15]], [[15, 0], [0, 15]], [[30, 10], [10, 30]]])
    g, l = expectation(X, pi, m, S)
    print(g)
    print(g.shape)
    print(l)
