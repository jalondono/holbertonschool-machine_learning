#!/usr/bin/env python3

import numpy as np
expectation = __import__('6-expectation').expectation

if __name__ == "__main__":
    X = np.random.randn(100, 6)
    m = np.random.randn(5, 6)
    S = np.random.randn(5, 6, 6)
    print(expectation(X, 'hello', m, S))
    print(expectation(X, np.array([[1, 2, 3, 4, 5]]), m, S))
    print(expectation(X, np.random.randn(5), m, S))
