#!/usr/bin/env python3

import numpy as np
initialize = __import__('0-initialize').initialize

if __name__ == "__main__":
    X = np.random.rand(100, 3)
    print(initialize(X, '2'))
    print(initialize(X, 0))
    print(initialize(X, -5))
