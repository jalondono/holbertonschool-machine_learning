#!/usr/bin/env python3

if __name__ == '__main__':
    minor = __import__('1-minor').minor

    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]

    print(minor(mat4))
