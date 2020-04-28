#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

# y-axis in bold
rc('font', weight='bold')

# The position of the bars on the x-axis
r = [0, 1, 2]

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
bottom1 = (fruit[0] + fruit[1])
bottom2 = (fruit[0] + fruit[1] + fruit[2])

# Names of group and bar width
names = ['Farrah', 'Fred', 'Felicia']
barWidth = 0.5

# Colors
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']


# Create brown bars
plt.bar(r,
        fruit[0],
        color=colors[0],
        edgecolor='white',
        width=barWidth,
        label='apples')
# Create green bars (middle), on top of the firs ones
plt.bar(r,
        fruit[1],
        bottom=fruit[0],
        color=colors[1],
        edgecolor='white',
        width=barWidth,
        label='bananas')
# Create green bars (top)
plt.bar(r,
        fruit[2],
        bottom=bottom1,
        color=colors[2],
        edgecolor='white',
        width=barWidth,
        label='oranges')

plt.bar(r,
        fruit[3],
        bottom=bottom2,
        color=colors[3],
        edgecolor='white',
        width=barWidth,
        label='peaches')

# Custom X axis
plt.ylim(0, 80)
plt.xticks(r, names, fontweight='bold')
plt.legend()

# Show graphic
plt.show()
