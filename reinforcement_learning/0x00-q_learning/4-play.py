#!/usr/bin/env python3
"""contains the play function"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    has the trained agent play an episode
    :param env: the FrozenLakeEnv instance
    :param Q: numpy.ndarray containing the Q-table
    :param max_steps: maximum number of steps in the episode
    :return: the total rewards for the episode
    """
    state = env.reset()
    env.render()
    for step in range(max_steps):
        # Choose action with highest Q-value for current state
        action = np.argmax(Q[state, :])

        # perform step (take new action)
        state, reward, done, info = env.step(action)

        # render current state of environment
        env.render()

        if done:
            break

    return reward
