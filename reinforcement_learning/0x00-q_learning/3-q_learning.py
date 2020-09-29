#!/usr/bin/env python3
""" Q-learning"""
import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    """
    uses epsilon-greedy to determine the next action
    :param Q: numpy.ndarray containing the q-table
    :param state: current state
    :param epsilon: epsilon to use for the calculation
    :return: next action index
    """
    pos_action = Q.shape[1]

    if np.random.uniform(0, 1) < epsilon:
        # Explore: select a random action
        action_index = np.random.randint(pos_action)
    else:
        # Exploit: select the action with max value (future reward)
        action_index = np.argmax(Q[state])

    return action_index


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    """
    performs Q-learning:
    :param env: is the FrozenLakeEnv instance
    :param Q: is a numpy.ndarray containing the Q-table
    :param episodes: is the total number of episodes to train over
    :param max_steps: is the maximum number of steps per episode
    :param alpha: is the learning rate
    :param gamma: is the discount rate
    :param epsilon: is the initial threshold for epsilon greedy
    :param min_epsilon: is the minimum value that epsilon should decay to
    :param epsilon_decay: is the decay rate for updating
     epsilon between episodes
    :return: Q, total_rewards
    * When the agent falls in a hole, the reward should
     be updated to be -1
    * Q is the updated Q-table
    * total_rewards is a list containing the rewards per episode
    """

    initial_epsilon = epsilon
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        reward_sum = 0

        for step in range(max_steps):
            # epsilon_greedy
            action = epsilon_greedy(Q, state, epsilon)

            # perform step (take new action)
            new_state, reward, done, info = env.step(action)

            if done and reward == 0:
                reward = -1

            # update Q-table for (state, action)
            Q[state, action] = Q[state, action] + alpha * \
                               (reward + gamma * np.max(Q[new_state, :]) -
                                Q[state, action])

            state = new_state
            reward_sum += reward

            if done:
                break

        total_rewards.append(reward_sum)

        # Exploration rate decay
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q, total_rewards
