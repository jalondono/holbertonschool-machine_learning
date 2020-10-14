#!/usr/bin/env python3
import gym
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from tensorflow.keras import layers
import tensorflow.keras as K
from rl.processors import Processor
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from PIL import Image
import numpy as np


class AtariProcessor(Processor):
    """ preprocessing based on
        Deep Learning Quick Reference by Mike Bernico """
    def process_observation(self, observation):
        """ resizing and grayscale """
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        # resize and convert to grayscale
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """ rescale
        We could perform this processing step in `process_observation`.
        In this case, however,
        we would need to store a `float32` array instead, which is 4x
        more memory intensive than
        an `uint8` array. This matters if we store 1M observations.
        """
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """ rewards between -1 and 1 """
        return np.clip(reward, -1., 1.)


def create_q_model(num_actions, window):
    """ preprocessing based on
        Deep Learning Quick Reference by Mike Bernico """

    # Network defined by the Deepmind paper

    inputs = layers.Input(shape=(window, 84, 84))
    # comment the line below to use with GPU
    inputs_sort = layers.Permute((2, 3, 1))(inputs)

    # Convolutions on the frames on the screen
    # Change data_format="channels_first" to use GPU
    # change inputs_sort by inputs to use GPU
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu",
                           data_format="channels_last")(inputs_sort)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu",
                           data_format="channels_last")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu",
                           data_format="channels_last")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return K.Model(inputs=inputs, outputs=action)


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    num_actions = env.action_space.n
    window = 4
    model = create_q_model(num_actions, window)
    model.summary()
    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy,
                   memory=memory, processor=processor,
                   nb_steps_warmup=50000, gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])
    # training
    dqn.fit(env,
            nb_steps=17500,
            log_interval=10000,
            visualize=False,
            verbose=2)

    # save the final weights.
    dqn.save_weights('policy.h5', overwrite=True)
