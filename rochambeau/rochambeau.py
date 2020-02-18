#!/usr/bin/env python3

import gym


ACTIONS = ['rock', 'paper', 'scissors']


def payoff(action1, action2):
    reward = 0, 0
    if action1 == 'rock':
        if action2 == 'rock':
            reward = 0, 0
        elif action2 == 'paper':
            reward = -1, 1
        elif action2 == 'scissors':
            reward = 1, -1
    elif action1 == 'paper':
        if action2 == 'rock':
            reward = 1, -1
        elif action2 == 'paper':
            reward = 0, 0
        elif action2 == 'scissors':
            reward = -1, 1
    elif action1 == 'scissors':
        if action2 == 'rock':
            reward = -1, 1
        elif action2 == 'paper':
            reward = 1, 1
        elif action2 == 'scissors':
            reward = 0, 0

    return reward


class Rochambeau(gym.Env):
    def __init__(self, opponent):
        self.opponent = opponent

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    pass
