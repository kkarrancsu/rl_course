#!/usr/bin/env python

import numpy as np
from numpy.random import RandomState


class UniformRandomOpponent:
    def __init__(self, num_actions, seed):
        self.num_actions = num_actions
        self.random_state = RandomState(seed)

    def act(self):
        return self.random_state.randint(self.num_actions)
