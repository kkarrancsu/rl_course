import numpy as np
from numpy.random import RandomState
import gym
from tqdm import tqdm


def int2nparr(int_val):
    """
    Converts an integer to a 0-D numpy array.  We do this to make the environment
    compatible w/ the Spinning up repository's algorithms, which seem to return
    actions in this manner

    :param int_val: the value to be converted to a 0D numpy array
    :return: the array containing the value
    """
    return np.array(int_val)


def init_pi(n_states, n_actions):
    """
    enumerate through every possible state and initialize a possible actiion for that state
    :return:
    """
    pass



def exploring_starts(env: gym.Env, num_episodes: int = 1000, SEED: int = 0):
    """
    Implements the Monte-Carlo Exploring Starts algorithm, given on p.99 of
    "Reinforcement Learning" - Sutton & Barto - 2nd Edition

    :param env: the Gym environment to train on
    :param num_episodes: # episodes to train
    :param SEED:  random seed value
    :return: policy function
    """
    # get size of observation-space & action space
    n_actions = len(env.action_space)
    n_states = len(env.observation_space)

    # TODO: initialize properly
    rso = RandomState(SEED)

    # mapping between every possible state and an action
    pi = init_pi(n_states, n_actions)
    q = np.zeros((n_states, n_actions))
    returns = np.zeros((n_states, n_actions))

    for ii in tqdm(range(num_episodes)):
        # reset the environment
        obs = env.reset()
        # generate an episode following pi
        while True:
            # TODO: compute action based on pi
            aa = int2nparr(1)
            # step in the environment to get next state
            observation_space, reward, done, _ = env.step(aa)

            # store reward and compute return

            if done:
                break


        # TODO: update pi


if __name__ == '__main__':
    pass
