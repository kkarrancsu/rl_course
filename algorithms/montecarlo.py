import numpy as np
from numpy.random import RandomState
import gym
from tqdm import tqdm
import itertools


def int2nparr(int_val):
    """
    Converts an integer to a 0-D numpy array.  We do this to make the environment
    compatible w/ the Spinning up repository's algorithms, which seem to return
    actions in this manner

    :param int_val: the value to be converted to a 0D numpy array
    :return: the array containing the value
    """
    return np.array(int_val)


def init_pi(obs_space, action_space, random_state_obj):
    """
    enumerate through every possible state and initialize a possible action for that state
    :return:
    Todo: Since pi will be indexed by the states and actions, don't we need more information than just the number of
        states and actions?
        Kiran: I'm not sure I understand?
    """
    obs_hi = obs_space.high[0]  # Note: expecting type gym.spaces.Box
    obs_low = obs_space.low[0]
    n = obs_space.shape[0]  # Note: expecting a 1d shape
    act_n = action_space.n  # Note: expecting type gym.spaces.Discrete

    # get all 2**n possible obs and n*(n-1)/2 actions and make a dict with them as keys
    unique_observations = list(itertools.product(range(obs_low, obs_hi+1), repeat=n))
    pi = dict()
    for unique_observation in unique_observations:
        pi[unique_observation] = random_state_obj.randint(act_n)  # initialze to a random action

    return pi


def exploring_starts(env: gym.Env, num_episodes: int = 1000, discount_factor: float = 0.1, SEED: int = 0):
    """
    Implements the Monte-Carlo Exploring Starts algorithm, given on p.99 of
    "Reinforcement Learning" - Sutton & Barto - 2nd Edition

    :param env: the Gym environment to train on
    :param num_episodes: # episodes to train
    :param SEED:  random seed value
    :return: policy function
    """
    # get size of observation-space & action space
    n_actions = env.action_space.n
    n_obs = env.observation_space.shape[0]

    # TODO: initialize properly
    rso = RandomState(SEED)

    # mapping between every possible state and an action
    pi = init_pi(env.observation_space, env.action_space, rso)
    # for q and returns, instead of storing every possible state-space as a tuple,
    # we just store the base-10 representation of the state-space
    q = np.zeros((2**n_obs, n_actions))
    returns = np.zeros((2**n_obs, n_actions))

    for ii in tqdm(range(num_episodes)):
        # reset the environment
        obs = env.reset()
        # generate an episode following pi

        # TODO: optimize this ...
        reward_list = []
        state_list = []
        action_list = []

        while True:
            # compute action based on pi
            aa = int2nparr(pi[obs])
            # step in the environment to get next state
            obs, reward, done, _ = env.step(aa)
            reward_list.append(reward)
            state_list.append(obs)
            action_list.append(aa)

            if done:
                break

        # update pi
        G = 0
        for jj in range(len(reward_list)-2, -1, -1):  # iterate through the list backwards
            G = discount_factor*G + reward_list[jj]
            St = state_list[jj]
            At = action_list[jj]
            first_visit = False
            for kk in range(jj-1, -1, -1):
                if St == state_list[kk] and At == action_list[kk]:
                    first_visit = True
                    break
            if not first_visit:
                state_base10 = int("".join(str(x) for x in St), 2)
                returns[state_base10, At] = G  # TODO: this needs to be a running average
                # TODO: update best action based on argmax condition
                best_a = 1
                pi[St] = best_a


if __name__ == '__main__':
    pass
