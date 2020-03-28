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


def exploring_starts(env: gym.Env, num_episodes: int = 1000, max_episode_len=100,
                     discount_factor: float = 0.9, eps: float = 0.1, SEED: int = 0):
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
    pi = np.zeros(2 ** n_obs)
    # for q and returns_count, instead of storing every possible state-space as a tuple,
    # we just store the base-10 representation of the state-space
    q = np.zeros((2 ** n_obs, n_actions))
    returns_count = np.zeros((2 ** n_obs, n_actions))

    for ii in tqdm(range(num_episodes)):
        # reset the environment
        obs = env.reset()
        # generate an episode following pi

        # TODO: optimize this ...
        reward_list = []
        state_list = []
        action_list = []

        t = 0
        while True and t < max_episode_len:
            obs_base10 = int("".join(str(x) for x in obs), 2)

            # compute action based on pi, epsilon greedy
            aa = int(rso.randint(n_actions) if rso.uniform() < eps else pi[obs_base10])

            # step in the environment to get next state
            obs, reward, done, _ = env.step(aa)
            obs_base10 = int("".join(str(x) for x in obs), 2)

            reward_list.append(reward)
            state_list.append(obs_base10)
            action_list.append(aa)
            t += 1

            if done:
                break

        # update pi
        G = 0
        for jj in range(len(reward_list) - 2, -1, -1):  # iterate through the list backwards
            G = discount_factor * G + reward_list[jj + 1]
            St = state_list[jj]
            At = action_list[jj]
            first_visit = True
            for kk in range(jj - 1, -1, -1):
                if St == state_list[kk] and At == action_list[kk]:
                    first_visit = False
                    break
            if first_visit:
                returns_count[St, At] += 1
                q[St, At] = q[St, At] + (1 / returns_count[St, At]) * (G - q[St, At])
                best_a = np.argmax(q[St, :])
                pi[St] = best_a

    return pi
