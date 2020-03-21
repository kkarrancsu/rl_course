import numpy as np
from numpy.random import RandomState
import gym
from tqdm import tqdm
import itertools


def qlearning(env: gym.Env, num_episodes: int = 1000, max_episode_len=100,
              discount_factor: float = 0.9, eps: float = 0.1, SEED: int = 0):
    # get size of observation-space & action space
    n_actions = env.action_space.n
    n_obs = env.observation_space.shape[0]

    # TODO: initialize properly
    rso = RandomState(SEED)

    Q = np.zeros((2 ** n_obs, n_actions))  # initialize the Q table
    for ii in tqdm(range(num_episodes)):
        # reset the environment
        obs = env.reset()
        obs_base10 = int("".join(str(x) for x in obs), 2)

        t = 0
        done = False
        while not done and t < max_episode_len:
            if rso.uniform() < eps:
                # choose random action
                aa = rso.randint(n_actions)
            else:
                # choose best action according to Q
                aa = np.argmax(Q[obs_base10, :])

                # if Q values for a state are equal, choose randomly from the best
                if len(np.where(Q[obs_base10, :] == aa)[0]) > 1:
                    aa = rso.choice(np.where(Q[obs_base10, :] == aa)[0])

            # step in the environment to get next state
            obs, reward, done, _ = env.step(aa)
            obs_base10 = int("".join(str(x) for x in obs), 2)

            # update table entry for Q(s,a)
            Q[obs_base10, aa] = reward + discount_factor * np.max(Q[obs_base10, :])

            t += 1

    return Q


def sarsa(env: gym.Env, num_episodes: int = 1000, max_episode_len=100,
          discount_factor: float = 0.9, eps: float = 0.1, alpha=0.1, SEED: int = 0):
    # get size of observation-space & action space
    n_actions = env.action_space.n
    n_obs = env.observation_space.shape[0]

    # TODO: initialize properly
    rso = RandomState(SEED)

    Q = np.zeros((2 ** n_obs, n_actions))  # initialize the Q table
    for ii in tqdm(range(num_episodes)):
        # reset the environment
        obs = env.reset()
        obs_base10 = int("".join(str(x) for x in obs), 2)

        t = 0
        done = False
        while not done and t < max_episode_len:
            aa = rso.randint(n_actions) if rso.uniform() < eps else np.argmax(Q[obs_base10, :])

            # if Q values for a state are equal, choose randomly from the best
            if len(np.where(Q[obs_base10, :] == aa)[0]) > 1:
                aa = rso.choice(np.where(Q[obs_base10, :] == aa)[0])

            # step in the environment to get next state
            obs_tp1, reward_tp1, done, _ = env.step(aa)
            obs_tp1_base10 = int("".join(str(x) for x in obs), 2)

            # choose best action according to Q
            aa_tp1 = rso.randint(n_actions) if rso.uniform() < eps else np.argmax(Q[obs_tp1_base10, :])

            # if Q values for a state are equal, choose randomly from the best
            if len(np.where(Q[obs_tp1_base10, :] == aa_tp1)[0]) > 1:
                aa_tp1 = rso.choice(np.where(Q[obs_tp1_base10, :] == aa_tp1)[0])

            # update table entry for Q(s,a)
            Q[obs_base10, aa] = Q[obs_base10, aa] + \
                                alpha * (reward_tp1 + discount_factor * (Q[obs_tp1_base10, aa_tp1] - Q[obs_base10, aa]))

            obs_base10 = obs_tp1_base10
            t += 1

    return Q
