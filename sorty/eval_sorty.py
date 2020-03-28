import numpy as np
import gym
import pandas as pd
from tqdm import tqdm

import sorty

import algorithms.qlearning
import algorithms.montecarlo


N = 5
NUM_EVAL_ITERS = 100
MAX_ITERS_PER_EVAL = 200


def random_sorty_perf(arr_len=N, iters=NUM_EVAL_ITERS, max_iters=MAX_ITERS_PER_EVAL):
    env = gym.make('sorty-v0', n=arr_len)
    counts = []
    for i in range(iters):
        obs = env.reset()
        done = False
        count = 0
        while not done and count < max_iters:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            # print("obs: {}, reward: {}".format(obs, reward))
            count += 1
            if done:
                counts.append(count)

    return counts


def mces_sorty_perf(arr_len=N, iters=NUM_EVAL_ITERS, max_iters=MAX_ITERS_PER_EVAL):
    env = gym.make('sorty-v0', n=arr_len)
    pi = algorithms.montecarlo.exploring_starts(env, num_episodes=int(1e4))

    counts = []
    for i in range(iters):
        obs = env.reset()
        done = False
        count = 0
        while not done and count < max_iters:
            obs_base10 = int("".join(str(x) for x in obs), 2)
            action = pi[obs_base10]

            obs, reward, done, info = env.step(action)
            # print("obs: {}, reward: {}".format(obs, reward))
            count += 1
            if done and reward > 0:
                counts.append(count)
    return counts


def q_sorty_perf(arr_len=N, iters=NUM_EVAL_ITERS, max_iters=MAX_ITERS_PER_EVAL):
    env = gym.make('sorty-v0', n=arr_len)
    Q = algorithms.qlearning.qlearning(env, num_episodes=int(1e4), SEED=0, eps=0.01)
    # eps=0.1, SEED=0, num_episodes=1e4, produces poor results...
    #  we fixed it via random choice from equally best actions

    counts = []
    for i in range(iters):
        obs = env.reset()
        done = False
        count = 0
        while not done and count < max_iters:
            obs_base10 = int("".join(str(x) for x in obs), 2)
            action = np.argmax(Q[obs_base10])
            obs, reward, done, info = env.step(action)
            # print("obs: {}, reward: {}".format(obs, reward))
            count += 1
            if done and reward > 0:
                counts.append(count)
    return counts


def sarsa_sorty_perf(arr_len=N, iters=NUM_EVAL_ITERS, max_iters=MAX_ITERS_PER_EVAL,
                     num_episodes=int(1e4), alpha=0.5, eps=0.01):
    env = gym.make('sorty-v0', n=arr_len)
    Q = algorithms.qlearning.sarsa(env, num_episodes=num_episodes,
                                   SEED=5, alpha=alpha, eps=eps)

    counts = []
    for i in range(iters):
        obs = env.reset()
        done = False
        count = 0
        while not done and count < max_iters:
            obs_base10 = int("".join(str(x) for x in obs), 2)
            action = np.argmax(Q[obs_base10])
            obs, reward, done, info = env.step(action)
            # print("obs: {}, reward: {}".format(obs, reward))
            count += 1
            if done and reward > 0:
                counts.append(count)
    return counts


if __name__ == "__main__":
    mu = lambda x: 0 if len(x) == 0 else np.mean(x)
    std = lambda x: 0 if len(x) == 0 else np.std(x)

    # arr_len_vec = [3, 4, 5]
    # res_list = []
    # for arr_len in arr_len_vec:
    #     random_counts = random_sorty_perf(arr_len=arr_len)
    #     mces_counts = mces_sorty_perf(arr_len=arr_len)
    #     q_counts = q_sorty_perf(arr_len=arr_len)
    #     sarsa_counts = sarsa_sorty_perf(arr_len=arr_len)
    #
    #     d = dict(n=arr_len,
    #              random_n_success=len(random_counts),
    #              random_mu=mu(random_counts),
    #              random_std=std(random_counts),
    #              mces_n_success=len(mces_counts),
    #              mces_mu=mu(mces_counts),
    #              mces_std=std(mces_counts),
    #              q_n_success=len(q_counts),
    #              q_mu=mu(q_counts),
    #              q_std=std(q_counts),
    #              sarsa_n_success=len(sarsa_counts),
    #              sarsa_mu=mu(sarsa_counts),
    #              sarsa_std=std(sarsa_counts))
    #     res_list.append(d)
    #
    # # make a dataframe of results
    # results_df = pd.DataFrame(res_list)
    # results_df.to_csv('/tmp/sorty_results.csv', index=False)

    # c = sarsa_sorty_perf()
    # print(len(c), np.mean(c), np.std(c))

    num_episodes_vec = [1e4, 5e4, 1e5, 5e5, 1e6, 5e6]
    alpha_vec = [0.1, 0.3, 0.5, 0.7, 0.9]
    eps_vec = [0.001, 0.01, 0.03, 0.07, 0.1]
    res_list = []
    for num_episodes in num_episodes_vec:
        for alpha in alpha_vec:
            for eps in eps_vec:
                c = sarsa_sorty_perf(arr_len=N, iters=NUM_EVAL_ITERS, max_iters=MAX_ITERS_PER_EVAL,
                                     num_episodes=int(num_episodes), alpha=alpha, eps=eps)
                d = dict(num_episodes=num_episodes,
                         alpha=alpha,
                         eps=eps,
                         n_success=len(c),
                         mean_swaps=mu(c),
                         std_swaps=std(c))
                res_list.append(d)

    results_df = pd.DataFrame(res_list)
    results_df.to_csv('/tmp/sarsa_results.csv', index=False)
