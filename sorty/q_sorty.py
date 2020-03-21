import numpy as np
import gym

import sorty

import algorithms.qlearning


def q_sorty_perf(iters=100, max_iters=200):
    env = gym.make('sorty-v0', n=5)
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
    print("Average out of {}: {}".format(len(counts), np.mean(counts)))


def sarsa_sorty_perf(iters=100, max_iters=200):
    env = gym.make('sorty-v0', n=5)
    Q = algorithms.qlearning.sarsa(env, num_episodes=int(1e4), SEED=5, alpha=0.1, eps=0.01)

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
    print("Average out of {}: {}".format(len(counts), np.mean(counts)))


if __name__ == "__main__":
    q_sorty_perf()
    sarsa_sorty_perf()
