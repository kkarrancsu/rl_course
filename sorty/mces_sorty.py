import numpy as np
import gym
import sorty

import algorithms.montecarlo


def mces_sorty_perf(iters=100, max_iters=200):
    env = gym.make('sorty-v0', n=3)
    pi = algorithms.montecarlo.exploring_starts(env, num_episodes=int(1e4))

    env = gym.make('sorty-v0', n=3)
    counts = []
    for i in range(iters):
        env = gym.make('sorty-v0', n=3)
        obs = env.reset()
        done = False
        count = 0
        while not done and count < max_iters:
            action = pi[tuple(obs)]
            obs, reward, done, info = env.step(action)
            # print("obs: {}, reward: {}".format(obs, reward))
            count += 1
            if done and reward > 0:
                counts.append(count)
        counts.append(count)
    print("Average out of {}: {}".format(len(counts), np.mean(counts)))


if __name__ == "__main__":
    mces_sorty_perf()
