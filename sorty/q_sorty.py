import numpy as np
import gym

import sorty

import algorithms.qlearning


def q_sorty_perf(iters=100, max_iters=200):
    env = gym.make('sorty-v0', n=3)
    Q = algorithms.qlearning.qlearning(env, num_episodes=int(1e4))

    counts = []
    for i in range(iters):
        obs = env.reset()
        done = False
        count = 0
        while not done and count < max_iters:
            obs_base10 = int("".join(str(x) for x in obs), 2)
            action = Q[obs_base10]
            obs, reward, done, info = env.step(action)
            # print("obs: {}, reward: {}".format(obs, reward))
            count += 1
            if done and reward > 0:
                counts.append(count)
        counts.append(count)
    print("Average out of {}: {}".format(len(counts), np.mean(counts)))


if __name__ == "__main__":
    q_sorty_perf()
