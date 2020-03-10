import numpy as np
import gym
import sorty


def random_run_sorty(max_iters=50):
    # env = Sorty(n=n)
    env = gym.make('sorty-v0')
    obs = env.reset()
    done = False
    count = 0
    while not done and count < max_iters:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print("obs: {}, reward: {}".format(obs, reward))


def random_sorty_perf(iters=100, max_iters=100):
    # env = Sorty(n=n)
    counts = []
    for i in range(iters):
        env = gym.make('sorty-v0', n=6)
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
        counts.append(count)
    print("Average out of {}: {}".format(iters, np.mean(counts)))


if __name__ == "__main__":
    # random_run_sorty()
    random_sorty_perf()