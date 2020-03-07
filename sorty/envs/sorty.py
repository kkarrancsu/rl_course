import numpy as np
import gym
from gym import spaces


def leq_iip1_state(obs, max_val_to_sort):
    """
    Defines a state vector where the state is a binary vector of length n-1, where n=len(obs) and a
    1 indicates that obs[ii] <= obs[ii+1], and 0 otherwise.  In this representation, the RL agent
    "wins" when the state vector is all all ones

    :param obs: the observation vector, in which the first n elements are expected to be actual array to be sorted,
            and the second n elements is an indication of the sorting status
    :return: an updated observation vector which
    """
    n = int(len(obs)/2)

    obs[n:-1] = (obs[0:n-1] <= obs[1:n]).astype(int)
    # insert status for the last element
    if obs[n-1] == max_val_to_sort:
        obs[-1] = 1
    else:
        obs[-1] = 0

    # return updated observation
    return obs


class Sorty(gym.Env):
    """
    A gym environment to train an RL agent to sort an array
    """
    LO = 0
    HI = 1000

    def __init__(self, n=5, state_type=leq_iip1_state, SEED=1234):
        """

        :param n: the # of elements in the array
        """
        self.n = n
        # The state function handle
        self.F_state = state_type

        # TODO: Make this a Tuple(Discrete(),Discrete()) environment rather than
        #  a direct Discrete environment.  The interface will be easier to understand.
        # The action space is encoded as follows:
        #  For an array of size n, we have (n-1)+(n-2)+...+1 possible actions (swaps). This sum converges to the
        #  following closed form formula: (n-1)/2*(1+n-1) = n*(n-1)/2
        #  The value of the action maps to  which indices get swapped as follows.
        #  Here is a sample array of size 4: [0 1 2 3]
        #    action=0 --> swap[0,1]
        #    action=1 --> swap[0,2]
        #    action=2 --> swap[0,3]
        #    action=3 --> swap[1,2]
        #    action=4 --> swap[1,3]
        #    action=5 --> swap[2,3]
        # H Here is a sample array of size 5: [0 1 2 3 4]
        #    action=0 --> swap[0,1]
        #    action=1 --> swap[0,2]
        #    action=2 --> swap[0,3]
        #    action=3 --> swap[0,4]
        #    action=4 --> swap[1,2]
        #    action=5 --> swap[1,3]
        #    action=6 --> swap[1,4]
        #    action=7 --> swap[2,3]
        #    action=8 --> swap[2,4]
        #    action=9 --> swap[3,4]
        num_actions = int(n*(n-1)/2)
        self.action_space = spaces.Discrete(num_actions)
        # build a dictionary which maps each action to the indices which need to be swapped
        self.action_mapping = dict()
        from_ii = 0
        to_ii = 1
        for ii in range(num_actions):
            self.action_mapping[ii] = (from_ii, to_ii)
            to_ii += 1
            if to_ii >= n:
                from_ii += 1
                to_ii = from_ii + 1

        # create a random array, which is the array to be sorted.  In addition to the array, the agent receives a
        # binary vector concatenated which indicates something about the state of the task to be completed,
        # in this case, the sort.  Since we will experiment with different state space functions, refer to the
        # function documentation to understand what the state space is referring to here.
        self.random_state = np.random.RandomState(SEED)
        self.observation_space = np.concatenate((self.random_state.randint(Sorty.LO, Sorty.HI, self.n),
                                                 np.zeros(self.n))).astype(int)
        # num_states = len(self.observation)
        # self.observation_space = spaces.Discrete(num_states)
        self.max_val_to_sort = np.max(self.observation_space)
        # update the state-space
        self.F_state(self.observation_space, self.max_val_to_sort)

    def step(self, action):
        # this is really weird, but we need to do this b/c numpy is returning a 0D array.  I don't know why
        # it's doing that.  Is it b/c of the way the action_space was setup?  That seems OK to me ...
        # >> print(type(action)) --> numpy.ndarrya
        # >> print(action.ndim)  --> 0
        # >> print(action.size)  --> 1
        # To extract the actual action, I googled around and came across this solution from this
        # SO link: https://stackoverflow.com/a/35617558/1057098
        aa = action[()]

        ii, jj = self.action_mapping[aa]
        # update the underlying array
        tmp_val = self.observation_space[ii]
        self.observation_space[ii] = self.observation_space[jj]
        self.observation_space[jj] = tmp_val

        # compute the new state of the task
        self.F_state(self.observation_space, self.max_val_to_sort)

        # check if we're done & issue reward accordingly
        if np.sum(self.observation_space[self.n:]) == self.n:
            done = True
            # if we're sorted, give it a nice bar of gold
            reward = 100
        else:
            done = False
            # encourage it to sort as fast as possible
            reward = -1

        info_dict = dict()
        return self.observation_space, reward, done, info_dict

    def reset(self):
        # TODO: this code is a direct copy of what is in the constructor .. remedy that
        self.observation_space = np.concatenate((self.random_state.randint(Sorty.LO, Sorty.HI, self.n),
                                                 np.zeros(self.n))).astype(int)
        self.max_val_to_sort = np.max(self.observation_space)
        # update the state-space
        self.F_state(self.observation_space, self.max_val_to_sort)
        return self.observation_space

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    pass
