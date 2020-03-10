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
    n = int(len(obs) / 2)

    obs[n:-1] = (obs[0:n - 1] <= obs[1:n]).astype(int)
    # insert status for the last element
    if obs[n - 1] == max_val_to_sort:  # todo: new state function doesn't use max_val_to_sort, remove?
        obs[-1] = 1
    else:
        obs[-1] = 0

    # return updated observation
    return obs


def leq_iip2_state(state):
    """
    Take the state of the array and return an observation. The observation consists of an array of 0s and 1s, 0
    indicating that a value is less than or equal to an index to the right of it as follows:

    [idx0 < idx1, idx0 < idx2, idx0 < idx3, idx0 < idx4,
                  idx1 < idx2, idx1 < idx3, idx1 < idx4,
                               idx2 < idx3, idx2 < idx4,
                                            idx3 < idx4
    ]
    :param state: (np.array or list) array or list to be sorted
    :return (np.array) the observation
    """
    obs = []
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            obs.append(int(state[i] <= state[j]))
    return np.array(obs)


class Sorty(gym.Env):
    """
    A gym environment to train an RL agent to sort an array
    """
    LO = 0
    HI = 1000

    def __init__(self, n=5, state_type=leq_iip2_state, SEED=1234):
        """

        :param n: the # of elements in the array
        """
        self.n = n
        # The state function handle
        self.F_state = state_type

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
        num_actions = int(n * (n - 1) / 2)
        self._action_space = spaces.Discrete(num_actions)  # todo: this action space may be too large... @Chace: why?
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
        self.state, self.final_state = self._set_state()

        # observation_space is a reserved attribute (property) for gym objects, and should be a gym.Spaces object
        obs = self.F_state(self.state)
        self._observation_space = spaces.Box(0, 1, shape=(len(obs),), dtype=np.int8)

        # num_states = len(self.observation)
        # self.state = spaces.Discrete(num_states)
        self.max_val_to_sort = np.max(self.state)
        # update the state-space
        self.F_state(self.state)

    # I'm not super familiar with python properties, but I think this is how this is supposed to be done... so I'm
    # trying it.
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _set_state(self):
        state = self.random_state.choice(np.arange(Sorty.LO, Sorty.HI), self.n)  # sample without replacement
        final_state = np.sort(state)  # for double checking
        return state, final_state

    def step(self, action):
        if type(action) is int:
            aa = action
        else:
            try:
                aa = action[0]
            except IndexError:
                # this is really weird, but we need to do this b/c numpy is returning a 0D array.  I don't know why
                # it's doing that.  Is it b/c of the way the action_space was setup?  That seems OK to me ...
                # >> print(type(action)) --> numpy.ndarrya
                # >> print(action.ndim)  --> 0
                # >> print(action.size)  --> 1
                # To extract the actual action, I googled around and came across this solution from this
                # SO link: https://stackoverflow.com/a/35617558/1057098
                aa = action[()]
                # Chace: right this should not be necessary, what code is returning this?

        ii, jj = self.action_mapping[aa]
        # update the underlying array based on the action taken
        tmp_val = self.state[ii]
        self.state[ii] = self.state[jj]
        self.state[jj] = tmp_val

        # compute the new state of the task
        obs = self.F_state(self.state)

        # check if we're done & issue reward accordingly
        if np.array_equal(self.state, self.final_state):
            done = True
            # if we're sorted, give it a nice bar of gold
            reward = 100
        else:
            done = False
            # encourage it to sort as fast as possible
            reward = -1
            if (self.state == self.final_state).all():
                # todo: this Error is coming up for me, may need to change done condition to the check
                raise RuntimeError("Sort condition didn't work! Not done, but state was sorted")

        info_dict = dict()
        return obs, reward, done, info_dict

    def reset(self):
        self.state, self.final_state = self._set_state()
        # update the state-space
        obs = self.F_state(self.state)
        return obs

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    pass
