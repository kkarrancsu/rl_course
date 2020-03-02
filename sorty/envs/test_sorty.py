import unittest
import sorty

import numpy as np


class MyTestCase(unittest.TestCase):
    def test_leq_iip1_state(self):

        obs = np.asarray([1, 2, 3, 4, 0, 0, 0, 0]).astype(int)
        max_val_to_sort = 4
        obs_update_expected = np.asarray([1, 2, 3, 4, 1, 1, 1, 1]).astype(int)
        obs_update_actual = sorty.leq_iip1_state(obs, max_val_to_sort)
        self.assertTrue(np.array_equal(obs_update_expected, obs_update_actual))

        obs = np.asarray([1, 2, 4, 3, 0, 0, 0, 0]).astype(int)
        max_val_to_sort = 4
        obs_update_expected = np.asarray([1, 2, 4, 3, 1, 1, 0, 0]).astype(int)
        obs_update_actual = sorty.leq_iip1_state(obs, max_val_to_sort)
        self.assertTrue(np.array_equal(obs_update_expected, obs_update_actual))

        obs = np.asarray([4, 3, 2, 1, 0, 0, 0, 0]).astype(int)
        max_val_to_sort = 4
        obs_update_expected = np.asarray([4, 3, 2, 1, 0, 0, 0, 0]).astype(int)
        obs_update_actual = sorty.leq_iip1_state(obs, max_val_to_sort)
        self.assertTrue(np.array_equal(obs_update_expected, obs_update_actual))

    def test_action_space_mapping(self):
        so = sorty.Sorty(4)
        num_actions = 6  # n*(n-1)/2
        for ii in range(num_actions):
            swap_i, swap_j = so.action_mapping[ii]
            if ii == 0:
                self.assertEqual([swap_i, swap_j], [0, 1])
            elif ii == 1:
                self.assertEqual([swap_i, swap_j], [0, 2])
            elif ii == 2:
                self.assertEqual([swap_i, swap_j], [0, 3])
            elif ii == 3:
                self.assertEqual([swap_i, swap_j], [1, 2])
            elif ii == 4:
                self.assertEqual([swap_i, swap_j], [1, 3])
            elif ii == 5:
                self.assertEqual([swap_i, swap_j], [2, 3])

        so = sorty.Sorty(5)
        num_actions = 10  # n*(n-1)/2
        for ii in range(num_actions):
            swap_i, swap_j = so.action_mapping[ii]
            if ii == 0:
                self.assertEqual([swap_i, swap_j], [0, 1])
            elif ii == 1:
                self.assertEqual([swap_i, swap_j], [0, 2])
            elif ii == 2:
                self.assertEqual([swap_i, swap_j], [0, 3])
            elif ii == 3:
                self.assertEqual([swap_i, swap_j], [0, 4])
            elif ii == 4:
                self.assertEqual([swap_i, swap_j], [1, 2])
            elif ii == 5:
                self.assertEqual([swap_i, swap_j], [1, 3])
            elif ii == 6:
                self.assertEqual([swap_i, swap_j], [1, 4])
            elif ii == 7:
                self.assertEqual([swap_i, swap_j], [2, 3])
            elif ii == 8:
                self.assertEqual([swap_i, swap_j], [2, 4])
            elif ii == 9:
                self.assertEqual([swap_i, swap_j], [3, 4])


if __name__ == '__main__':
    unittest.main()
