import unittest

import numpy as np

from input_data import _balance_data


class TestInputData(unittest.TestCase):
    def test_balance_data(self):
        n_categories = 4
        labels_per_category = 8
        label_array = np.array(
            [[1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4], [1],
             [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4]])
        image_array = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32])
        expected_labels = np.array([[1], [2], [3], [4], [1], [2], [3], [4], [1], [1], [1], [1]])
        expected_images = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17, 21])

        computed_labels, computed_images = _balance_data(
            label_array, image_array, n_categories, labels_per_category, 1)

        self.assertTrue(np.array_equal(computed_labels, expected_labels))
        self.assertTrue(np.array_equal(computed_images, expected_images))
