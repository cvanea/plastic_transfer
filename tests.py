import unittest
from unittest.mock import MagicMock

import numpy as np

from run import Run

from input_data import _balance_data, get_category_images


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

    def test_get_positive_images(self):
        label_array = np.array([[1], [1], [2], [3], [2], [1], [1]])
        image_array = np.array([1, 2, 3, 4, 5, 6, 7])
        expected_images = np.array([1, 2, 6, 7])

        computed_images = get_category_images(label_array, image_array, 1)

        self.assertTrue(np.array_equal(computed_images, expected_images))

    def test_get_balanced_smaller_data(self):
        n_categories = 4
        labels_per_category = 4
        label_array = np.array(
            [[1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4], [1],
             [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4]])
        image_array = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32])
        expected_labels = np.array([[1], [2], [3], [4], [1], [1]])
        expected_images = np.array([1, 2, 3, 4, 5, 9])

        computed_labels, computed_images = _balance_data(
            label_array, image_array, n_categories, labels_per_category, 1)

        self.assertTrue(np.array_equal(computed_labels, expected_labels))
        self.assertTrue(np.array_equal(computed_images, expected_images))



class TestRun(unittest.TestCase):
    def tearDown(self):
        pass

    def test_update_csv(self):
        m = MagicMock()
        m.val = {
            'mcc': [1, 2, 3],
            'acc': [1, 2, 3]
        }
        m.train = {
            'mcc': [1, 2, 3],
            'acc': [1, 2, 3]
        }
        m.test = {
            'mcc': [1, 2, 3],
            'acc': [1, 2, 3]
        }

        m2 = MagicMock()
        m2.val = {
            'mcc': [4, 5, 6],
            'acc': [4, 5, 6]
        }
        m2.train = {
            'mcc': [4, 5, 6],
            'acc': [4, 5, 6]
        }
        m2.test = {
            'mcc': [4, 5, 6],
            'acc': [4, 5, 6]
        }

        r = Run("unit_test", 0)
        r.target.update(1, m)
        r.naive.update(1, m)
        r.target.update(2, m2)
        r.naive.update(2, m2)
        r.target.update(3, m)
        r.naive.update(3, m)
        r.target.update(4, m2)
        r.naive.update(4, m2)

        self.assertListEqual(m.val['mcc'], list(r.target.val.mcc.df["1"].values))
        self.assertListEqual(m2.val['mcc'], list(r.target.val.mcc.df["2"].values))
        self.assertListEqual(m.val['mcc'], list(r.target.val.mcc.df["3"].values))
        self.assertListEqual(m2.val['mcc'], list(r.target.val.mcc.df["4"].values))