import unittest
import numpy as np
import process, stats


class TestOrnsteinUhlenbeck(unittest.TestCase):
    def test_shape(self):
        t, samples = process.ornstein_uhlenbeck(1, 0.1, size=4)
        self.assertEqual(len(t), 11)
        self.assertEqual(samples.shape, (4, 11))

    def test_x0_num(self):
        t, samples = process.ornstein_uhlenbeck(1, 0.1, x_0=2.7, size=4)
        self.assertTrue(np.array_equal(samples[:, 0], np.full(4, 2.7)))

    def test_x0_array(self):
        x_0 = np.array([1, 2, -0.5, 0])
        t, samples = process.ornstein_uhlenbeck(1, 0.1, x_0=x_0, size=4)
        self.assertTrue(np.array_equal(samples[:, 0], x_0))


class TestSingleAutocorrelation(unittest.TestCase):
    def test_consistent_with_full_autocorrelation(self):
        t, samples = process.ornstein_uhlenbeck(1, 0.1, size=4)
        autocor = stats.autocorrelation(samples)
        single_autocor = stats.single_autocorrelation(samples, 6)
        self.assertTrue(np.array_equal(autocor[:,6], single_autocor))