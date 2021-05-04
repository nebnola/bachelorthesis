import unittest
import numpy as np
import process, stats, fit


class TestOrnsteinUhlenbeck(unittest.TestCase):
    def test_shape(self):
        t, samples = process.ornstein_uhlenbeck(1, 0.1, theta=1, sigma=0.5, size=4)
        self.assertEqual(len(t), 11)
        self.assertEqual(samples.shape, (4, 11))

    def test_x0_num(self):
        t, samples = process.ornstein_uhlenbeck(1, 0.1, theta=1, sigma=0.5, x_0=2.7, size=4)
        self.assertTrue(np.array_equal(samples[:, 0], np.full(4, 2.7)))

    def test_x0_array(self):
        x_0 = np.array([1, 2, -0.5, 0])
        t, samples = process.ornstein_uhlenbeck(1, 0.1, theta=1, sigma=0.5, x_0=x_0, size=4)
        self.assertTrue(np.array_equal(samples[:, 0], x_0))


class TestSingleAutocorrelation(unittest.TestCase):
    def test_consistent_with_full_autocorrelation(self):
        t, samples = process.ornstein_uhlenbeck(1, 0.1, theta=1, sigma=0.5, size=4)
        autocor = stats.autocorrelation(samples)
        single_autocor = stats.single_autocorrelation(samples, 6)
        self.assertTrue(np.array_equal(autocor[:,6], single_autocor))


class TestMatricesFromParams(unittest.TestCase):
    def test_3d(self):
        expected_lambdas = np.array([1+1j, 2-1j, 3])
        b_entries = np.array([2-1j, -1.5+2j, -1-1j, 4, 2+1.5j, 1.5-0.5j])
        expected_b = np.array([[2-1j, -1.5+2j, -1-1j],[0,4,2+1.5j],[0,0,1.5-0.5j]])
        param = np.concatenate((expected_lambdas.real , b_entries.real , expected_lambdas.imag , b_entries.imag))
        lambdas, b = fit.matrices_from_params(param, 3)
        self.assertEqual(lambdas.tolist() , expected_lambdas.tolist())
        self.assertEqual(expected_b.tolist(), b.tolist())

    def test_raise_value_error(self):
        param = list(range(6))
        self.assertRaises(ValueError, fit.matrices_from_params, param, 3)


class TestResiduals(unittest.TestCase):
    def test_dimensions(self):
        lambdas = np.array([1 + 1j, 2 - 1j, 3])
        b_entries = np.array([2 - 1j, -1.5 + 2j, -1 - 1j, 4, 2 + 1.5j, 1.5 - 0.5j])
        param = np.concatenate((lambdas.real, b_entries.real, lambdas.imag, b_entries.imag))
        t = np.linspace(0,1,20)
        res = fit.residuals_ou(param, fit.example_autocorrelation, t, 3)
        self.assertEqual(res.shape, (20,))
