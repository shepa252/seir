import unittest

import numpy as np

from seir.van_der_pol import simulate, van_der_pol_derivatives


class VanDerPolTests(unittest.TestCase):
    def test_derivative_at_origin_is_zero(self):
        self.assertEqual(van_der_pol_derivatives(0.0, (0.0, 0.0), 1.0), (0.0, 0.0))

    def test_zero_mu_matches_harmonic_oscillator_derivatives(self):
        self.assertEqual(van_der_pol_derivatives(0.0, (2.0, 3.0), 0.0), (3.0, -2.0))

    def test_simulation_returns_position_and_velocity(self):
        times = np.linspace(0.0, 10.0, 201)

        state = simulate(times, initial_state=(1.0, 0.0), mu=1.0)

        self.assertEqual(state.shape, (len(times), 2))
        self.assertTrue(np.all(np.isfinite(state)))
        np.testing.assert_allclose(state[0], (1.0, 0.0))

    def test_rejects_negative_mu(self):
        with self.assertRaisesRegex(ValueError, "Mu"):
            simulate((0.0, 1.0), mu=-1.0)

    def test_rejects_non_increasing_times(self):
        with self.assertRaisesRegex(ValueError, "строго возрастающем порядке"):
            simulate((0.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
