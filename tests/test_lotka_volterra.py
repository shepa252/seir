import unittest

import numpy as np

from seir.lotka_volterra import lotka_volterra_derivatives, simulate


class LotkaVolterraTests(unittest.TestCase):
    def test_derivatives_at_positive_equilibrium_are_zero(self):
        alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5
        equilibrium = (gamma / delta, alpha / beta)

        derivatives = lotka_volterra_derivatives(
            0.0, equilibrium, alpha, beta, delta, gamma
        )

        np.testing.assert_allclose(derivatives, (0.0, 0.0), atol=1e-12)

    def test_simulation_returns_nonnegative_populations_with_requested_shape(self):
        times = np.linspace(0.0, 10.0, 101)

        populations = simulate(times)

        self.assertEqual(populations.shape, (len(times), 2))
        self.assertTrue(np.all(np.isfinite(populations)))
        self.assertTrue(np.all(populations >= 0.0))
        np.testing.assert_allclose(populations[0], (40.0, 9.0))

    def test_rejects_non_increasing_times(self):
        with self.assertRaisesRegex(ValueError, "строго возрастающем порядке"): 
            simulate((0.0, 1.0, 1.0))

    def test_rejects_negative_initial_population(self):
        with self.assertRaisesRegex(ValueError, "Initial_state"):
            simulate((0.0, 1.0), initial_state=(-1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
