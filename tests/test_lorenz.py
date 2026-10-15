import unittest

import numpy as np

from seir.lorenz import lorenz_derivatives, simulate


class LorenzTests(unittest.TestCase):
    def test_origin_is_an_equilibrium(self):
        self.assertEqual(lorenz_derivatives(0.0, (0.0, 0.0, 0.0), 10.0, 28.0, 8.0 / 3.0), (0.0, 0.0, 0.0))

    def test_derivatives_match_lorenz_equations(self):
        derivatives = lorenz_derivatives(0.0, (1.0, 2.0, 3.0), 10.0, 28.0, 8.0 / 3.0)
        np.testing.assert_allclose(derivatives, (10.0, 23.0, -6.0))

    def test_simulation_returns_three_finite_coordinates(self):
        times = np.linspace(0.0, 2.0, 101)

        trajectory = simulate(times)

        self.assertEqual(trajectory.shape, (len(times), 3))
        self.assertTrue(np.all(np.isfinite(trajectory)))
        np.testing.assert_allclose(trajectory[0], (1.0, 1.0, 1.0))

    def test_rejects_negative_parameter(self):
        with self.assertRaisesRegex(ValueError, "Parameters"):
            simulate((0.0, 1.0), parameters=(10.0, -1.0, 8.0 / 3.0))

    def test_rejects_non_increasing_times(self):
        with self.assertRaisesRegex(ValueError, "строго возрастающем порядке"):
            simulate((0.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
