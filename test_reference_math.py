import itertools
import unittest

import numpy as np


class ReferenceMathTest(unittest.TestCase):
    def test_subset_distribution_and_gradient_invariants(self):
        scores = np.asarray([1.2, -0.4, 0.1, 0.7, -0.2], dtype=np.float64)
        positives = np.asarray([0, 2], dtype=np.int64)
        p = 2
        temperature = 0.37

        subsets = list(itertools.combinations(range(scores.size), p))
        choices = np.zeros((len(subsets), scores.size), dtype=np.float64)
        for row, subset in enumerate(subsets):
            choices[row, list(subset)] = 1.0
        utilities = choices @ scores
        shifted = utilities / temperature
        shifted -= shifted.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        marginals = probabilities @ choices

        target = np.zeros_like(scores)
        target[positives] = 1.0
        gradient = marginals - target

        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertAlmostEqual(float(marginals.sum()), float(p))
        self.assertAlmostEqual(float(gradient.sum()), 0.0)
        self.assertTrue(np.all(marginals >= 0.0))
        self.assertTrue(np.all(marginals <= 1.0))


if __name__ == "__main__":
    unittest.main()
