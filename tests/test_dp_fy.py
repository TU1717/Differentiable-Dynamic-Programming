import itertools
import unittest


try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from losses import (
        dp_fy_loss,
        exact_inclusion_marginals,
        log_partition_dp,
        structured_surplus,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DPFYTest(unittest.TestCase):
    def setUp(self):
        self.scores = torch.tensor(
            [
                [1.2, -0.4, 0.1, 0.7, -0.2],
                [0.3, 0.9, -0.2, 0.5, 0.1],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        self.positives = torch.tensor([[0, 2], [1, 3]])
        self.p = 2
        self.temperature = 0.37

    def enumerate_distribution(self, scores):
        subsets = list(itertools.combinations(range(scores.shape[1]), self.p))
        utilities = torch.stack(
            [scores[:, subset].sum(dim=1) for subset in subsets], dim=1
        )
        log_weights = utilities / self.temperature
        probabilities = torch.softmax(log_weights, dim=1)
        return subsets, utilities, log_weights, probabilities

    def test_partition_matches_explicit_subset_enumeration(self):
        _, _, log_weights, _ = self.enumerate_distribution(self.scores)
        expected = torch.logsumexp(log_weights, dim=1)
        actual = log_partition_dp(self.scores, self.p, self.temperature)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_loss_is_exact_subset_negative_log_likelihood(self):
        _, utilities, log_weights, _ = self.enumerate_distribution(self.scores)
        log_z = torch.logsumexp(log_weights, dim=1)
        target_utility = self.scores.gather(1, self.positives).sum(dim=1)
        expected = (self.temperature * log_z - target_utility).mean()
        actual, diagnostics = dp_fy_loss(
            self.scores,
            self.positives,
            self.p,
            self.temperature,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertGreaterEqual(float(actual.detach()), 0.0)
        self.assertEqual(diagnostics["selection_mass"], float(self.p))

    def test_gradient_is_marginal_minus_target(self):
        subsets, _, _, probabilities = self.enumerate_distribution(self.scores)
        choices = self.scores.new_zeros(len(subsets), self.scores.shape[1])
        for row, subset in enumerate(subsets):
            choices[row, list(subset)] = 1.0
        marginals = probabilities @ choices
        target = self.scores.new_zeros(self.scores.shape)
        target.scatter_(1, self.positives, 1.0)
        expected_gradient = (marginals - target) / self.scores.shape[0]

        loss, _ = dp_fy_loss(
            self.scores,
            self.positives,
            self.p,
            self.temperature,
        )
        actual_gradient = torch.autograd.grad(loss, self.scores)[0]
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-11, atol=1e-11
        )
        torch.testing.assert_close(
            actual_gradient.sum(dim=1),
            torch.zeros(self.scores.shape[0], dtype=self.scores.dtype),
            rtol=0.0,
            atol=1e-12,
        )

    def test_diagnostic_marginals_have_exact_mass(self):
        marginals = exact_inclusion_marginals(
            self.scores, self.p, self.temperature
        )
        self.assertTrue(bool((marginals >= 0.0).all().item()))
        self.assertTrue(bool((marginals <= 1.0).all().item()))
        torch.testing.assert_close(
            marginals.sum(dim=1),
            torch.full(
                (self.scores.shape[0],),
                float(self.p),
                dtype=self.scores.dtype,
            ),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_loss_is_invariant_to_rowwise_constant_shifts(self):
        shift = torch.tensor([[2.7], [-1.4]], dtype=self.scores.dtype)
        original, _ = dp_fy_loss(
            self.scores,
            self.positives,
            self.p,
            self.temperature,
        )
        shifted, _ = dp_fy_loss(
            self.scores + shift,
            self.positives,
            self.p,
            self.temperature,
        )
        torch.testing.assert_close(original, shifted, rtol=1e-12, atol=1e-12)

    def test_surplus_gradient_matches_diagnostic_marginals(self):
        surplus = structured_surplus(
            self.scores, self.p, self.temperature
        ).sum()
        gradient = torch.autograd.grad(surplus, self.scores)[0]
        marginals = exact_inclusion_marginals(
            self.scores, self.p, self.temperature
        )
        torch.testing.assert_close(gradient, marginals, rtol=1e-12, atol=1e-12)

    def test_duplicate_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            dp_fy_loss(
                self.scores,
                torch.tensor([[0, 0], [1, 3]]),
                self.p,
                self.temperature,
            )


if __name__ == "__main__":
    unittest.main()
