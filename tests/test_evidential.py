import unittest

import torch

from models.evidential import fuse_opinions, opinion_from_logits


class EvidentialTests(unittest.TestCase):
    def test_opinion_is_normalized(self):
        opinion = opinion_from_logits(torch.tensor([[4.0, -4.0]]))
        total = opinion.belief.sum() + opinion.uncertainty
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-6))
        self.assertTrue(torch.isfinite(opinion.alpha).all())

    def test_fusion_is_finite_under_conflict(self):
        first = opinion_from_logits(torch.tensor([[12.0, -12.0]]))
        second = opinion_from_logits(torch.tensor([[-12.0, 12.0]]))
        fused = fuse_opinions(first, second)
        self.assertTrue(torch.isfinite(fused.alpha).all())
        self.assertGreaterEqual(float(fused.uncertainty), 0.0)
        self.assertLessEqual(float(fused.uncertainty), 1.0)

    def test_uncertain_inputs_remain_uncertain(self):
        first = opinion_from_logits(torch.zeros(1, 2))
        second = opinion_from_logits(torch.zeros(1, 2))
        fused = fuse_opinions(first, second)
        self.assertAlmostEqual(float(fused.expected[0, 0]), 0.5, places=5)
        self.assertGreater(float(fused.uncertainty), 0.0)


if __name__ == "__main__":
    unittest.main()
