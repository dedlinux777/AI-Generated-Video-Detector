"""Evidential opinions and Dempster-Shafer fusion utilities."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Opinion:
    """A two-class Dirichlet opinion."""

    alpha: torch.Tensor
    expected: torch.Tensor
    belief: torch.Tensor
    uncertainty: torch.Tensor
    entropy: torch.Tensor


def opinion_from_logits(logits: torch.Tensor) -> Opinion:
    """Convert classifier logits into a numerically safe evidential opinion."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Evidential classification requires logits shaped [batch, 2].")

    evidence = torch.nn.functional.softplus(logits).clamp_min(0.0)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True).clamp_min(torch.finfo(alpha.dtype).eps)
    expected = alpha / strength
    belief = evidence / strength
    uncertainty = (2.0 / strength).clamp(0.0, 1.0)
    entropy = -(expected * expected.clamp_min(1e-8).log()).sum(dim=1)
    return Opinion(alpha, expected, belief, uncertainty, entropy)


def fuse_opinions(first: Opinion, second: Opinion) -> Opinion:
    """Fuse two opinions with the conjunctive Dempster-Shafer rule."""
    if first.alpha.shape != second.alpha.shape:
        raise ValueError("Opinions must have matching batch and class dimensions.")

    b1, b2 = first.belief, second.belief
    u1, u2 = first.uncertainty, second.uncertainty
    conflict = (b1[:, 0] * b2[:, 1] + b1[:, 1] * b2[:, 0]).unsqueeze(1)
    denominator = (1.0 - conflict).clamp_min(1e-8)
    fused_belief = (b1 * b2 + b1 * u2 + b2 * u1) / denominator
    fused_uncertainty = (u1 * u2) / denominator
    fused_uncertainty = fused_uncertainty.clamp(0.0, 1.0)

    strength = (2.0 / fused_uncertainty.clamp_min(1e-8)).clamp_min(2.0)
    evidence = (fused_belief * strength).clamp_min(0.0)
    alpha = evidence + 1.0
    expected = alpha / alpha.sum(dim=1, keepdim=True)
    entropy = -(expected * expected.clamp_min(1e-8).log()).sum(dim=1)
    return Opinion(alpha, expected, fused_belief, fused_uncertainty, entropy)


def decision(opinion: Opinion, uncertainty_threshold: float = 0.6,
             entropy_threshold: float = 0.95) -> str:
    """Return REAL, FAKE, or AMBIGUOUS using uncertainty and normalized entropy."""
    expected = opinion.expected[0]
    normalized_entropy = float(opinion.entropy[0] / math.log(2.0))
    if (float(opinion.uncertainty[0]) >= uncertainty_threshold
            or normalized_entropy >= entropy_threshold):
        return "AMBIGUOUS"
    return "REAL" if int(expected.argmax()) == 0 else "FAKE"
