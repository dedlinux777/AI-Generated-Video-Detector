"""Dirichlet opinions and Dempster-Shafer fusion.

The committed encoders do not emit evidential logits: the image encoder emits
sigmoid scores and RawNet emits log-probabilities.  The probability conversion
below is therefore an explicitly labelled baseline, not a claim that the
checkpoint learned calibrated evidence.
"""

from dataclasses import dataclass
import math

import torch


_EPS = torch.finfo(torch.float32).eps


@dataclass(frozen=True)
class Opinion:
    """A batch of K-class Dirichlet opinions."""

    alpha: torch.Tensor
    expected: torch.Tensor
    belief: torch.Tensor
    uncertainty: torch.Tensor
    entropy: torch.Tensor

    def __post_init__(self) -> None:
        if self.alpha.ndim != 2 or self.alpha.shape[1] < 2:
            raise ValueError("Opinion tensors must have shape [batch, classes].")
        for value in (self.alpha, self.expected, self.belief, self.uncertainty, self.entropy):
            if not torch.isfinite(value).all():
                raise ValueError("Opinions must contain only finite values.")
        if (self.alpha <= 0).any() or (self.belief < 0).any():
            raise ValueError("Dirichlet concentrations and beliefs must be positive.")
        if (self.uncertainty < 0).any() or (self.uncertainty > 1).any():
            raise ValueError("Uncertainty must be in [0, 1].")
        if not torch.allclose(
            self.belief.sum(dim=1) + self.uncertainty,
            torch.ones_like(self.uncertainty),
            atol=1e-5,
        ):
            raise ValueError("Belief and uncertainty must sum to one.")


def opinion_from_alpha(alpha: torch.Tensor) -> Opinion:
    """Represent model-produced Dirichlet concentrations without recalibration."""
    alpha = torch.as_tensor(alpha, dtype=torch.float32)
    if alpha.ndim != 2 or alpha.shape[1] < 2:
        raise ValueError("Dirichlet concentrations must have shape [batch, classes].")
    if not torch.isfinite(alpha).all() or (alpha < 1).any():
        raise ValueError("Dirichlet concentrations must be finite and at least one.")
    classes = alpha.shape[1]
    strength = alpha.sum(dim=1, keepdim=True)
    evidence = alpha - 1.0
    belief = evidence.clamp_min(0.0) / strength
    uncertainty = (classes / strength).squeeze(1)
    expected = alpha / strength
    entropy = -(expected * expected.clamp_min(_EPS).log()).sum(dim=1)
    return Opinion(alpha, expected, belief, uncertainty, entropy)


def _probability_opinion(probabilities: torch.Tensor, evidence_strength: float) -> Opinion:
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    if probabilities.ndim == 1:
        probabilities = probabilities.unsqueeze(0)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Classification probabilities must have shape [batch, classes].")
    if not math.isfinite(evidence_strength) or evidence_strength <= 0:
        raise ValueError("Evidence strength must be a positive finite number.")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Classification probabilities must be finite and non-negative.")
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(_EPS)
    classes = probabilities.shape[1]
    evidence = probabilities * evidence_strength
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    expected = alpha / strength
    belief = evidence / strength
    uncertainty = classes / strength.squeeze(1)
    entropy = -(expected * expected.clamp_min(_EPS).log()).sum(dim=1)
    return Opinion(alpha, expected, belief, uncertainty, entropy)


def opinion_from_probabilities(
    probabilities: torch.Tensor, evidence_strength: float = 20.0
) -> Opinion:
    """Create a transparent, non-calibrated Dirichlet baseline from probabilities."""
    return _probability_opinion(probabilities, evidence_strength)


def opinion_from_log_probabilities(
    log_probabilities: torch.Tensor, evidence_strength: float = 20.0
) -> Opinion:
    """Create an opinion from a model output known to be log-softmax output."""
    log_probabilities = torch.as_tensor(log_probabilities, dtype=torch.float32)
    if not torch.isfinite(log_probabilities).all():
        raise ValueError("Log-probabilities must be finite.")
    return _probability_opinion(log_probabilities.exp(), evidence_strength)


def opinion_from_logits(logits: torch.Tensor, evidence_strength: float = 20.0) -> Opinion:
    """Create an opinion from unnormalised classifier logits."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Classifier logits must have shape [batch, classes].")
    return _probability_opinion(logits.softmax(dim=1), evidence_strength)


def fuse_opinions(first: Opinion, second: Opinion) -> Opinion:
    """Fuse opinions with normalized Dempster-Shafer conjunctive combination."""
    if first.alpha.shape != second.alpha.shape:
        raise ValueError("Opinions must have matching batch and class dimensions.")

    b1, b2 = first.belief, second.belief
    u1, u2 = first.uncertainty.unsqueeze(1), second.uncertainty.unsqueeze(1)
    pairwise = b1.unsqueeze(2) * b2.unsqueeze(1)
    conflict = pairwise.sum(dim=(1, 2)) - (b1 * b2).sum(dim=1)
    denominator = 1.0 - conflict
    if (denominator <= _EPS).any():
        raise ValueError("Dempster-Shafer fusion encountered total conflict.")

    fused_belief = (b1 * b2 + b1 * u2 + b2 * u1) / denominator.unsqueeze(1)
    fused_uncertainty = (u1 * u2 / denominator.unsqueeze(1)).squeeze(1)
    classes = first.alpha.shape[1]
    strength = classes / fused_uncertainty.clamp_min(_EPS)
    evidence = fused_belief * strength.unsqueeze(1)
    alpha = evidence + 1.0
    expected = alpha / alpha.sum(dim=1, keepdim=True)
    entropy = -(expected * expected.clamp_min(_EPS).log()).sum(dim=1)
    return Opinion(alpha, expected, fused_belief, fused_uncertainty, entropy)


def decision(
    opinion: Opinion,
    uncertainty_threshold: float = 0.6,
    entropy_threshold: float | None = None,
    margin_threshold: float = 0.05,
) -> str:
    """Return REAL, FAKE, or AMBIGUOUS using abstention criteria.

    Predictive entropy is reported for diagnostics but is not an epistemic
    uncertainty measure, so it is opt-in for abstention.  A validation study
    should set these thresholds for a trained evidential checkpoint.
    """
    if not 0 <= uncertainty_threshold <= 1 or not 0 <= margin_threshold <= 1:
        raise ValueError("Decision thresholds must be in [0, 1].")
    if entropy_threshold is not None and not 0 <= entropy_threshold <= 1:
        raise ValueError("Entropy threshold must be in [0, 1].")
    expected = opinion.expected[0]
    normalized_entropy = float(opinion.entropy[0] / math.log(expected.numel()))
    margin = float((expected.max() - expected.min()))
    if (
        float(opinion.uncertainty[0]) >= uncertainty_threshold
        or margin < margin_threshold
        or (entropy_threshold is not None and normalized_entropy >= entropy_threshold)
    ):
        return "AMBIGUOUS"
    return "REAL" if int(expected.argmax()) == 0 else "FAKE"
