"""Shared implementation for the Colab ETMC training and evaluation workflow."""
from __future__ import annotations

import json
import random
import time
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Allow scripts invoked as `python training/NN_*.py` to import repository models.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cached_arrays(csv_path: str | Path):
    """Read prepared CSV rows and disk cache lazily; modalities remain paired by row."""
    frame = pd.read_csv(csv_path)
    return frame


class PreparedAVDataset(Dataset):
    def __init__(self, csv_path: str | Path):
        self.frame = load_cached_arrays(csv_path)
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        frames = np.load(row.cache_frames, mmap_mode="r").copy()
        audio = np.load(row.cache_audio, mmap_mode="r").copy()
        # Cache stores RGB as [T,H,W,C]; model consumes [T,C,H,W].
        if frames.ndim == 4 and frames.shape[-1] == 3:
            frames = np.transpose(frames, (0, 3, 1, 2)).copy()
        return {
            "frames": torch.from_numpy(frames).float(),
            "audio": torch.from_numpy(audio).float(),
            "visual_label": torch.tensor(int(row.visual_label), dtype=torch.long),
            "audio_label": torch.tensor(int(row.audio_label), dtype=torch.long),
            "overall_label": torch.tensor(int(row.overall_label), dtype=torch.long),
        }


class ETMC(nn.Module):
    """EfficientNet + RawNet with video-level frame averaging and DS fusion.

    The pseudo-view is a learned joint representation, not an independent sensor.
    """
    def __init__(self, image_pretrained: bool = True, rawnet_hidden: int = 1024):
        super().__init__()
        import timm
        # The repository's B4 NS EfficientNet family is used as the visual backbone.
        self.visual_pretrained_loaded = False
        self.visual_init_error = None
        try:
            self.visual_encoder = timm.create_model("tf_efficientnet_b4_ns", pretrained=image_pretrained,
                                                     drop_path_rate=0.5, num_classes=0)
            self.visual_pretrained_loaded = bool(image_pretrained)
        except Exception as exc:
            if not image_pretrained: raise
            self.visual_init_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"Could not download compatible timm EfficientNet-B4 NS weights ({self.visual_init_error}); falling back to random EfficientNet initialization.", RuntimeWarning)
            self.visual_encoder = timm.create_model("tf_efficientnet_b4_ns", pretrained=False,
                                                     drop_path_rate=0.5, num_classes=0)
        self.visual_dim = 1792
        # Reuse the repository RawNet implementation as a feature encoder.
        args = SimpleNamespace(device="cpu", in_channels=1, gru_node=rawnet_hidden,
                               nb_gru_layer=1, nb_fc_node=rawnet_hidden,
                               nb_classes=2, pretrained_audio_encoder=False,
                               freeze_audio_encoder=False)
        from models.image import RawNet as RepoRawNet
        self.audio_encoder = RepoRawNet(args)
        self.audio_dim = rawnet_hidden
        self.visual_head = nn.Linear(self.visual_dim, 2)
        self.audio_head = nn.Linear(self.audio_dim, 2)
        self.pseudo_head = nn.Sequential(nn.Linear(self.visual_dim + self.audio_dim, 512),
                                         nn.ReLU(inplace=True), nn.Dropout(0.2),
                                         nn.Linear(512, 2))

    @staticmethod
    def evidence(logits): return F.softplus(logits.float())

    @staticmethod
    def ds_combine(alpha_a, alpha_b):
        """Stable two-opinion Dempster-Shafer combination with conflict guard."""
        eps = torch.finfo(alpha_a.dtype).eps
        k = alpha_a.shape[-1]
        sa = alpha_a.sum(-1, keepdim=True).clamp_min(eps)
        sb = alpha_b.sum(-1, keepdim=True).clamp_min(eps)
        ba, bb = (alpha_a - 1).clamp_min(0) / sa, (alpha_b - 1).clamp_min(0) / sb
        ua, ub = (k / sa).clamp(0, 1), (k / sb).clamp(0, 1)
        conflict = (ba.unsqueeze(2) * bb.unsqueeze(1)).sum((1, 2), keepdim=False).unsqueeze(1)
        denom = (1.0 - conflict).clamp_min(1e-6)
        belief = (ba * bb + ba * ub + bb * ua) / denom
        uncertainty = (ua * ub) / denom
        # Near-total conflict can produce a mass sum above one after clamping.
        total = (belief.sum(-1, keepdim=True) + uncertainty).clamp_min(eps)
        belief, uncertainty = belief / total, uncertainty / total
        strength = (k / uncertainty.clamp_min(1e-6)).clamp_min(k)
        fused_evidence = belief * strength
        return (fused_evidence + 1.0).clamp_min(1.0)

    def forward(self, frames, waveform):
        # frames [B,T,3,H,W]; aggregate sampled frames into one video opinion.
        b, t, c, h, w = frames.shape
        images = frames.reshape(b * t, c, h, w)
        visual_features = self.visual_encoder.forward_features(images)
        visual_features = F.adaptive_avg_pool2d(visual_features, 1).flatten(1)
        visual_features = visual_features.reshape(b, t, -1).mean(dim=1)
        # SincConv stores its working device as a plain attribute, so keep it aligned
        # after model.to(device) (the historical implementation predates buffers).
        self.audio_encoder.Sinc_conv.device = str(waveform.device)
        audio_features = self.audio_encoder.forward_features(waveform)
        visual_alpha = self.evidence(self.visual_head(visual_features)) + 1.0
        audio_alpha = self.evidence(self.audio_head(audio_features)) + 1.0
        pseudo_features = torch.cat([visual_features, audio_features], dim=1)
        pseudo_alpha = self.evidence(self.pseudo_head(pseudo_features)) + 1.0
        fused_alpha = self.ds_combine(self.ds_combine(visual_alpha, audio_alpha), pseudo_alpha)
        return visual_alpha, audio_alpha, pseudo_alpha, fused_alpha

    def fused_without_pseudo(self, visual_alpha, audio_alpha):
        return self.ds_combine(visual_alpha, audio_alpha)

    def set_stage(self, stage: int):
        for p in self.visual_encoder.parameters(): p.requires_grad = False
        for p in self.audio_encoder.parameters(): p.requires_grad = False
        if stage >= 2:
            # Fine-tune later EfficientNet blocks and the final RawNet blocks/GRU.
            for name, p in self.visual_encoder.named_parameters():
                if any(s in name for s in ("blocks.5", "blocks.6", "conv_head", "bn2")):
                    p.requires_grad = True
            for name, p in self.audio_encoder.named_parameters():
                if name.startswith(("block4", "block5", "bn_before_gru", "gru", "fc1_gru")):
                    p.requires_grad = True
        for module in (self.visual_head, self.audio_head, self.pseudo_head):
            for p in module.parameters(): p.requires_grad = True


def evidential_loss(labels, alpha, class_weights=None, step=0, anneal_steps=1000):
    """Expected categorical cross-entropy plus annealed Dirichlet KL penalty."""
    alpha = alpha.float().clamp_min(1.0)
    classes = alpha.shape[1]
    onehot = F.one_hot(labels, classes).float()
    strength = alpha.sum(1, keepdim=True).clamp_min(1e-6)
    nll = (onehot * (torch.digamma(strength) - torch.digamma(alpha))).sum(1)
    if class_weights is not None: nll = nll * class_weights[labels]
    adjusted = onehot + (1.0 - onehot) * alpha
    sa = adjusted.sum(1, keepdim=True).clamp_min(1e-6)
    log_b = torch.lgamma(sa) - torch.lgamma(adjusted).sum(1, keepdim=True)
    log_b_uniform = torch.lgamma(torch.tensor(float(classes), device=alpha.device))
    log_b_uniform = log_b_uniform - classes * torch.lgamma(torch.tensor(1.0, device=alpha.device))
    kl = ((adjusted - 1) * (torch.digamma(adjusted) - torch.digamma(sa))).sum(1, keepdim=True) + log_b - log_b_uniform
    return nll.mean() + min(1.0, step / max(1, anneal_steps)) * kl.mean()


def uncertainty(alpha): return alpha.shape[1] / alpha.sum(dim=1).clamp_min(1e-8)


def normalize_frames(frames):
    """Cached RGB bytes -> tensor normalized using the timm B4 NS data config."""
    x = frames.float() / 255.0
    mean = torch.tensor((0.485, 0.456, 0.406), device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    return (x - mean) / std


def load_pretrained_encoders(model: ETMC, checkpoint_path: str | Path | None):
    report = {"checkpoint": str(checkpoint_path) if checkpoint_path else None, "loaded": {}, "notes": [],
              "visual_initialization": "timm pretrained EfficientNet-B4 NS" if model.visual_pretrained_loaded else "random EfficientNet-B4 NS initialization",
              "visual_initialization_error": model.visual_init_error}
    if not checkpoint_path or not Path(checkpoint_path).exists():
        report["notes"].append(f"No repository checkpoint found; visual branch uses {report['visual_initialization']} and RawNet uses its standard initialized weights.")
        return report
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # RawNet checkpoint is a PyTorch state dict compatible with the repository's RawNet module.
    state = ckpt.get("spec_encoder", {}) if isinstance(ckpt, dict) else {}
    target = model.audio_encoder.state_dict()
    compatible = {k: v for k, v in state.items() if k in target and tuple(target[k].shape) == tuple(v.shape)}
    if compatible:
        result = model.audio_encoder.load_state_dict(compatible, strict=False)
        report["loaded"]["spec_encoder"] = {"matched_tensors": len(compatible), "total_tensors": len(target),
                                              "missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}
    else:
        report["notes"].append("spec_encoder had no matching PyTorch tensors (state keys or dimensions differ); RawNet uses initialized weights.")
    # The rgb_encoder entry in model.pth is an ONNX-export initializer map for TF EfficientNetV2-B0,
    # not a PyTorch timm B4 state_dict; loading it into the repository B4 backbone is unsupported.
    report["notes"].append("rgb_encoder is a TensorFlow EfficientNetV2-B0 ONNX initializer map, incompatible with the PyTorch timm EfficientNet-B4 backbone; ignored explicitly.")
    return report


def checkpoint_model_state(checkpoint):
    return checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(value, f, indent=2, default=str)

