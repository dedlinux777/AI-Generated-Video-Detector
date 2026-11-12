"""Inference for the trained ETMC video and standalone-audio modes."""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from models.TMC import ETMC
from models.evidential import Opinion, decision, opinion_from_alpha
from models import image


ROOT = Path(__file__).resolve().parent
ETMC_CHECKPOINT = ROOT / "checkpoints" / "etmc_model_best.pt"
VIDEO_SIZE = 256
AUDIO_RATE = 16_000
AUDIO_LENGTH = 64_600
EXPECTED_CLASS_MAPPING = {"0": "real", "1": "fake"}

DEFAULT_ARCHITECTURE = {
    "dropout": 0.2,
    "hidden": [],
    "num_image_embeds": 1,
    "n_classes": 2,
    "device": "cpu",
    "pretrained_image_encoder": False,
    "freeze_image_encoder": True,
    "pretrained_audio_encoder": False,
    "freeze_audio_encoder": True,
    "rgb_feature_size": 1,
    "spec_feature_size": 2,
    "nb_samp": AUDIO_LENGTH,
    "first_conv": 1024,
    "in_channels": 1,
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
}


class AudioStreamUnavailable(ValueError):
    """The input media has no audio stream to extract."""


class AudioExtractionError(ValueError):
    """An audio stream exists or may exist, but extraction failed."""


def _architecture_args(config: dict[str, Any]) -> SimpleNamespace:
    if not isinstance(config, dict):
        raise ValueError("ETMC architecture_config must be a dictionary.")
    values = dict(config)
    missing = [key for key in DEFAULT_ARCHITECTURE if key not in values]
    if missing:
        raise ValueError(f"ETMC architecture configuration is missing: {', '.join(missing)}")
    return SimpleNamespace(**values)


def _validate_mapping(mapping: Any) -> None:
    if mapping != EXPECTED_CLASS_MAPPING:
        raise ValueError(
            f"Unsupported ETMC class mapping {mapping!r}; expected {EXPECTED_CLASS_MAPPING!r}."
        )


@functools.lru_cache(maxsize=1)
def load_complete_etmc(path: str | os.PathLike[str]) -> ETMC:
    """Reconstruct and strictly load a complete trained ETMC checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained ETMC checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("ETMC checkpoint must be a dictionary.")
    required = {"state_dict", "class_mapping", "architecture", "architecture_config"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(
            f"ETMC checkpoint is incomplete; missing keys: {', '.join(sorted(missing))}."
        )
    if checkpoint["architecture"] != "ETMC":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint['architecture']!r}.")
    _validate_mapping(checkpoint["class_mapping"])
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("ETMC checkpoint state_dict must be a non-empty dictionary.")
    try:
        model = ETMC(_architecture_args(checkpoint["architecture_config"]))
        model.load_state_dict(state_dict, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"ETMC checkpoint cannot be strictly loaded: {error}") from error
    model.eval()
    return model


class _AudioETMCBranch(nn.Module):
    def __init__(self, args: SimpleNamespace) -> None:
        super().__init__()
        self.specenc = image.RawNet(args)
        self.spec_depth = nn.ModuleList()
        last_size = args.spec_feature_size
        for hidden in args.hidden:
            self.spec_depth.extend([nn.Linear(last_size, hidden), nn.ReLU(), nn.Dropout(args.dropout)])
            last_size = hidden
        self.spec_depth.append(nn.Linear(last_size, args.n_classes))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        output = self.spec_depth_forward(self.specenc(audio).flatten(1))
        return torch.nn.functional.softplus(output) + 1.0

    def spec_depth_forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.spec_depth:
            value = layer(value)
        return value


@functools.lru_cache(maxsize=1)
def _load_audio_branch() -> _AudioETMCBranch:
    checkpoint = torch.load(ETMC_CHECKPOINT, map_location="cpu")
    required = {"state_dict", "class_mapping", "architecture", "architecture_config"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError("The ETMC checkpoint is incomplete and cannot load audio independently.")
    if checkpoint["architecture"] != "ETMC":
        raise ValueError("Standalone audio requires an ETMC checkpoint.")
    _validate_mapping(checkpoint["class_mapping"])
    args = _architecture_args(checkpoint["architecture_config"])
    branch = _AudioETMCBranch(args)
    state_dict = {
        key: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("specenc.") or key.startswith("spec_depth.")
    }
    expected = branch.state_dict()
    if set(state_dict) != set(expected):
        raise ValueError("ETMC checkpoint does not contain a complete acoustic branch.")
    branch.load_state_dict(state_dict, strict=True)
    branch.eval()
    return branch


@functools.lru_cache(maxsize=1)
def _load_etmc_model() -> ETMC:
    return load_complete_etmc(str(ETMC_CHECKPOINT))


def _prepare_audio(audio: np.ndarray, rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != AUDIO_RATE and audio.size:
        output_length = max(1, round(audio.size * AUDIO_RATE / rate))
        old_x = np.linspace(0.0, 1.0, audio.size, endpoint=False)
        new_x = np.linspace(0.0, 1.0, output_length, endpoint=False)
        audio = np.interp(new_x, old_x, audio).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.pad(audio[:AUDIO_LENGTH], (0, max(0, AUDIO_LENGTH - audio.size)))[:AUDIO_LENGTH]


def load_audio(path: str | os.PathLike[str]) -> np.ndarray:
    try:
        import soundfile as sf
        audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    except (ImportError, OSError, RuntimeError, ValueError):
        try:
            import torchaudio
            waveform, rate = torchaudio.load(str(path))
            audio = waveform.mean(dim=0).numpy()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise ValueError("The audio file could not be decoded.") from error
    return _prepare_audio(audio, int(rate))


def extract_video_audio(video_path: str | os.PathLike[str]) -> str:
    output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output.close()
    command = [
        "ffmpeg", "-v", "error", "-i", str(video_path), "-map", "0:a:0",
        "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "wav", "-y", output.name,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            if "matches no streams" in stderr.lower() or "does not contain any stream" in stderr.lower():
                raise AudioStreamUnavailable("The video has no readable audio stream.")
            raise AudioExtractionError(f"Audio extraction failed: {stderr or 'FFmpeg returned an error.'}")
        if not Path(output.name).is_file() or Path(output.name).stat().st_size == 0:
            raise AudioExtractionError("Audio extraction produced an empty file.")
        return output.name
    except Exception:
        Path(output.name).unlink(missing_ok=True)
        raise


def preprocess_video(input_video: str | os.PathLike[str], n_frames: int = 8) -> list[torch.Tensor]:
    capture = cv2.VideoCapture(str(input_video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        return []
    indices = set(np.linspace(0, frame_count - 1, min(n_frames, frame_count), dtype=int))
    frames: list[torch.Tensor] = []
    for index in range(frame_count):
        if not capture.grab() or index not in indices:
            continue
        ok, frame = capture.retrieve()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (VIDEO_SIZE, VIDEO_SIZE)).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(frame.transpose(2, 0, 1)))
    capture.release()
    return frames


def _predict_etmc_opinions(frames: list[torch.Tensor], audio: np.ndarray) -> tuple[Opinion, Opinion, Opinion, Opinion]:
    model = _load_etmc_model()
    with torch.inference_mode():
        spec_alpha, rgb_alpha, pseudo_alpha, fused_alpha = model(
            torch.stack(frames), torch.from_numpy(audio).unsqueeze(0).expand(len(frames), -1)
        )
    visual_alpha = rgb_alpha.mean(dim=0, keepdim=True)
    audio_opinion = opinion_from_alpha(spec_alpha.mean(dim=0, keepdim=True))
    visual_opinion = opinion_from_alpha(visual_alpha)
    pseudo_opinion = opinion_from_alpha(pseudo_alpha.mean(dim=0, keepdim=True))
    fused_opinion = opinion_from_alpha(fused_alpha.mean(dim=0, keepdim=True))
    return visual_opinion, audio_opinion, pseudo_opinion, fused_opinion


def _format_opinion(name: str, opinion: Opinion) -> str:
    expected = opinion.expected[0].tolist()
    belief = opinion.belief[0].tolist()
    return (
        f"{name}: REAL expected={expected[0]:.3f}, FAKE expected={expected[1]:.3f}; "
        f"REAL belief={belief[0]:.3f}, FAKE belief={belief[1]:.3f}; "
        f"uncertainty={float(opinion.uncertainty[0]):.3f}"
    )


def deepfakes_spec_predict(input_audio: tuple[np.ndarray, int] | str) -> str:
    """Run only the trained ETMC acoustic head and never the visual branch."""
    model = _load_audio_branch()
    if isinstance(input_audio, (str, os.PathLike)):
        audio = load_audio(input_audio)
    else:
        raw, rate = input_audio
        audio = _prepare_audio(raw, int(rate))
    with torch.inference_mode():
        spec_opinion = opinion_from_alpha(model(torch.from_numpy(audio).unsqueeze(0)))
    return f"{decision(spec_opinion)}\n{_format_opinion('Audio opinion', spec_opinion)}"


def deepfakes_video_predict(input_video: str | None) -> str:
    if not input_video:
        raise ValueError("A video file is required.")
    frames = preprocess_video(input_video)
    if not frames:
        raise ValueError("No readable frames were found in the video.")
    audio_path = extract_video_audio(input_video)
    try:
        opinions = _predict_etmc_opinions(frames, load_audio(audio_path))
    finally:
        Path(audio_path).unlink(missing_ok=True)
    visual, audio, pseudo, fused = opinions
    return "\n".join([
        f"Fused result: {decision(fused)}",
        _format_opinion("Visual opinion", visual),
        _format_opinion("Audio opinion", audio),
        _format_opinion("Pseudo-view opinion", pseudo),
        _format_opinion("Fused opinion", fused),
        "Fusion: trained ETMC Dempster-Shafer fusion.",
    ])
