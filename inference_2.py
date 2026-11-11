"""Inference for the two supported modes: multimodal video and standalone audio.

The committed checkpoint predates the evidential heads.  Its branch logits are
therefore converted to evidence at inference time; ``load_complete_etmc`` is
provided for a future checkpoint containing trained heads and fusion weights.
"""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from models.evidential import (
    Opinion,
    decision,
    fuse_opinions,
    opinion_from_log_probabilities,
    opinion_from_probabilities,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "model.pth"
# The exported EfficientNet graph has a dynamic batch and fixed 224x224 NHWC
# input.  This is the runtime contract of the committed visual checkpoint.
VIDEO_SIZE = 224
AUDIO_RATE = 16_000
AUDIO_LENGTH = 64_600

# The saved visual classifier historically used index 0 for REAL, while the
# saved RawNet inference used index 1 for REAL.  Both are remapped to this
# canonical order before evidence is fused.
VISUAL_CLASS_ORDER = (0, 1)  # REAL, FAKE
AUDIO_CLASS_ORDER = (1, 0)  # REAL, FAKE

AUDIO_ARGS = {
    "nb_samp": AUDIO_LENGTH,
    "first_conv": 1024,
    "in_channels": 1,
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 2,
    "device": "cpu",
    "pretrained_audio_encoder": False,
    "freeze_audio_encoder": False,
}


class ModelArgs:
    """Minimal attribute container expected by the legacy RawNet implementation."""

    def __init__(self) -> None:
        self.__dict__.update(AUDIO_ARGS)


class AudioStreamUnavailable(ValueError):
    """The input media has no audio stream to extract."""


class AudioExtractionError(ValueError):
    """An audio stream exists or may exist, but extraction failed."""


def _canonicalize_logits(logits: torch.Tensor, class_order: tuple[int, int]) -> torch.Tensor:
    return logits[..., list(class_order)]


@functools.lru_cache(maxsize=1)
def _load_audio_model() -> Any:
    try:
        from models import image
    except ImportError as error:
        raise RuntimeError("Audio inference requires the RawNet model dependencies.") from error

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    if not isinstance(checkpoint, dict) or not {"rgb_encoder", "spec_encoder"} <= checkpoint.keys():
        raise RuntimeError("The checkpoint does not contain the saved visual and audio encoders.")

    audio = image.RawNet(ModelArgs())
    audio.load_state_dict(checkpoint["spec_encoder"], strict=True)
    audio.eval()
    return audio


@functools.lru_cache(maxsize=1)
def _load_visual_model() -> Any:
    try:
        import onnx
        from onnx2pytorch import ConvertModel
    except ImportError as error:
        raise RuntimeError(
            "Video inference requires the onnx and onnx2pytorch dependencies."
        ) from error

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    if not isinstance(checkpoint, dict) or not {"rgb_encoder", "spec_encoder"} <= checkpoint.keys():
        raise RuntimeError("The checkpoint does not contain the saved visual and audio encoders.")

    visual = ConvertModel(onnx.load(str(ROOT / "checkpoints" / "efficientnet.onnx")))
    visual.load_state_dict(checkpoint["rgb_encoder"], strict=True)
    visual.eval()
    return visual


def load_complete_etmc(path: str | os.PathLike[str]) -> Any:
    """Load a future complete ETMC checkpoint without overwriting old weights."""
    checkpoint = torch.load(path, map_location="cpu")
    required = {"model_state", "class_mapping", "architecture"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError("Complete ETMC checkpoint must include model_state, class_mapping, and architecture.")
    return checkpoint


def _resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == AUDIO_RATE:
        return audio
    if audio.size == 0:
        return audio
    output_length = max(1, round(audio.size * AUDIO_RATE / source_rate))
    old_x = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    new_x = np.linspace(0.0, 1.0, output_length, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def _prepare_audio(audio: np.ndarray, rate: int) -> np.ndarray:
    """Mono-mix, resample, normalize, and length-normalize a waveform."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = _resample(audio, int(rate))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    if audio.size < AUDIO_LENGTH:
        audio = np.pad(audio, (0, AUDIO_LENGTH - audio.size))
    else:
        audio = audio[:AUDIO_LENGTH]
    return audio.astype(np.float32)


def load_audio(path: str | os.PathLike[str]) -> np.ndarray:
    """Decode, mono-mix, resample, normalize, and length-normalize audio."""
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
    """Extract the original audio stream.

    Missing streams are reported separately from codec, I/O, and FFmpeg
    failures so callers can safely choose visual-only inference.
    """
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
            missing_stream = (
                "matches no streams" in stderr.lower()
                or "does not contain any stream" in stderr.lower()
            )
            if missing_stream:
                raise AudioStreamUnavailable("The video has no readable audio stream.")
            raise AudioExtractionError(
                f"Audio extraction failed: {stderr or 'FFmpeg returned an error.'}"
            )
        if not Path(output.name).is_file() or Path(output.name).stat().st_size == 0:
            raise AudioExtractionError("Audio extraction produced an empty file.")
        return output.name
    except Exception:
        Path(output.name).unlink(missing_ok=True)
        raise


def preprocess_video(input_video: str | os.PathLike[str], n_frames: int = 8) -> list[torch.Tensor]:
    """Sample full frames using the visual checkpoint's 224px preprocessing."""
    capture = cv2.VideoCapture(str(input_video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        return []
    indices = set(np.linspace(0, frame_count - 1, min(n_frames, frame_count), dtype=int))
    frames: list[torch.Tensor] = []
    for index in range(frame_count):
        if not capture.grab():
            continue
        if index not in indices:
            continue
        ok, frame = capture.retrieve()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (VIDEO_SIZE, VIDEO_SIZE)).astype(np.float32) / 255.0
        # The exported EfficientNet ONNX graph consumes NHWC tensors.
        frames.append(torch.from_numpy(frame))
    capture.release()
    return frames


def _format_opinion(name: str, opinion: Opinion) -> str:
    expected = opinion.expected[0].tolist()
    belief = opinion.belief[0].tolist()
    uncertainty = float(opinion.uncertainty[0])
    entropy = float(opinion.entropy[0] / np.log(2.0))
    return (
        f"{name}: REAL expected={expected[0]:.3f}, FAKE expected={expected[1]:.3f}; "
        f"REAL belief={belief[0]:.3f}, FAKE belief={belief[1]:.3f}; "
        f"uncertainty={uncertainty:.3f}; normalized entropy={entropy:.3f}"
    )


def _predict_audio_opinion(audio: np.ndarray) -> Opinion:
    audio_model = _load_audio_model()
    with torch.inference_mode():
        log_probabilities = audio_model(torch.from_numpy(audio).unsqueeze(0))
    return opinion_from_log_probabilities(
        _canonicalize_logits(log_probabilities, AUDIO_CLASS_ORDER)
    )


def deepfakes_spec_predict(input_audio: tuple[np.ndarray, int] | str) -> str:
    """Predict standalone audio without invoking the visual branch."""
    if isinstance(input_audio, (str, os.PathLike)):
        audio = load_audio(input_audio)
    else:
        raw, rate = input_audio
        audio = _prepare_audio(raw, int(rate))
    opinion = _predict_audio_opinion(audio)
    return f"{decision(opinion)}\n{_format_opinion('Audio opinion', opinion)}"


def deepfakes_video_predict(input_video: str | None) -> str:
    """Analyze a video, falling back to visual-only inference when audio is absent."""
    if not input_video:
        raise ValueError("A video file is required.")
    frames = preprocess_video(input_video)
    if not frames:
        raise ValueError("No readable frames were found in the video.")

    visual_model = _load_visual_model()
    with torch.inference_mode():
        probabilities = visual_model(torch.stack(frames))
    visual_opinion = opinion_from_probabilities(
        _canonicalize_logits(probabilities, VISUAL_CLASS_ORDER).mean(dim=0, keepdim=True)
    )

    try:
        audio_path = extract_video_audio(input_video)
    except AudioStreamUnavailable:
        return "\n".join([
            f"Visual result: {decision(visual_opinion)}",
            _format_opinion("Visual opinion", visual_opinion),
            "Audio stream unavailable. Prediction is based on visual analysis only.",
        ])

    try:
        audio = load_audio(audio_path)
        audio_opinion = _predict_audio_opinion(audio)
        fused_opinion = fuse_opinions(visual_opinion, audio_opinion)
        return "\n".join([
            f"Fused result: {decision(fused_opinion)}",
            _format_opinion("Visual opinion", visual_opinion),
            _format_opinion("Audio opinion", audio_opinion),
            _format_opinion("Fused opinion", fused_opinion),
            "Fusion: Dempster-Shafer evidence combination (not probability averaging).",
        ])
    finally:
        Path(audio_path).unlink(missing_ok=True)
