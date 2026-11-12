"""
Full FakeAVCeleb evaluation for the existing repository checkpoint.

Expected layout:
datasets/
    FAKE/   # overall manipulated: FakeVideo-* and RealVideo-FakeAudio
    REAL/   # overall authentic: RealVideo-RealAudio
    fakeavceleb_balanced_300/  # ignored

The filename prefix determines modality labels:
    RealVideo-RealAudio -> visual=REAL, audio=REAL, overall=REAL
    RealVideo-FakeAudio -> visual=REAL, audio=FAKE, overall=FAKE
    FakeVideo-RealAudio -> visual=FAKE, audio=REAL, overall=FAKE
    FakeVideo-FakeAudio -> visual=FAKE, audio=FAKE, overall=FAKE

This evaluates the EXISTING visual and audio branches independently.
It does NOT evaluate ETMC multimodal fusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
EXPECTED_AUDIO_SAMPLES = 64600
AUDIO_SAMPLE_RATE = 16000
CATEGORY_RE = re.compile(
    r"^(FakeVideo|RealVideo)-(FakeAudio|RealAudio)_",
    flags=re.IGNORECASE,
)


def parse_category(filename: str):
    match = CATEGORY_RE.match(Path(filename).name)
    if not match:
        raise ValueError(
            "Filename does not begin with one of the expected categories: "
            "RealVideo-RealAudio_, RealVideo-FakeAudio_, "
            "FakeVideo-RealAudio_, FakeVideo-FakeAudio_."
        )
    video_kind, audio_kind = match.groups()
    visual_label = int(video_kind.lower() == "fakevideo")
    audio_label = int(audio_kind.lower() == "fakeaudio")
    overall_label = int(visual_label == 1 or audio_label == 1)
    return visual_label, audio_label, overall_label, f"{video_kind}-{audio_kind}"


def discover_dataset(dataset_root: Path):
    rows, errors = [], []
    for overall_folder, expected_overall in (("REAL", 0), ("FAKE", 1)):
        folder = dataset_root / overall_folder
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected directory not found: {folder}")
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            try:
                visual, audio, overall, category = parse_category(path.name)
                if overall != expected_overall:
                    errors.append({
                        "path": str(path),
                        "error": (
                            f"Folder/category mismatch: in {overall_folder}, "
                            f"but filename implies overall_label={overall}"
                        ),
                    })
                    continue
                actor_match = re.search(r"_(id\d+)_", path.name, flags=re.IGNORECASE)
                actor_id = actor_match.group(1) if actor_match else "unknown"
                rows.append({
                    "path": path.resolve(),
                    "relative_path": str(path.relative_to(dataset_root)),
                    "filename": path.name,
                    "actor_id": actor_id,
                    "visual_label": visual,
                    "audio_label": audio,
                    "overall_label": overall,
                    "category": category,
                })
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})

    if not rows:
        raise RuntimeError(f"No valid videos found under {dataset_root}")
    return rows, errors


def load_models(device_name: str):
    import torch

    # inference_2 parses command-line arguments at import time.
    old_argv, old_cwd = sys.argv[:], Path.cwd()
    try:
        sys.argv = ["inference_2.py"]
        os.chdir(PROJECT_ROOT)
        import inference_2
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    device = torch.device(device_name)
    inference_2.img_model.to(device).eval()
    inference_2.spec_model.to(device).eval()
    return inference_2, device


def predict_video(video_path: Path, inference_module, device):
    import torch

    frames = inference_module.preprocess_video(str(video_path), n_frames=3)
    if not frames:
        raise RuntimeError("Could not decode any sampled video frames")

    real_scores, fake_scores = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for frame in frames:
            output = inference_module.img_model(frame.to(device))
            scores = output.detach().float().cpu().numpy().reshape(-1)
            if scores.size < 2 or not np.isfinite(scores[:2]).all():
                raise RuntimeError(f"Invalid visual model output: {scores}")
            # Existing image branch convention: index 0 = REAL, index 1 = FAKE.
            real_scores.append(float(scores[0]))
            fake_scores.append(float(scores[1]))
    elapsed = time.perf_counter() - start

    real_score = float(np.mean(real_scores))
    fake_score = float(np.mean(fake_scores))
    predicted = 0 if real_score > 0.5 else 1
    return predicted, real_score, fake_score, elapsed, len(frames)


def extract_audio_16k(video_path: Path, ffmpeg_executable: str):
    """Decode embedded audio to mono 16 kHz float32 using FFmpeg."""
    command = [
        ffmpeg_executable, "-v", "error", "-i", str(video_path),
        "-map", "0:a:0", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg audio extraction failed: {message[-500:]}")
    waveform = np.frombuffer(completed.stdout, dtype="<f4").copy()
    if waveform.size == 0:
        raise RuntimeError("Video has no decodable audio stream")
    if not np.isfinite(waveform).all():
        raise RuntimeError("Extracted waveform contains NaN or infinite values")
    return waveform


def predict_audio(video_path: Path, inference_module, device, ffmpeg_executable: str):
    import torch

    waveform = extract_audio_16k(video_path, ffmpeg_executable)

    # Match the RawNet checkpoint's configured input length (nb_samp=64600).
    # Long files are center-cropped; short files are zero-padded.
    if waveform.size > EXPECTED_AUDIO_SAMPLES:
        start_index = (waveform.size - EXPECTED_AUDIO_SAMPLES) // 2
        waveform = waveform[start_index:start_index + EXPECTED_AUDIO_SAMPLES]
    elif waveform.size < EXPECTED_AUDIO_SAMPLES:
        missing = EXPECTED_AUDIO_SAMPLES - waveform.size
        left = missing // 2
        waveform = np.pad(waveform, (left, missing - left), mode="constant")

    waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)
    audio_tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        log_scores = inference_module.spec_model(audio_tensor)
        scores = torch.exp(log_scores).detach().float().cpu().numpy().reshape(-1)
    elapsed = time.perf_counter() - start

    if scores.size < 2 or not np.isfinite(scores[:2]).all():
        raise RuntimeError(f"Invalid RawNet output: {scores}")

    # Repository RawNet convention: class 0 = FAKE, class 1 = REAL.
    fake_score = float(scores[0])
    real_score = float(scores[1])
    predicted = 1 if fake_score >= real_score else 0
    return predicted, real_score, fake_score, elapsed, waveform.size


def metric_bundle(records):
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

    y_true = np.asarray([r["ground_truth_id"] for r in records], dtype=int)
    y_pred = np.asarray([r["predicted_id"] for r in records], dtype=int)
    fake_scores = np.asarray([r["fake_score"] for r in records], dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def div(a, b):
        return float(a / b) if b else None

    return {
        "sample_count": int(len(records)),
        "class_distribution": {
            "real": int(np.sum(y_true == 0)),
            "fake": int(np.sum(y_true == 1)),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_real": div(tn, tn + fp),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": div(fp, fp + tn),
        "false_negative_rate": div(fn, fn + tp),
        "roc_auc": (
            float(roc_auc_score(y_true, fake_scores))
            if len(np.unique(y_true)) == 2 else None
        ),
        "confusion_matrix": {
            "labels": ["REAL (0)", "FAKE (1)"],
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def save_plots(records, metrics, modality, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    cm = metrics["confusion_matrix"]
    matrix = [
        [cm["true_negative"], cm["false_positive"]],
        [cm["false_negative"], cm["true_positive"]],
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"],
        xlabel="Predicted label", ylabel="True label",
        title=f"{modality.title()} confusion matrix",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / f"{modality}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    y_true = np.asarray([r["ground_truth_id"] for r in records])
    scores = np.asarray([r["fake_score"] for r in records])
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, scores)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(
            xlabel="False Positive Rate", ylabel="True Positive Rate",
            title=f"{modality.title()} ROC curve",
        )
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / f"{modality}_roc_curve.png", dpi=180)
        plt.close(fig)


def evaluate_branch(rows, branch, inference_module, device, ffmpeg, output_dir):
    records, skipped = [], []
    start_total = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        try:
            if branch == "video":
                pred, real_score, fake_score, elapsed, detail = predict_video(
                    row["path"], inference_module, device
                )
                truth = row["visual_label"]
            else:
                pred, real_score, fake_score, elapsed, detail = predict_audio(
                    row["path"], inference_module, device, ffmpeg
                )
                truth = row["audio_label"]

            records.append({
                "filename": row["relative_path"],
                "category": row["category"],
                "actor_id": row["actor_id"],
                "ground_truth_id": truth,
                "ground_truth_label": "FAKE" if truth else "REAL",
                "predicted_id": pred,
                "predicted_label": "FAKE" if pred else "REAL",
                "real_score": real_score,
                "fake_score": fake_score,
                "correct": bool(pred == truth),
                "inference_time_seconds": elapsed,
                "frames_sampled_or_audio_samples": detail,
            })
        except Exception as exc:
            skipped.append({"filename": row["relative_path"], "reason": str(exc)})

        if index % 50 == 0 or index == len(rows):
            print(
                f"[{branch}] {index}/{len(rows)} processed; "
                f"successful={len(records)}, skipped={len(skipped)}"
            )

    if not records:
        raise RuntimeError(f"No {branch} samples evaluated. First errors: {skipped[:5]}")

    metrics = metric_bundle(records)
    metrics.update({
        "modality": branch,
        "inference_time_total_seconds": float(
            sum(r["inference_time_seconds"] for r in records)
        ),
        "average_inference_time_seconds": float(
            np.mean([r["inference_time_seconds"] for r in records])
        ),
        "wall_clock_evaluation_seconds": time.perf_counter() - start_total,
        "skipped_count": len(skipped),
        "skipped_samples": skipped,
        "ground_truth_column": "visual_label" if branch == "video" else "audio_label",
        "model_scope": (
            "Existing EfficientNet visual branch; three sampled frames per video"
            if branch == "video"
            else "Existing RawNet audio branch; embedded audio decoded to mono 16 kHz"
        ),
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{branch}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    with (output_dir / f"{branch}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_plots(records, metrics, branch, output_dir)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=PROJECT_ROOT / "datasets",
        help="Directory containing FAKE/ and REAL/ folders.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "evaluation_results_full",
    )
    parser.add_argument(
        "--modality", choices=["video", "audio", "both"], default="both"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Optional smoke test limit; 0 evaluates all discovered samples.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = None
    if args.modality in ("audio", "both"):
        import shutil
        ffmpeg_path = shutil.which(args.ffmpeg)
        if ffmpeg_path is None:
            raise FileNotFoundError(
                "FFmpeg executable not found. Install FFmpeg and ensure ffmpeg.exe "
                "is on PATH, then run `ffmpeg -version`."
            )

    rows, discovery_errors = discover_dataset(dataset_root)
    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    print(f"Dataset root: {dataset_root}")
    print(f"Discovered valid videos: {len(rows)}")
    print("Category counts:")
    for category in sorted({r["category"] for r in rows}):
        print(f"  {category}: {sum(r['category'] == category for r in rows)}")
    print(f"Discovery errors: {len(discovery_errors)}")

    # Save the derived modality labels so the mapping is auditable.
    with (output_dir / "derived_dataset_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "relative_path", "category", "actor_id",
            "visual_label", "audio_label", "overall_label",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})

    with (output_dir / "discovery_errors.json").open("w", encoding="utf-8") as f:
        json.dump(discovery_errors, f, indent=2)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    inference_module, device = load_models(device)
    print(f"Using device: {device}")

    results = {}
    if args.modality in ("video", "both"):
        results["video"] = evaluate_branch(
            rows, "video", inference_module, device, ffmpeg_path, output_dir
        )
    if args.modality in ("audio", "both"):
        results["audio"] = evaluate_branch(
            rows, "audio", inference_module, device, ffmpeg_path, output_dir
        )

    summary = {
        "dataset_root": str(dataset_root),
        "discovered_samples": len(rows),
        "category_distribution": {
            category: sum(r["category"] == category for r in rows)
            for category in sorted({r["category"] for r in rows})
        },
        "device": str(device),
        "evaluation_scope": "full dataset folder scan; not automatically an independent test set",
        "warning": (
            "These are independent visual-branch and audio-branch evaluations. "
            "They are NOT fused ETMC predictions. Confirm the checkpoint's training "
            "data before describing these results as held-out test performance."
        ),
        "results": results,
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation completed.")
    print(f"Results: {output_dir}")
    for branch, metrics in results.items():
        print(
            f"{branch}: n={metrics['sample_count']} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision_fake']:.4f} "
            f"recall={metrics['recall_fake_sensitivity']:.4f} "
            f"F1={metrics['f1_fake']:.4f} "
            f"AUC={metrics['roc_auc']}"
        )


if __name__ == "__main__":
    main()
