"""Reproducible, sample-level evaluation for the existing detector.

The evaluator deliberately uses the RGB and RawNet encoders loaded by
``inference_2.py``. It does not instantiate a replacement model or alter the
ETMC implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
LABEL_TO_ID = {"fake": 1, "real": 0}
VIDEO_EXTENSIONS = {".avi", ".flv", ".mkv", ".mov", ".mp4", ".webm"}
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}


def set_seed(seed: int) -> None:
    """Set deterministic seeds for the Python, NumPy, and Torch runtimes."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def normalize_label(value: Any) -> int:
    """Map the repository's real/fake labels to real=0 and fake=1."""
    label = str(value).strip().lower()
    if label not in LABEL_TO_ID:
        raise ValueError(
            f"Unsupported label {value!r}; expected one of {sorted(LABEL_TO_ID)}"
        )
    return LABEL_TO_ID[label]


def _column_name(columns: Iterable[str], candidates: Iterable[str]) -> str:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        "Could not identify a required manifest column. "
        f"Available columns: {sorted(lowered.values())}; expected one of {list(candidates)}"
    )


def read_manifest(path: Path, modality: str) -> list[dict[str, Any]]:
    """Read a labelled CSV after explicitly identifying its path and label columns."""
    import pandas as pd

    frame = pd.read_csv(path)
    path_column = _column_name(
        frame.columns,
        (
            "video_path" if modality == "video" else "audio_path",
            "file_path",
            "filepath",
            "filename",
            "file",
            "path",
            "name",
        ),
    )
    label_column = _column_name(frame.columns, ("label", "class", "target", "ground_truth"))
    rows = []
    for index, row in frame.iterrows():
        rows.append(
            {
                "filename": str(row[path_column]),
                "label": normalize_label(row[label_column]),
                "manifest_row": int(index) + 2,
            }
        )
    if not rows:
        raise ValueError(f"Manifest contains no samples: {path}")
    return rows


def resolve_manifest_path(data_dir: Path, csv_path: Path | None) -> tuple[Path, Path]:
    """Resolve a manifest and its data root without guessing among multiple CSVs."""
    if csv_path is not None:
        manifest = csv_path.resolve()
        root = data_dir.resolve() if data_dir.is_dir() else manifest.parent
        return manifest, root
    if data_dir.suffix.lower() == ".csv":
        return data_dir.resolve(), data_dir.parent.resolve()
    candidates = sorted(data_dir.glob("*.csv"))
    if len(candidates) != 1:
        raise ValueError(
            "A labelled CSV is required. Pass --csv explicitly when the data directory "
            f"contains zero or multiple CSV files (found {len(candidates)})."
        )
    return candidates[0].resolve(), data_dir.resolve()


def resolve_sample_path(filename: str, data_root: Path, manifest: Path) -> Path:
    raw = Path(filename.replace("\\", os.sep))
    candidates = [raw if raw.is_absolute() else data_root / raw, manifest.parent / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def reject_training_path(path: Path) -> None:
    """Prevent an obvious training directory from being evaluated as a test set."""
    if any(part.lower() in {"train", "training"} for part in path.resolve().parts):
        raise ValueError(
            f"Refusing to evaluate a path that appears to be training data: {path}. "
            "Provide an independent labelled evaluation split."
        )


def discover_audio_rows(data_dir: Path, csv_path: Path | None) -> tuple[list[dict[str, Any]], Path]:
    """Read audio CSV metadata, or class-labelled real/fake subdirectories."""
    if csv_path is not None or data_dir.suffix.lower() == ".csv":
        manifest, root = resolve_manifest_path(data_dir, csv_path)
        return read_manifest(manifest, "audio"), root
    if not data_dir.is_dir():
        raise ValueError(f"Audio dataset directory does not exist: {data_dir}")
    rows = []
    for label_name, label_id in LABEL_TO_ID.items():
        label_dir = data_dir / label_name
        if label_dir.is_dir():
            for path in sorted(label_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                    rows.append({"filename": str(path.relative_to(data_dir)), "label": label_id})
    if not rows:
        raise ValueError(
            "No labelled audio samples found. Provide --audio_csv with an explicit "
            "path/label manifest or use real/ and fake/ subdirectories."
        )
    return rows, data_dir.resolve()


def load_inference_models(device_name: str):
    """Load the unchanged application models through inference_2.py."""
    import torch

    device = torch.device(device_name)
    original_argv = sys.argv
    original_cwd = Path.cwd()
    try:
        sys.argv = ["inference_2.py"]
        os.chdir(PROJECT_ROOT)
        import inference_2
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)
    inference_2.img_model.to(device).eval()
    inference_2.spec_model.to(device).eval()
    return inference_2, device


def predict_video(path: Path, inference_module: Any, device: Any) -> tuple[int, float, float, float]:
    import torch

    frames = inference_module.preprocess_video(str(path))
    if not frames:
        raise ValueError("No readable frames were extracted")
    real_scores = []
    fake_scores = []
    started = time.perf_counter()
    with torch.inference_mode():
        for frame in frames:
            output = inference_module.img_model(frame.to(device))
            scores = np.squeeze(output.detach().cpu().numpy())
            if scores.size < 2 or not np.isfinite(scores[:2]).all():
                raise ValueError("Model returned invalid video scores")
            real_scores.append(float(scores[0]))
            fake_scores.append(float(scores[1]))
    elapsed = time.perf_counter() - started
    real_score = float(np.mean(real_scores))
    fake_score = float(np.mean(fake_scores))
    # This is the application's existing frame aggregation and threshold.
    predicted = 0 if real_score > 0.5 else 1
    return predicted, real_score, fake_score, elapsed


def load_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, _ = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio file contains no usable mono waveform")
    return audio


def predict_audio(path: Path, inference_module: Any, device: Any) -> tuple[int, float, float, float]:
    import torch

    audio = inference_module.preprocess_audio(load_audio(path)).to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        log_scores = inference_module.spec_model(audio)
        scores = torch.exp(log_scores).detach().cpu().numpy().reshape(-1)
    elapsed = time.perf_counter() - started
    if scores.size < 2 or not np.isfinite(scores[:2]).all():
        raise ValueError("Model returned invalid audio scores")
    fake_score, real_score = float(scores[0]), float(scores[1])
    predicted = 1 if fake_score >= real_score else 0
    return predicted, real_score, fake_score, elapsed


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate binary metrics with null values when their denominators are invalid."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

    y_true = np.asarray([row["ground_truth_id"] for row in rows], dtype=int)
    y_pred = np.asarray([row["predicted_id"] for row in rows], dtype=int)
    scores = np.asarray([row["fake_score"] for row in rows], dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    result: dict[str, Any] = {
        "positive_class": "fake (1); real is negative (0)",
        "sample_count": int(len(rows)),
        "correct_count": int(np.sum(y_true == y_pred)),
        "incorrect_count": int(np.sum(y_true != y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(rows) else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if (tp + fp) else None,
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)) if (tp + fn) else None,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)) if (2 * tp + fp + fn) else None,
        "specificity": ratio(tn, tn + fp),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
        "confusion_matrix": {
            "labels": ["real (0)", "fake (1)"],
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "roc_auc": None,
        "class_distribution": {
            "real (0)": int(np.sum(y_true == 0)),
            "fake (1)": int(np.sum(y_true == 1)),
        },
    }
    if len(np.unique(y_true)) == 2 and np.isfinite(scores).all():
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
    return result


def save_confusion_plot(metrics: dict[str, Any], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    cm = metrics["confusion_matrix"]
    matrix = [[cm["true_negative"], cm["false_positive"]], [cm["false_negative"], cm["true_positive"]]]
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])
    for row_index in range(2):
        for column_index in range(2):
            plt.text(column_index, row_index, matrix[row_index][column_index],
                     ha="center", va="center")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_roc_plot(rows: list[dict[str, Any]], path: Path, title: str) -> bool:
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    y_true = np.asarray([row["ground_truth_id"] for row in rows], dtype=int)
    scores = np.asarray([row["fake_score"] for row in rows], dtype=float)
    if len(np.unique(y_true)) != 2 or not np.isfinite(scores).all():
        return False
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(5, 4))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {calculate_metrics(rows)['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def evaluate_modality(
    modality: str,
    rows: list[dict[str, Any]],
    root: Path,
    output_dir: Path,
    inference_module: Any,
    device: Any,
    checkpoint_path: Path,
    dataset_path: Path,
    evaluation_set_type: str = "unknown",
) -> dict[str, Any]:
    predictions = []
    skipped = []
    total_started = time.perf_counter()
    predictor = predict_video if modality == "video" else predict_audio
    for row in rows:
        sample_path = resolve_sample_path(row["filename"], root, dataset_path)
        try:
            predicted, real_score, fake_score, inference_time = predictor(sample_path, inference_module, device)
            predictions.append({
                "filename": row["filename"],
                "ground_truth_label": "fake" if row["label"] else "real",
                "predicted_label": "fake" if predicted else "real",
                "real_score": real_score,
                "fake_score": fake_score,
                "correct": bool(predicted == row["label"]),
                "inference_time_seconds": inference_time,
                "ground_truth_id": row["label"],
                "predicted_id": predicted,
            })
        except Exception as exc:
            skipped.append({"filename": row["filename"], "reason": str(exc)})
    if not predictions:
        raise RuntimeError(f"No valid {modality} samples were evaluated; skipped={len(skipped)}")

    metrics = calculate_metrics(predictions)
    metrics.update({
        "modality": modality,
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_path.stem,
        "evaluation_set_type": evaluation_set_type,
        "device": str(device),
        "configuration": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": __import__("torch").__version__,
            "preprocessing_module": "inference_2.py",
            "video_frames_sampled": 3 if modality == "video" else None,
            "audio_preprocessing": "inference_2.preprocess_audio (waveform, batch dimension only)",
            "aggregation": "mean frame scores; real if mean real score > 0.5" if modality == "video" else "RawNet log-softmax scores; argmax",
        },
        "evaluation_time_seconds": time.perf_counter() - total_started,
        "inference_time_seconds_total": float(sum(row["inference_time_seconds"] for row in predictions)),
        "average_inference_time_seconds": float(np.mean([row["inference_time_seconds"] for row in predictions])),
        "skipped_count": len(skipped),
        "skipped_samples": skipped,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{modality}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "Filename", "Ground-truth label", "Predicted label",
            "Real-class score/probability", "Fake-class score/probability",
            "Correct/incorrect", "Inference time",
        ])
        writer.writeheader()
        for row in predictions:
            writer.writerow({
                "Filename": row["filename"],
                "Ground-truth label": row["ground_truth_label"],
                "Predicted label": row["predicted_label"],
                "Real-class score/probability": row["real_score"],
                "Fake-class score/probability": row["fake_score"],
                "Correct/incorrect": "correct" if row["correct"] else "incorrect",
                "Inference time": row["inference_time_seconds"],
            })
    with (output_dir / f"{modality}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    save_confusion_plot(metrics, output_dir / f"{modality}_confusion_matrix.png", f"{modality.title()} confusion matrix")
    save_roc_plot(predictions, output_dir / f"{modality}_roc_curve.png", f"{modality.title()} ROC curve")
    return metrics


def write_summary(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "Evaluation summary",
        "Positive class: fake (1); negative class: real (0)",
        "Metrics are omitted when a valid labelled evaluation sample is unavailable.",
        "",
    ]
    for modality, metrics in results.items():
        lines.extend([
            f"{modality.title()} evaluation",
            f"Dataset: {metrics['dataset_name']} ({metrics['evaluation_set_type']})",
            f"Samples: {metrics['sample_count']} (correct {metrics['correct_count']}, incorrect {metrics['incorrect_count']})",
            f"Class distribution: {metrics['class_distribution']}",
            f"Accuracy: {metrics['accuracy']}",
            f"Precision: {metrics['precision']}",
            f"Recall/sensitivity: {metrics['recall_sensitivity']}",
            f"Specificity: {metrics['specificity']}",
            f"F1-score: {metrics['f1_score']}",
            f"ROC-AUC: {metrics['roc_auc']}",
            f"False positive rate: {metrics['false_positive_rate']}",
            f"False negative rate: {metrics['false_negative_rate']}",
            f"Average inference time (seconds): {metrics['average_inference_time_seconds']}",
            f"Skipped samples: {metrics['skipped_count']}",
            "",
        ])
    (output_dir / "evaluation_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_evaluation_report(
    results: dict[str, dict[str, Any]],
    output_dir: Path,
    status: str,
    evaluation_set_type: str,
) -> None:
    """Write a report without manufacturing metrics when evaluation is unavailable."""
    lines = [
        "# Performance Evaluation Report",
        "",
        f"**Status:** {status}",
        f"**Evaluation-set designation:** {evaluation_set_type}",
        "**Positive class:** fake (1); real is negative (0).",
        "",
        "## Methodology",
        "",
        "The existing application checkpoint is evaluated without retraining or "
        "architectural changes. Video inference samples frames with the existing "
        "`inference_2.preprocess_video` function, obtains one RGB prediction per "
        "frame, averages frame scores, and records one final prediction per video. "
        "Audio inference loads one waveform per file, applies the existing "
        "`preprocess_audio` batch-dimension conversion, and records one RawNet "
        "prediction per audio file.",
        "",
        "Metrics are calculated at the complete-file level using scikit-learn. "
        "Frame rows are never treated as independent video samples.",
        "",
        "## Dataset and model",
        "",
        "The repository contains FakeAVCeleb CSV manifests with `video_path` and "
        "`label` columns, but the referenced media is not present in this checkout. "
        "No labelled standalone-audio manifest is included. Therefore, an "
        "independent test-set result cannot be claimed from this checkout.",
        "",
        f"Checkpoint: `{PROJECT_ROOT / 'checkpoints' / 'model.pth'}`",
        "",
    ]
    if results:
        lines.extend(["## Results", ""])
        for modality, metrics in results.items():
            cm = metrics["confusion_matrix"]
            lines.extend([
                f"### {modality.title()}",
                f"- Dataset: `{metrics['dataset_name']}` ({metrics['evaluation_set_type']})",
                f"- Samples: {metrics['sample_count']}; correct: {metrics['correct_count']}; incorrect: {metrics['incorrect_count']}",
                f"- Class distribution: {metrics['class_distribution']}",
                f"- Accuracy: {metrics['accuracy']}",
                f"- Precision: {metrics['precision']}",
                f"- Recall / sensitivity: {metrics['recall_sensitivity']}",
                f"- Specificity: {metrics['specificity']}",
                f"- F1-score: {metrics['f1_score']}",
                f"- False-positive rate: {metrics['false_positive_rate']}",
                f"- False-negative rate: {metrics['false_negative_rate']}",
                f"- ROC-AUC: {metrics['roc_auc']}",
                f"- Confusion matrix (TN, FP, FN, TP): ({cm['true_negative']}, {cm['false_positive']}, {cm['false_negative']}, {cm['true_positive']})",
                f"- Total evaluation time: {metrics['evaluation_time_seconds']} seconds",
                f"- Average inference time per sample: {metrics['average_inference_time_seconds']} seconds",
                "",
            ])
    else:
        lines.extend([
            "## Results",
            "",
            "No quantitative results were generated because no valid labelled "
            "evaluation samples were available. This is an evaluation limitation, "
            "not a zero-valued experiment.",
            "",
        ])
    lines.extend([
        "## Limitations",
        "",
        "- The bundled manifests reference external FakeAVCeleb media that is not "
        "present locally.",
        "- The repository does not provide a labelled standalone-audio test manifest.",
        "- No result should be described as independent-test, validation, or demo "
        "performance until the corresponding labelled split is supplied and the "
        "command is rerun with `--evaluation-set-type`.",
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("video", "audio", "both"), required=True)
    parser.add_argument("--data_dir", type=Path, help="Video/audio root or a labelled CSV")
    parser.add_argument("--csv", type=Path, help="Labelled video CSV (for --modality video)")
    parser.add_argument("--video_dir", type=Path)
    parser.add_argument("--video_csv", type=Path)
    parser.add_argument("--audio_dir", type=Path)
    parser.add_argument("--audio_csv", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "model.pth")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-set-type",
        choices=("independent_test", "validation", "demo", "unknown"),
        default="unknown",
        help="Dataset designation recorded in reports; never inferred from metrics.",
    )
    args = parser.parse_args()
    if args.modality == "video" and args.data_dir is None:
        args.data_dir = args.video_dir
        args.csv = args.video_csv
    if args.modality == "audio" and args.data_dir is None:
        args.data_dir = args.audio_dir
        args.csv = args.audio_csv
    return args


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        application_checkpoint = (PROJECT_ROOT / "checkpoints" / "model.pth").resolve()
        if args.checkpoint.resolve() != application_checkpoint:
            raise ValueError(
                f"This evaluator uses the application's checkpoint at {application_checkpoint}; "
                "a replacement checkpoint is not supported."
            )
        if not application_checkpoint.is_file():
            raise FileNotFoundError(f"Application checkpoint not found: {application_checkpoint}")
        inference_module, device = load_inference_models(device)
        if args.modality in ("video", "both"):
            video_dir = args.video_dir or args.data_dir
            video_csv = args.video_csv or args.csv
            if video_dir is None:
                raise ValueError("Video evaluation requires --data_dir or --video_dir")
            manifest, root = resolve_manifest_path(video_dir, video_csv)
            reject_training_path(manifest)
            reject_training_path(root)
            results["video"] = evaluate_modality(
                "video", read_manifest(manifest, "video"), root, args.output_dir,
                inference_module, device, args.checkpoint.resolve(), manifest,
                args.evaluation_set_type,
            )
        if args.modality in ("audio", "both"):
            audio_dir = args.audio_dir or args.data_dir
            audio_csv = args.audio_csv if args.modality == "both" else (args.audio_csv or args.csv)
            if audio_dir is None:
                raise ValueError("Audio evaluation requires --data_dir or --audio_dir")
            reject_training_path(audio_dir)
            rows, root = discover_audio_rows(audio_dir, audio_csv)
            dataset_path = audio_csv.resolve() if audio_csv else audio_dir.resolve()
            reject_training_path(dataset_path)
            results["audio"] = evaluate_modality(
                "audio", rows, root, args.output_dir, inference_module,
                device, args.checkpoint.resolve(), dataset_path,
                args.evaluation_set_type,
            )
        write_summary(results, args.output_dir)
        write_evaluation_report(
            results, args.output_dir, "completed", args.evaluation_set_type
        )
    except Exception as exc:
        write_summary({}, args.output_dir)
        write_evaluation_report(
            results, args.output_dir, f"not completed: {exc}", args.evaluation_set_type
        )
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Evaluation outputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
