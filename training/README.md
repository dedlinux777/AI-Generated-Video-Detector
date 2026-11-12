# ETMC audiovisual training in Google Colab

This workflow prepares an actor/source grouped dataset, trains ETMC with separate visual/audio/overall labels, and evaluates a saved checkpoint on the untouched test split. The learned pseudo-view is built from both embeddings and is not an independent physical sensor.

## Colab setup

Use a GPU runtime when available. Install FFmpeg from apt, while allowing Colab's compatible CUDA PyTorch build to remain installed:

```python
!git clone https://github.com/dedlinux777/AI-Generated-Video-Detector.git
%cd AI-Generated-Video-Detector
!pip install -r requirements-training.txt
!apt-get -qq update && apt-get -qq install -y ffmpeg
from google.colab import drive
drive.mount('/content/drive')
```

If Colab's torch/torchvision versions are incompatible, select a matching CUDA build from the official PyTorch install selector, then rerun requirements. TensorFlow is not needed.

## Drive layout

`--dataset-root` and `--manifest` are configurable. Example:

```text
/content/drive/MyDrive/dataset/
├── FAKE/
├── REAL/
├── etmc_multimodal_manifest.csv
└── manifest.csv                  # optional legacy manifest
```

Use Drive locations for `--output-root`, `--output-dir`, and evaluation output if results must survive a Colab reset.

## Execution order

### 1. Validate, split, and cache

```python
!python training/01_prepare_data.py \
  --dataset-root "/content/drive/MyDrive/dataset" \
  --manifest "/content/drive/MyDrive/dataset/etmc_multimodal_manifest.csv" \
  --output-root "/content/drive/MyDrive/etmc_run/prepared_dataset" \
  --frames 3 --audio-seconds 3 --seed 42
```

The script detects common path/label/category/identity aliases. Invalid rows and duplicate paths are recorded in `data_validation_report.json` before exclusion from split files. Frames are evenly sampled from the center window, converted BGR→RGB, face-cropped with MTCNN when available, resized to 256×256, and cached on disk. If face detection fails, a centered square crop is used. At load time, RGB is normalized with timm's B4 NS data config (mean 0.485/0.456/0.406, std 0.229/0.224/0.225). Audio is read from `audio_path` when decodable; otherwise FFmpeg extracts it from the paired video using the same center-window offset. It is converted to mono, resampled to 16 kHz, and fixed to three seconds (48,000 samples), padding shorter clips. Corrupt or silent audio is explicitly reported.

Splits use `actor_id`, then an available source-identity column, then a filename-derived key. The report records the grouping choice and overlap. If source identities cannot be resolved reliably, identity-independent evaluation cannot be claimed. Group constraints take precedence over exact proportions and category balance.

### 2. Train ETMC

```python
!python training/02_train_etmc.py \
  --prepared-root "/content/drive/MyDrive/etmc_run/prepared_dataset" \
  --checkpoint "checkpoints/model.pth" \
  --output-dir "/content/drive/MyDrive/etmc_run/outputs" \
  --epochs 10 --batch-size 4 --max-runtime-minutes 120
```

Optional: `--resume .../latest_etmc_checkpoint.pth`, `--max-steps N`, `--gradient-accumulation N`; `--smoke-test` exercises synthetic forward/backward and checkpoint reload without dataset access. Training uses two stages, head/backbone learning rates of 1e-4/1e-5, AdamW, branch-specific class weights computed from training labels, AMP on CUDA, gradient clipping, and validation fused F1 early stopping.

Outputs: `best_etmc_model.pth`, `latest_etmc_checkpoint.pth`, `training_history.csv/json`, `training_summary.json`, and `config.json`. The checkpoints include model, optimizer/scheduler state, epoch, best validation score, class mapping, preprocessing, threshold metadata, and seed.

### 3. Evaluate the test split

```python
!python training/03_evaluate_etmc.py \
  --prepared-root "/content/drive/MyDrive/etmc_run/prepared_dataset" \
  --checkpoint "/content/drive/MyDrive/etmc_run/outputs/best_etmc_model.pth" \
  --output-dir "/content/drive/MyDrive/etmc_run/evaluation_results"
```

Default threshold is 0.5. ROC-AUC uses fake-class probability. Modes include visual-only (visual labels), audio-only (audio labels), pseudo-view, visual+audio DS fusion without pseudo, and full ETMC fusion (overall labels). `--ambiguity-quantile 0.95` derives an uncertainty cutoff from validation data and emits REAL/FAKE/AMBIGUOUS decisions; it never uses test labels to set the cutoff.

## Checkpoint compatibility

`checkpoints/model.pth` contains a PyTorch RawNet `spec_encoder`, whose compatible tensors are loaded into the repository RawNet feature encoder; missing/incompatible tensors are reported. Its `rgb_encoder` is a TensorFlow EfficientNetV2-B0 ONNX initializer map and cannot load into PyTorch timm EfficientNet-B4. That branch is explicitly ignored and B4 uses timm pretrained weights instead. The original checkpoint is never overwritten. The local workspace has no new manifest or Drive media, so actual CSV/media validation must run in Colab.

The legacy `models/TMC.py` remains available to existing app/inference code. Its image wrapper returns a sigmoid scalar and its RawNet wrapper returns class log probabilities, while its ETMC heads are parameterized as if both were feature embeddings; it also depends on hard-coded local pretrained paths. The Colab training model therefore uses the same repository EfficientNet-B4 timm family and RawNet acoustic feature extractor with explicit embeddings and compatible evidential heads. `models/image.py` only gains a `forward_features()` split; its existing `RawNet.forward()` output behavior is unchanged.

## Hardware and smoke checks

B4, RawNet, and 3×256px frames require a GPU for practical training. Start with 16 GB VRAM and batch size 2–4; lower batch size to 1 and use gradient accumulation if needed. CPU supports smoke checks but is not practical for the full dataset. Drive decoding and caching overhead varies; a full run may take longer than two hours. No runtime guarantee is made. Data arrays are loaded from disk per sample rather than holding all media in RAM.

```python
!python training/01_prepare_data.py --help
!python training/02_train_etmc.py --help
!python training/03_evaluate_etmc.py --help
!python training/02_train_etmc.py --smoke-test --output-dir /tmp/etmc_smoke
```

EfficientNet attempts to load timm pretrained B4 NS weights; if unavailable, it logs a warning and records the random-initialization fallback in `config.json`. The provided RawNet uses adaptive pooling and accepts varying waveform lengths; this workflow uses fixed-length 16 kHz mono waveforms for reproducible batching.
