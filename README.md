# AI-Generated Video and Audio Detector

A research and educational application for two independent tasks: detecting manipulated videos and detecting synthetic audio. Each mode accepts its own input independently.

## Detection Workflows

- **Video Detection:** OpenCV decodes an uploaded video, samples frames, converts them from BGR to RGB, resizes and normalizes them, and passes each frame to the EfficientNet-based visual encoder. Frame predictions are averaged to produce a REAL or FAKE video result.
- **Audio Detection:** Uploaded or microphone-recorded audio is decoded, converted to mono, and passed as a raw waveform to the existing RawNet classifier for a REAL or FAKE audio result.

The EfficientNet encoder is converted from `checkpoints/efficientnet.onnx` and its trained weights are loaded from `checkpoints/model.pth`. RawNet uses the audio encoder weights from the same checkpoint. The two inference workflows run independently; no audio-video synchronization analysis is performed.

The ETMC training architecture contains visual and audio branches with fusion layers; the application loads and invokes each trained branch independently for its corresponding input mode.

Predictions are research results, not proof of authenticity.

## Requirements

- Python 3.10 or newer.
- CPU or CUDA-capable PyTorch, depending on your hardware.
- Git LFS to download the trained checkpoint files when they are stored as LFS objects.

## Installation

```powershell
git clone https://github.com/dedlinux777/AI-Generated-Video-Detector.git
cd AI-Generated-Video-Detector

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Git LFS is installed, fetch the model files with:

```powershell
git lfs install
git lfs pull
```

The application expects these files in `checkpoints/`:

- `model.pth`
- `efficientnet.onnx`

## Run

```powershell
python app.py
```

Open the local Gradio URL shown in the terminal. The application contains **Video Detection** and **Audio Detection** tabs. Audio accepts uploads and microphone recordings.

## Supported Inputs

- **Video:** MP4, AVI, MOV, MKV, WebM, and FLV, subject to the installed OpenCV and Gradio decoders.
- **Audio:** WAV, MP3, FLAC, OGG, AAC, M4A, and other formats that `soundfile` or `torchaudio` can decode.

## Static Frontend and Legacy API

`frontendhtml/` is a separate static frontend; `python app.py` launches the Gradio application and does not serve these files. The static page posts video/audio uploads to the legacy Flask API configured in `frontendhtml/script.js`. That backend forwards requests to its configured Hugging Face Space and reports service failures instead of returning demo predictions. Configure the API URL for the environment where the static page and backend are deployed.

## Project Layout

| Path | Purpose |
| --- | --- |
| `app.py` | Gradio video/audio application entry point |
| `inference_2.py` | EfficientNet frame and RawNet audio inference |
| `models/` | Visual encoder, RawNet, and ETMC model components |
| `checkpoints/` | Trained model weights and EfficientNet ONNX model |
| `audios/`, `videos/` | Sample media |
| `frontendhtml/` | Independent static frontend |
| `gradioapibackend.py` | Legacy Flask API used by the static frontend |

## Notes

- `static_ffmpeg` configures bundled FFmpeg binaries for media decoding.
- Do not upload private or sensitive media when using a shared deployment.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).