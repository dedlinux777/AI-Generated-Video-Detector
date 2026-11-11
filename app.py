import static_ffmpeg

static_ffmpeg.add_paths()

import gradio as gr
import numpy as np
import inference_2 as inference


title = "Video and Audio Deepfake Detector"


def _load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Load audio without relying on a system FFmpeg installation."""
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    except (ImportError, OSError, RuntimeError, ValueError):
        try:
            import torchaudio

            audio_tensor, sample_rate = torchaudio.load(audio_path)
            audio = audio_tensor.mean(dim=0).numpy()
        except (ImportError, OSError, RuntimeError, ValueError) as torchaudio_error:
            raise ValueError(
                "The uploaded audio format could not be decoded."
            ) from torchaudio_error

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _predict_audio(audio_path: str | None) -> str:
    if not audio_path:
        gr.Warning("Please upload or record an audio file.")
        return "No audio file was provided."

    try:
        audio, sample_rate = _load_audio(audio_path)
        return inference.deepfakes_spec_predict((audio, sample_rate))
    except (ConnectionResetError, FileNotFoundError, OSError, ValueError, RuntimeError):
        gr.Warning("The audio file could not be processed. Please try another file.")
        return "Audio processing failed."


video_interface = gr.Interface(
    inference.deepfakes_video_predict,
    gr.Video(),
    gr.Textbox(),
    examples=["videos/REAL.mp4", "videos/FAKE.mp4"],
    cache_examples=False,
    title="Video Detection",
    description="Analyze sampled video frames with the EfficientNet visual encoder.",
)

audio_interface = gr.Interface(
    _predict_audio,
    gr.Audio(
        type="filepath",
        format=None,
        sources=["upload", "microphone"],
    ),
    gr.Textbox(),
    examples=[
        "audios/REAL.mp3",
        "audios/ElevenLabs_2026-11-05T03_44_44_Roger - Laid-Back, Casual, Resonant_pre_s35_sim65_v4.mp3",
    ],
    cache_examples=False,
    title="Audio Detection",
    description="Analyze uploaded or recorded audio with RawNet.",
)


app = gr.TabbedInterface(
    interface_list=[video_interface, audio_interface],
    tab_names=["Video Detection", "Audio Detection"],
    title=title,
)

if __name__ == '__main__':
    app.launch(share = False)
