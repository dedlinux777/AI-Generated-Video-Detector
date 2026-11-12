# System Architecture Verification Report

## System overview

The repository contains an EfficientNet-based RGB encoder, a RawNet
raw-waveform encoder, and an ETMC/TMC evidential fusion model. The deployed
Gradio application currently exposes two user-facing modes: video detection and
standalone audio detection. The application loads the modality encoders from
`checkpoints/model.pth` through `inference_2.py`; it does not retrain or replace
the checkpoint.

The six-stage design below preserves the intended organization while
distinguishing code that is demonstrably implemented from intended features
that are absent or used only by the training architecture.

## Verification table

| Stage | Intended functionality | Actual implementation | Relevant file/function | Status |
|---|---|---|---|---|
| 1. Dual-Input Media Layer | Video containing visual frames and audio; standalone audio | Gradio accepts video and audio inputs. The application video path analyzes visual frames; it does not extract the video's audio track. | [app.py](../app.py), `inference_2.deepfakes_video_predict`, `inference_2.deepfakes_spec_predict` | Partially implemented |
| 2. Preprocessing and stream decoupling | Frame/audio extraction, MTCNN face crops, 256x256 resize, 16 kHz mono standardization | Video frames are sampled and resized to 256x256. No MTCNN face detection/cropping is present in the active inference path. Audio is converted to a tensor with an added batch dimension; no active 16 kHz resampling is performed. | [inference_2.py](../inference_2.py), `preprocess_video`, `preprocess_img`, `preprocess_audio` | Partially implemented |
| 3. Deep neural backbones | EfficientNet visual and RawNet acoustic feature extraction | The RGB checkpoint is loaded into the ONNX-converted EfficientNet classifier. RawNet is loaded from the `spec_encoder` checkpoint state. | [models/image.py](../models/image.py), `ImageEncoder`, `RawNet`; [inference_2.py](../inference_2.py) | Implemented |
| 4. Subjective Logic and evidential heads | Evidence, Dirichlet alpha, belief, uncertainty | ETMC/TMC applies Softplus evidence and `alpha = evidence + 1`; `DS_Combin_two` computes belief, uncertainty, conflict, and combined alpha. These heads are present in the model/training implementation. | [models/TMC.py](../models/TMC.py), `TMC.DS_Combin_two`, `ETMC.forward` | Implemented |
| 5. Trusted Multi-View Classification Engine | Acoustic/visual opinions, pseudo-view, Dempster-Shafer fusion; audio-only branch | ETMC creates acoustic, RGB, pseudo-view, and fused alpha outputs. Standalone application audio uses RawNet directly and does not require visual inference. The active application video/audio functions do not call the full ETMC forward pass. | [models/TMC.py](../models/TMC.py), [inference_2.py](../inference_2.py) | Partially implemented |
| 6. Decision and Interface Layer | Gradio predictions for video/audio and reported confidence | Exactly two Gradio tabs are exposed. Video averages RGB real/fake scores and declares real when mean real score is greater than 0.5. Audio exponentiates RawNet log-softmax output and selects the higher class; class 1 is treated as real by the existing function. | [app.py](../app.py), `inference_2.deepfakes_video_predict`, `inference_2.deepfakes_spec_predict` | Implemented |

## Technical description

### Input modalities and preprocessing

The active interface accepts video files and standalone audio files. In
`preprocess_video`, OpenCV samples three evenly spaced frames by default. Each
frame is converted from BGR to RGB and resized to 256x256 by
`preprocess_img`. The active `inference_2.py` preprocessing preserves the
existing tensor layout used by the application.

The active audio path receives a waveform, adds a batch dimension, and sends it
to RawNet. Although RawNet is parameterized with a 16,000 Hz filter-bank sample
rate, the active inference code does not resample input files. MTCNN face
detection, facial alignment, video-audio extraction, and predictive Shannon
entropy are not demonstrated by the active code and must not be reported as
implemented features.

### EfficientNet and RawNet

`ImageEncoder` wraps the EfficientNet classifier and applies a sigmoid to its
two output scores. `RawNet` processes the waveform through Sinc convolution,
residual blocks, attention layers, a GRU, and a final log-softmax classifier.
The application checkpoint contains `rgb_encoder` and `spec_encoder` state
dictionaries, which are loaded without changing the model definitions.

### Evidential learning and subjective logic

The ETMC/TMC training architecture applies:

\[
e = \operatorname{Softplus}(z), \qquad \alpha = e + 1
\]

For each view, the implementation computes:

\[
S = \sum_k \alpha_k,\qquad b_k = \frac{\alpha_k-1}{S},\qquad u = \frac{c}{S}
\]

where \(c\) is the number of classes. `DS_Combin_two` combines two opinions,
computes the conflict term \(K\), and returns combined Dirichlet parameters.
`ETMC.forward` combines the depth/acoustic view with the RGB view and then
combines that result with the pseudo-view generated from concatenated RGB and
acoustic features.

The code does not expose an ambiguity label or entropy-based decision in the
Gradio output. The report should therefore describe belief/uncertainty as
internal evidential quantities, not as user-facing ambiguous predictions.

### Final classification

The active video function computes the mean of the per-frame RGB scores and
returns `REAL` if the mean real score exceeds 0.5; otherwise it returns `FAKE`
and reports the mean fake score. The active audio function exponentiates the
RawNet log-softmax vector and uses the class index selected by `argmax`; its
existing text logic maps class 1 to real and class 0 to fake.

## Limitations for the project report

- The local checkout does not contain the external FakeAVCeleb media referenced
  by its CSV manifests.
- No labelled standalone-audio evaluation manifest is included.
- Independent-test performance cannot be claimed until a held-out labelled
  split is supplied and evaluated with `evaluate.py`.
- The deployed application uses modality-specific encoders rather than calling
  the complete ETMC multimodal forward pass at inference time.
- MTCNN face cropping, video-audio extraction, active 16 kHz resampling, and
  ambiguity/entropy outputs are intended design elements or future enhancements,
  not verified active behavior.
