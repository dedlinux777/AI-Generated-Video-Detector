# About

This is a powerful open-source tool designed to detect fake images, videos, and audios. Utilizing state-of-the-art deep learning techniques like EfficientNetV2 and MTCNN, It offers frame-by-frame video analysis, enabling high-accuracy deepfake detection. It's developed with a focus on ease of use, making it accessible for researchers, developers, and security analysts..

---

## Features

- Multimedia Detection: Detect deepfakes in videos and audio files using a unified platform.
- High Accuracy: Leverages EfficientNetV2 for enhanced prediction performance and accurate results. 
- Real-Time Video Analysis: Frame-by-frame analysis of videos with automatic face detection.
- User-Friendly Interface: Easy-to-use interface built with Gradio for uploading and processing media files.
- Open Source: Completely open source under the MIT license, making it available for developers to extend and improve.

---

## Demo-Data

You can test the deepfake detection capabilities of this by uploading your video files. The tool will analyze each frame of the video, detect faces, and determine the likelihood of the video being real or fake.

Examples:  
1. [Video1-fake-1-ff.mp4](#)
2. [Video6-real-1-ff.mp4](#)

---

## How It Works

DeepSecure-AI uses the following architecture:

1. Face Detection:  
   The [MTCNN](https://mtcnn.readthedocs.io/en/latest/) model detects faces in each frame of the video. If no face is detected, it will use the previous frame's face to ensure accuracy.

2. Fake vs. Real Classification:  
   Once the face is detected, it's resized and fed into the [EfficientNetV2](https://medium.com/aiguys/review-efficientnetv2-smaller-models-and-faster-training-47d4215dcdfb) deep learning model, which determines the likelihood of the frame being real or fake.

3. Fake Confidence:  
   A final prediction is generated as a percentage score, indicating the confidence that the media is fake.

4. Results:  
   This provides an output video, highlighting the detected faces and a summary of whether the input is classified as real or fake.

---

## Project Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.10
- Gradio (pip install gradio)
- TensorFlow (pip install tensorflow)
- OpenCV (pip install opencv-python)
- PyTorch (pip install torch torchvision torchaudio)
- facenet-pytorch (pip install facenet-pytorch)
- MoviePy (pip install moviepy)

### Installation

1. Clone the repository:
        git clone https://github.com/Pranesh-2005/AI-Generated-Video-Detector.git
      
2. Navigate to the cloned repo:
        cd AI-Generated-Video-Detector or Go To the repo manually

3. Install required dependencies:
        pip install -r requirements.txt
    

3. Download the pre-trained model weights for EfficientNetV2 and place them in the project folder.

### Running the Application

1. Launch the Gradio interface:
        python app.py
    

2. The web interface will be available locally. You can upload a video, and model will analyze and display results.

---

## Example Usage

Upload a video or audio file to detect fake media. Here are some sample predictions:

- Video Analysis: The tool will analyze sampled frames and classify whether the video is fake or real.
- Audio Analysis: The tool will classify a standalone audio waveform.

### Reproducible evaluation

`evaluate.py` evaluates one prediction per labelled video or audio file and writes
metrics, prediction CSVs, confusion-matrix plots, ROC plots when both classes are
available, a text summary, and `evaluation_report.md` to `evaluation_results/`.
Video manifests must
contain `video_path` and `label` columns, with labels `real` or `fake`.

```text
python evaluate.py --modality video --data_dir <dataset-root> --csv <labelled-video.csv>
python evaluate.py --modality audio --data_dir <audio-root> --audio_csv <labelled-audio.csv> --evaluation-set-type independent_test
python evaluate.py --modality both --video_dir <dataset-root> --video_csv <labelled-video.csv> --audio_dir <audio-root> --audio_csv <labelled-audio.csv> --evaluation-set-type independent_test
```

For audio, the manifest must contain an audio/file path column and a `label`
column. A directory-based audio dataset is supported when it contains `real/`
and `fake/` subdirectories. The positive class is `fake` (1); `real` is the
negative class (0). Missing or corrupt samples are recorded as skipped.
The six-stage implementation audit is documented in
`evaluation_results/system_architecture_report.md`.

---

## Technologies Used

- TensorFlow: For building and training deep learning models.
- EfficientNetV2: The core model for image and video classification.
- MTCNN: For face detection in images and videos.
- OpenCV: For video processing and frame manipulation.
- MoviePy: For video editing and result generation.
- Gradio: To create a user-friendly interface for interacting with the deepfake detector.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Disclaimer

This is a research project and is designed for educational purposes.Please use responsibly and always give proper credit when utilizing the model in your work.
