import os
import cv2
import onnx
import torch
import argparse
import numpy as np
from models.TMC import ETMC
from models import image

from onnx2pytorch import ConvertModel

onnx_model = onnx.load('checkpoints/efficientnet.onnx')
pytorch_model = ConvertModel(onnx_model)

#Set random seed for reproducibility.
torch.manual_seed(42)


# Define the audio_args dictionary
audio_args = {
    'nb_samp': 64600,
    'first_conv': 1024,
    'in_channels': 1,
    'filts': [20, [20, 20], [20, 128], [128, 128]],
    'blocks': [2, 4],
    'nb_fc_node': 1024,
    'gru_node': 1024,
    'nb_gru_layer': 3,
    'nb_classes': 2
}


def get_args(parser):
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="datasets/train/fakeavceleb*")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=1024)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="MMDF")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--pretrained_image_encoder", type=bool, default = False)
    parser.add_argument("--freeze_image_encoder", type=bool, default = False)
    parser.add_argument("--pretrained_audio_encoder", type = bool, default=False)
    parser.add_argument("--freeze_audio_encoder", type = bool, default = False)
    parser.add_argument("--augment_dataset", type = bool, default = True)

    for key, value in audio_args.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

def model_summary(args):
    '''Prints the model summary.'''
    model = ETMC(args)

    for name, layer in model.named_modules():
        print(name, layer)

def load_multimodal_model(args):
    '''Load multimodal model'''
    model = ETMC(args)
    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'))
    model.load_state_dict(ckpt, strict = True)
    model.eval()
    return model

def load_img_modality_model(args):
    '''Loads image modality model.'''
    rgb_encoder = pytorch_model

    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'))
    rgb_encoder.load_state_dict(ckpt['rgb_encoder'], strict = True)
    rgb_encoder.eval()
    return rgb_encoder

def load_spec_modality_model(args):
    spec_encoder = image.RawNet(args)
    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'))
    spec_encoder.load_state_dict(ckpt['spec_encoder'], strict = True)
    spec_encoder.eval()
    return spec_encoder


#Load models.
parser = argparse.ArgumentParser(description="Inference models")
get_args(parser)
args, remaining_args = parser.parse_known_args()
assert remaining_args == [], remaining_args

spec_model = load_spec_modality_model(args)

img_model = load_img_modality_model(args)


def preprocess_frame(face):
    face = face.astype(np.float32) / 255
    face = cv2.resize(face, (256, 256))
    return torch.from_numpy(face).unsqueeze(0)

def preprocess_audio(audio_file, sample_rate=16000):
    audio = torch.as_tensor(audio_file, dtype=torch.float32)
    if sample_rate and sample_rate != 16000:
        import torchaudio.functional as audio_functional

        audio = audio_functional.resample(audio, sample_rate, 16000)
    return audio.unsqueeze(0)

def deepfakes_spec_predict(input_audio):
    x, sample_rate = input_audio
    audio = preprocess_audio(x, sample_rate)
    with torch.inference_mode():
        log_probabilities = spec_model(audio)
        probabilities = torch.exp(log_probabilities).squeeze(0)

    class_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[class_index].item()) * 100
    label = "REAL" if class_index == 1 else "FAKE"
    return f"The audio is {label}. Confidence score is: {confidence:.3f}%"


def preprocess_video(input_video, n_frames = 3):
    v_cap = cv2.VideoCapture(input_video)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pick 'n_frames' evenly spaced frames to sample
    if n_frames is None:
        sample = np.arange(0, v_len)
    else:
        sample = set(np.linspace(0, v_len - 1, n_frames).astype(int))

    #Loop through frames.
    frames = []
    for j in range(v_len):
        success = v_cap.grab()
        if j in sample:
            # Load frame
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess_frame(frame)
            frames.append(frame)
    v_cap.release()
    return frames


def deepfakes_video_predict(input_video):
    '''Perform inference on a video.'''
    video_frames = preprocess_video(input_video)
    if not video_frames:
        raise ValueError("No readable frames were found in the video.")

    frames = torch.cat(video_frames, dim=0)
    with torch.inference_mode():
        predictions = img_model(frames).cpu().numpy()

    real_faces_mean = np.mean(predictions[:, 0])
    fake_faces_mean = np.mean(predictions[:, 1])

    if real_faces_mean > 0.5:
        preds = round(real_faces_mean * 100, 3)
        text2 = f"The video is REAL. \nConfidence score is: {preds}%"

    else:
        preds = round(fake_faces_mean * 100, 3)
        text2 = f"The video is FAKE. \nConfidence score is: {preds}%"

    return text2

