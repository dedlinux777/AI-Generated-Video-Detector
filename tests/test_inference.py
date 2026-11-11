import unittest
from unittest.mock import Mock, patch

import torch

import inference_2
from models.evidential import opinion_from_probabilities


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.frames = [torch.zeros(224, 224, 3)]
        self.visual_model = Mock(return_value=torch.tensor([[0.8, 0.2]]))
        self.visual_opinion = opinion_from_probabilities(torch.tensor([[0.8, 0.2]]))
        self.audio_opinion = opinion_from_probabilities(torch.tensor([[0.2, 0.8]]))

    def test_video_with_audio_runs_both_modalities_and_fusion(self):
        with patch.object(inference_2, "preprocess_video", return_value=self.frames), \
             patch.object(inference_2, "_load_visual_model", return_value=self.visual_model), \
             patch.object(inference_2, "extract_video_audio", return_value="audio.wav"), \
             patch.object(inference_2, "load_audio", return_value=torch.zeros(64600).numpy()), \
             patch.object(inference_2, "_predict_audio_opinion",
                          return_value=self.audio_opinion) as predict_audio, \
             patch.object(inference_2, "fuse_opinions",
                          return_value=self.visual_opinion) as fuse:
            result = inference_2.deepfakes_video_predict("video.mp4")

        self.assertIn("Fused result:", result)
        self.assertIn("Audio opinion:", result)
        predict_audio.assert_called_once()
        fuse.assert_called_once()

    def test_video_without_audio_returns_visual_only_result(self):
        with patch.object(inference_2, "preprocess_video", return_value=self.frames), \
             patch.object(inference_2, "_load_visual_model", return_value=self.visual_model), \
             patch.object(
                 inference_2,
                 "extract_video_audio",
                 side_effect=inference_2.AudioStreamUnavailable,
             ), \
             patch.object(inference_2, "_predict_audio_opinion") as predict_audio, \
             patch.object(inference_2, "fuse_opinions") as fuse:
            result = inference_2.deepfakes_video_predict("silent.mp4")

        self.assertIn("Visual result:", result)
        self.assertIn(
            "Audio stream unavailable. Prediction is based on visual analysis only.",
            result,
        )
        self.assertNotIn("Audio opinion:", result)
        predict_audio.assert_not_called()
        fuse.assert_not_called()

    def test_invalid_video_reports_missing_frames(self):
        with patch.object(inference_2, "preprocess_video", return_value=[]):
            with self.assertRaisesRegex(ValueError, "No readable frames"):
                inference_2.deepfakes_video_predict("invalid.mp4")

    def test_audio_extraction_failure_is_not_treated_as_missing_audio(self):
        extraction_error = inference_2.AudioExtractionError("codec failure")
        with patch.object(inference_2, "preprocess_video", return_value=self.frames), \
             patch.object(inference_2, "_load_visual_model", return_value=self.visual_model), \
             patch.object(inference_2, "extract_video_audio",
                          side_effect=extraction_error):
            with self.assertRaisesRegex(
                inference_2.AudioExtractionError, "codec failure"
            ):
                inference_2.deepfakes_video_predict("broken-audio.mp4")

    def test_standalone_audio_inference_is_preserved(self):
        with patch.object(inference_2, "load_audio", return_value=torch.zeros(64600).numpy()), \
             patch.object(inference_2, "_predict_audio_opinion",
                          return_value=self.audio_opinion):
            result = inference_2.deepfakes_spec_predict("audio.wav")

        self.assertIn("Audio opinion:", result)


if __name__ == "__main__":
    unittest.main()
