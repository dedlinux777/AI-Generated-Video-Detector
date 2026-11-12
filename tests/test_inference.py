import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import inference_2
from models.evidential import opinion_from_alpha


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.opinions = tuple(
            opinion_from_alpha(torch.tensor([[2.0, 3.0]])) for _ in range(4)
        )

    def test_strict_loader_rejects_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"rgb_encoder": {}, "spec_encoder": {}}, path)
            with self.assertRaisesRegex(ValueError, "missing keys"):
                inference_2.load_complete_etmc(path)

    def test_strict_loader_passes_strict_flag(self):
        fake_model = Mock()
        fake_model.load_state_dict.return_value = None
        fake_model.eval.return_value = fake_model
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "etmc.pt"
            torch.save({
                "state_dict": {"weight": torch.ones(1)},
                "class_mapping": inference_2.EXPECTED_CLASS_MAPPING,
                "architecture": "ETMC",
                "architecture_config": inference_2.DEFAULT_ARCHITECTURE,
            }, path)
            with patch.object(inference_2, "ETMC", return_value=fake_model):
                loaded = inference_2.load_complete_etmc(path)
        self.assertIs(loaded, fake_model)
        fake_model.load_state_dict.assert_called_once_with({"weight": torch.ones(1)}, strict=True)

    def test_opinions_have_expected_shapes(self):
        for opinion in self.opinions:
            self.assertEqual(opinion.alpha.shape, (1, 2))
            self.assertEqual(opinion.expected.shape, (1, 2))

    def test_video_uses_trained_etmc_fusion_and_returns_all_opinions(self):
        with patch.object(inference_2, "preprocess_video", return_value=[torch.zeros(3, 256, 256)]), \
             patch.object(inference_2, "extract_video_audio", return_value="audio.wav"), \
             patch.object(inference_2, "load_audio", return_value=torch.zeros(64600).numpy()), \
             patch.object(inference_2, "_predict_etmc_opinions", return_value=self.opinions) as predict:
            result = inference_2.deepfakes_video_predict("video.mp4")
        predict.assert_called_once()
        self.assertIn("Visual opinion:", result)
        self.assertIn("Audio opinion:", result)
        self.assertIn("Pseudo-view opinion:", result)
        self.assertIn("Fused opinion:", result)
        self.assertIn("trained ETMC Dempster-Shafer fusion", result)

    def test_standalone_audio_does_not_load_or_execute_visual_branch(self):
        branch = Mock()
        branch.return_value = torch.tensor([[2.0, 3.0]])
        with patch.object(inference_2, "_load_audio_branch", return_value=branch), \
             patch.object(inference_2, "_load_etmc_model") as full_model, \
             patch.object(inference_2, "load_audio", return_value=torch.zeros(64600).numpy()):
            result = inference_2.deepfakes_spec_predict("audio.wav")
        full_model.assert_not_called()
        branch.assert_called_once()
        self.assertIn("Audio opinion:", result)

    def test_class_mapping_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save({
                "state_dict": {},
                "class_mapping": {"0": "fake", "1": "real"},
                "architecture": "ETMC",
                "architecture_config": inference_2.DEFAULT_ARCHITECTURE,
            }, path)
            with self.assertRaisesRegex(ValueError, "class mapping"):
                inference_2.load_complete_etmc(path)


if __name__ == "__main__":
    unittest.main()
