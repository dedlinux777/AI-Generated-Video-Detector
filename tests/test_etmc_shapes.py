import unittest
from types import SimpleNamespace

import torch

from models.TMC import ETMC
from inference_2 import DEFAULT_ARCHITECTURE


class ETMCShapeTests(unittest.TestCase):
    def test_actual_etmc_modules_produce_two_class_opinions(self):
        model = ETMC(SimpleNamespace(**DEFAULT_ARCHITECTURE)).eval()
        with torch.inference_mode():
            visual_features = model.rgbenc(torch.zeros(1, 3, 256, 256)).flatten(1)
            audio_features = model.specenc(torch.zeros(1, 64600)).flatten(1)
            outputs = model(
                torch.zeros(1, 3, 256, 256),
                torch.zeros(1, 64600),
            )

        self.assertEqual(visual_features.shape, (1, 1))
        self.assertEqual(audio_features.shape, (1, 2))
        self.assertEqual([output.shape for output in outputs], [(1, 2)] * 4)
        for output in outputs:
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue((output > 0).all())


if __name__ == "__main__":
    unittest.main()
