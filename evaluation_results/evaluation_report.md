# Performance Evaluation Report

**Status:** not completed: No labelled audio samples found. Provide --audio_csv with an explicit path/label manifest or use real/ and fake/ subdirectories.
**Evaluation-set designation:** unknown
**Positive class:** fake (1); real is negative (0).

## Methodology

The existing application checkpoint is evaluated without retraining or architectural changes. Video inference samples frames with the existing `inference_2.preprocess_video` function, obtains one RGB prediction per frame, averages frame scores, and records one final prediction per video. Audio inference loads one waveform per file, applies the existing `preprocess_audio` batch-dimension conversion, and records one RawNet prediction per audio file.

Metrics are calculated at the complete-file level using scikit-learn. Frame rows are never treated as independent video samples.

## Dataset and model

The repository contains FakeAVCeleb CSV manifests with `video_path` and `label` columns, but the referenced media is not present in this checkout. No labelled standalone-audio manifest is included. Therefore, an independent test-set result cannot be claimed from this checkout.

Checkpoint: `D:\dedlinux UVCE\7th Sem\Major Project\AI-Generated-Video-Detector\checkpoints\model.pth`

## Results

No quantitative results were generated because no valid labelled evaluation samples were available. This is an evaluation limitation, not a zero-valued experiment.

## Limitations

- The bundled manifests reference external FakeAVCeleb media that is not present locally.
- The repository does not provide a labelled standalone-audio test manifest.
- No result should be described as independent-test, validation, or demo performance until the corresponding labelled split is supplied and the command is rerun with `--evaluation-set-type`.
