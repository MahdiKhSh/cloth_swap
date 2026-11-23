# Cloth Swap Studio

AI-assisted clothing replacement for video, using SAM3 segmentation, reference-garment tracking, and a React/FastAPI studio.

## Source status
Initial repository bootstrap. The full application and the SAM3 tuning interface are being imported from the audited local source.

### SAM3 tuning
The studio provides per-project controls for model thresholds, tracker configuration, inference resolution, mask processing, and preview. Settings that alter masks invalidate relevant cached outputs.

## Security
Model checkpoints, generated videos, private datasets and `studio_data/` are excluded from source control.
