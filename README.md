# Cloth Swap Studio — repository bootstrap

This repository is being initialized for the Cloth Swap Studio project.

## Currently committed

- `cloth_swap/studio/sam3_controls.py`: schema, type/range validation, defaults and per-project configuration overlays for SAM3 tracking and garment-mask processing.
- `tests/test_sam3_controls.py`: GPU-free validation tests for those controls.
- `docs/SAM3_TUNING.md`: tuning guidance and integration notes.

**IMPORTANT:** This is **not yet the complete, runnable application source**. In particular, the existing FastAPI service, SAM3 segmentation integration, React interface, garment bible and video pipeline files have **not** been uploaded to this repository. Do not deploy it as a working replacement for your studio.

The complete locally audited source bundle exists separately in the ChatGPT conversation as a downloadable ZIP. The new SAM3 settings implementation was validated locally against that source with 108 Python tests passing; this is not an end-to-end GPU certification.

No models, generated videos, or private project data have been uploaded.
