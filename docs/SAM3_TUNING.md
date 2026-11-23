# SAM3 per-project tuning

**Status:** settings/validation module and unit tests committed. This repository does not yet contain the entire ClothSwapStudio application; do not treat this as a runnable release.

The settings source is `cloth_swap/studio/sam3_controls.py`.

### Groups

- `concept`: SAM3 text-conditioned video model thresholds, identity association, reconditioning and track longevity.
- `tracker`: click/box tracker mask post-processing.
- `processing`: studio-specific downscaling, mask resampling, morphology, protection edges and actor overlap.
- `preview`: text-prompted still-image mask threshold.

### Suggested starting points (not guaranteed improvements)

| Profile | part_tracking_max_side | part_mask_resample |
|---|---:|---|
| Fast | 960 | nearest |
| Balanced | 1440 | linear |
| Fine garment edges | 2048 (or native dimensions, if feasible) | linear |

Leave `image_size` at **1008** unless the installed Transformers model and processor support a different size. This is a model preprocessing size, not the final resolution of the video.

### Important warnings

- Increasing tracking resolution can sharply increase VRAM use and runtime.
- Enlarging a low-resolution **binary mask** is not the same as recovering image detail.
- SAM3 detection and tracking masks are not guaranteed to align accurately with collars, wrists or hair when occluded.
- Modifying inputs that affect tracking must invalidate the relevant cached masks and require a new track.
- `VIDEO_CONFIG_MAP` must be validated against the **installed** `transformers` version; mismatching model config keys must fail visibly, never silently drop a setting.

### Integration into existing studio (not yet committed as runnable integration)

Backend must call `effective(base_cfg["sam3"], project.state.get("sam3_settings"))` for each job; use `validate(...)` before storing project overrides. The worker must receive a per-job copy. Model-dependent changes should invalidate masks and rendered references.

The source workspace includes a React panel, FastAPI routes and processing integration. They still require a complete application-source import and runtime verification before this repo can be used as a runnable application.
