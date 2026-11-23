"""Per-project SAM3 model and segmentation controls.

This module intentionally separates Hugging Face video model settings from
studio-specific mask resampling/cleanup settings. It has no GPU dependencies.
"""
from __future__ import annotations

import math

# key -> (default, min, max, group, invalidates, Persian description)
FIELDS = {
    "score_threshold": (0.5, 0.0, 1.0, "concept", "layers", "حداقل اطمینان تشخیص متن"),
    "new_det_threshold": (0.7, 0.0, 1.0, "concept", "layers", "حداقل امتیاز تشخیص جدید"),
    "det_nms_thresh": (0.1, 0.0, 1.0, "concept", "layers", "NMS IoU"),
    "assoc_iou_thresh": (0.1, 0.0, 1.0, "concept", "layers", "آستانه تطبیق تشخیص و ردیابی"),
    "trk_assoc_iou_thresh": (0.5, 0.0, 1.0, "concept", "layers", "آستانه تطبیق ردیابی"),
    "high_conf_thresh": (0.8, 0.0, 1.0, "concept", "layers", "اطمینان بالا"),
    "high_iou_thresh": (0.8, 0.0, 1.0, "concept", "layers", "هم‌پوشانی بالا"),
    "recondition_on_trk_masks": (True, None, None, "concept", "layers", "استفاده از ماسک ردیابی"),
    "recondition_every_nth_frame": (16, 1, 240, "concept", "layers", "فاصله تجدید شرط در فریم‌ها"),
    "hotstart_delay": (15, 0, 120, "concept", "layers", "تأخیر شروع"),
    "hotstart_unmatch_thresh": (8, 0, 120, "concept", "layers", "حد عدم تطبیق در شروع"),
    "hotstart_dup_thresh": (8, 0, 120, "concept", "layers", "حد تشخیص تکراری در شروع"),
    "suppress_unmatched_only_within_hotstart": (True, None, None, "concept", "layers", "حذف ردیابی ناهماهنگ فقط در شروع"),
    "init_trk_keep_alive": (30, 0, 300, "concept", "layers", "عمر اولیه ترک"),
    "max_trk_keep_alive": (30, 0, 300, "concept", "layers", "حداکثر عمر ترک"),
    "min_trk_keep_alive": (-1, -1, 300, "concept", "layers", "حداقل عمر ترک"),
    "suppress_overlapping_based_on_recent_occlusion_threshold": (0.7, 0, 1, "concept", "layers", "حذف ترک‌های هم‌پوشان"),
    "decrease_trk_keep_alive_for_empty_masklets": (False, None, None, "concept", "layers", "کاهش عمر ترک با ماسک خالی"),
    "fill_hole_area": (16, 0, 10000, "concept", "layers", "ترمیم حفره‌های کوچک"),
    "max_num_objects": (10000, 1, 10000, "concept", "layers", "حداکثر تعداد آبجکت‌ها"),
    "image_size": (1008, 560, 1400, "concept", "layers", "ابعاد پردازش داخلی مدل"),
    "tracker_mask_threshold": (0.0, -2.0, 2.0, "tracker", "actor", "آستانه لاجیت ماسک ردیاب"),
    "tracker_max_hole_area": (0.0, 0.0, 20000.0, "tracker", "actor", "مساحت ترمیم حفره"),
    "tracker_max_sprinkle_area": (0.0, 0.0, 20000.0, "tracker", "actor", "مساحت حذف لکه"),
    "tracker_non_overlap": (False, None, None, "tracker", "actor", "جلوگیری از هم‌پوشانی ماسک‌ها"),
    "occlusion_min_iou": (0.05, 0.0, 1.0, "tracker", "actor", "حداقل IoU جهت تشخیص پرش"),
    "occlusion_min_area": (0.4, 0.0, 1.0, "tracker", "actor", "حداقل نسبت مساحت ماسک"),
    "image_mask_threshold": (0.5, 0.0, 1.0, "preview", "preview", "آستانه ماسک تصویر"),
    "part_tracking_max_side": (960, 256, 4096, "processing", "layers", "حداکثر ضلع پردازش لباس"),
    "part_mask_resample": ("nearest", None, None, "processing", "layers", "روش بازگرداندن ماسک"),
    "part_mask_dilate_px": (0, -30, 30, "processing", "layers", "انبساط/فرسایش لبه"),
    "part_mask_blur_px": (0, 0, 10, "processing", "layers", "هموارسازی Gaussian"),
    "part_fill_holes": (False, None, None, "processing", "layers", "پرکردن حفره‌ها"),
    "part_min_component_px": (0, 0, 100000, "processing", "layers", "حذف اجزای کوچک"),
    "part_protection_padding_frac": (0.006, 0.0, 0.05, "processing", "compose", "حاشیه محافظت از ماسک"),
    "concept_min_overlap": (0.3, 0.0, 1.0, "processing", "layers", "حداقل هم‌پوشانی آبجکت و شخص"),
}

ENUMS = {"part_mask_resample": ("nearest", "linear")}
CATEGORIES = {
    "tracker": "ردیابی با نقطه و کادر",
    "concept": "ردیابی متن‌محور لباس",
    "processing": "کیفیت و پالایش ماسک",
    "preview": "پیش‌نمایش تک‌فریم",
}
VIDEO_CONFIG_MAP = {
    k: k for k, v in FIELDS.items() if v[3] == "concept"
    and k not in ("image_size", "score_threshold", "new_det_threshold")
}
VIDEO_CONFIG_MAP.update({
    "score_threshold": "score_threshold_detection",
    "new_det_threshold": "new_det_thresh",
})


def defaults() -> dict:
    return {key: spec[0] for key, spec in FIELDS.items()}


def validate(values: dict, *, partial: bool = False) -> dict:
    if not isinstance(values, dict):
        raise ValueError("SAM3 settings must be an object")
    for name, value in values.items():
        if name not in FIELDS:
            raise ValueError(f"Unknown SAM3 setting: {name}")
        default, low, high, *_ = FIELDS[name]
        if isinstance(default, bool):
            if type(value) is not bool:
                raise ValueError(f"sam3.{name} must be true or false")
        elif isinstance(default, str):
            if type(value) is not str or value not in ENUMS[name]:
                raise ValueError(f"sam3.{name} must be one of {ENUMS[name]}")
        elif isinstance(default, int):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"sam3.{name} must be an integer between {low} and {high}")
        else:
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"sam3.{name} must be a number between {low} and {high}")
    if not partial:
        merged = {**defaults(), **values}
        if merged["min_trk_keep_alive"] > merged["max_trk_keep_alive"]:
            raise ValueError("Minimum track lifespan cannot exceed maximum track lifespan")
        if merged["init_trk_keep_alive"] > merged["max_trk_keep_alive"]:
            raise ValueError("Initial track lifespan cannot exceed maximum track lifespan")
    return values


def effective(base: dict, overrides: dict | None = None) -> dict:
    values = defaults()
    for key in FIELDS:
        if key in base:
            values[key] = base[key]
    if overrides:
        values.update(overrides)
    validate(values)
    return values


def schema() -> dict:
    return {
        "groups": CATEGORIES,
        "fields": [
            dict(
                key=k, default=v[0], minimum=v[1], maximum=v[2],
                group=v[3], effect=v[4], help=v[5],
                type=("boolean" if isinstance(v[0], bool) else
                      "select" if isinstance(v[0], str) else
                      "integer" if isinstance(v[0], int) else "number"),
                options=list(ENUMS[k]) if k in ENUMS else None,
            )
            for k, v in FIELDS.items()
        ],
    }
