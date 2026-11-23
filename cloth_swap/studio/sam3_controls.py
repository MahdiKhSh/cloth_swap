"""Safe, versioned, per-project SAM3 controls for the installed Transformers implementation.

Only settings genuinely wired to this studio's inference path are exposed.
Other SAM3 checkpoint internals are deliberately not relabelled as user-tunable knobs.
"""
from __future__ import annotations

import math

# name, default, lower, upper, category, requires, description (Persian)
FIELDS = {
    # Detection/association parameters of transformers.Sam3VideoConfig
    "score_threshold": (0.5, 0.0, 1.0, "concept", "layers", "حداقل اطمینان تشخیص متن؛ کمتر = اشیای بیشتر و خطای مثبت بیشتر"),
    "new_det_threshold": (0.7, 0.0, 1.0, "concept", "layers", "حداقل امتیاز برای شروع ردیابی یک آبجکت تازه"),
    "det_nms_thresh": (0.1, 0.0, 1.0, "concept", "layers", "حذف تشخیص‌های تکراری (NMS IoU)"),
    "assoc_iou_thresh": (0.1, 0.0, 1.0, "concept", "layers", "حداقل هم‌پوشانی اتصال تشخیص جدید به ردیابی"),
    "trk_assoc_iou_thresh": (0.5, 0.0, 1.0, "concept", "layers", "آستانه سخت‌گیرانه تطبیق ردیابی"),
    "high_conf_thresh": (0.8, 0.0, 1.0, "concept", "layers", "اطمینان بالا برای تصحیح مسیر ردیابی"),
    "high_iou_thresh": (0.8, 0.0, 1.0, "concept", "layers", "آستانه هم‌پوشانی بالا برای بازیابی ردیابی"),
    "recondition_on_trk_masks": (True, None, None, "concept", "layers", "حافظه ردیاب از ماسک ترک استفاده کند، نه تشخیص اولیه"),
    "recondition_every_nth_frame": (16, 1, 240, "concept", "layers", "تجدید شرط ردیاب هر چند فریم انجام شود"),
    "hotstart_delay": (15, 0, 120, "concept", "layers", "تعداد فریم برای تأیید اولیه ترک"),
    "hotstart_unmatch_thresh": (8, 0, 120, "concept", "layers", "تعداد فریم بی‌تطبیق پیش از حذف در شروع"),
    "hotstart_dup_thresh": (8, 0, 120, "concept", "layers", "تعداد فریم تشخیص تکراری برای حذف"),
    "suppress_unmatched_only_within_hotstart": (True, None, None, "concept", "layers", "حذف ترک‌های ناهماهنگ فقط در شروع"),
    "init_trk_keep_alive": (30, 0, 300, "concept", "layers", "عمر اولیه ترک هنگام پوشیدگی"),
    "max_trk_keep_alive": (30, 0, 300, "concept", "layers", "بیشترین عمر ترک هنگام پوشیدگی"),
    "min_trk_keep_alive": (-1, -1, 300, "concept", "layers", "کمترین عمر ترک (-1 = رفتار پیش‌فرض)"),
    "suppress_overlapping_based_on_recent_occlusion_threshold": (0.7, 0, 1, "concept", "layers", "حداکثر هم‌پوشانی برای جلوگیری از ترک تکراری"),
    "decrease_trk_keep_alive_for_empty_masklets": (False, None, None, "concept", "layers", "کاهش عمر ترک در صورت ماسک خالی"),
    "fill_hole_area": (16, 0, 10000, "concept", "layers", "پر کردن حفره‌های کوچک ماسک تشخیص"),
    "max_num_objects": (10000, 1, 10000, "concept", "layers", "سقف تعداد آبجکت‌های قابل‌ردیابی"),
    "image_size": (1008, 560, 1400, "concept", "layers", "ابعاد داخلی ورودی مدل متن‌محور (1008 مقدار آموزش؛ تغییر ممکن است دقت را کم کند)"),
    # Point/box tracker postprocess: HF Sam3TrackerVideoProcessor.post_process_masks
    "tracker_mask_threshold": (0.0, -2.0, 2.0, "tracker", "actor", "آستانه لاجیت ماسک ردیابی نقطه‌ای؛ مقدار پیش‌فرض صفر"),
    "tracker_max_hole_area": (0.0, 0.0, 20000.0, "tracker", "actor", "اندازه بیشینه حفره برای ترمیم ماسک ردیاب"),
    "tracker_max_sprinkle_area": (0.0, 0.0, 20000.0, "tracker", "actor", "اندازه بیشینه لکه‌های کوچک برای پاکسازی"),
    "tracker_non_overlap": (False, None, None, "tracker", "actor", "عدم هم‌پوشانی ماسک چند آبجکت در یک فریم"),
    "occlusion_min_iou": (0.05, 0.0, 1.0, "tracker", "actor", "حداقل IoU تشخیص پرش ماسک در برابر فریم معتبر قبل"),
    "occlusion_min_area": (0.4, 0.0, 1.0, "tracker", "actor", "حداقل نسبت مساحت ماسک به میانه برای تشخیص پوشیدگی"),
    # Image preview only (SAM3 image detection/point)
    "image_mask_threshold": (0.5, 0.0, 1.0, "preview", "preview", "آستانه باینری ماسک تشخیص متن روی تصویر"),
    # Studio-specific scaling, compositing, and validation settings
    "part_tracking_max_side": (960, 256, 4096, "processing", "layers", "بزرگ‌ترین ضلع کلیپ ماسک‌گذاری لباس؛ 4096 یعنی بدون کوچک‌سازی و حافظه بیشتر"),
    "part_mask_resample": ("nearest", None, None, "processing", "layers", "روش بازگرداندن ماسک به ابعاد ویدئو: nearest یا linear"),
    "part_mask_dilate_px": (0, -30, 30, "processing", "layers", "انبساط لبه ماسک لباس برحسب پیکسل در رزولوشن نهایی (منفی = فرسایش)"),
    "part_mask_blur_px": (0, 0, 10, "processing", "layers", "هموارسازی ماسک لباس با فیلتر Gaussian (صفر = خاموش)"),
    "part_fill_holes": (False, None, None, "processing", "layers", "پر کردن حفره‌های داخل ماسک لباس (ممکن است فاصله بین بازوها را هم پر کند)"),
    "part_min_component_px": (0, 0, 100000, "processing", "layers", "حذف لکه‌های متصل کوچک‌تر از مساحت تعیین‌شده؛ صفر = خاموش"),
    "part_protection_padding_frac": (0.006, 0, 0.05, "processing", "compose", "میزان گسترش ماسک محافظتی نسبت به ضلع کوچک‌تر فریم"),
    "concept_min_overlap": (0.3, 0, 1, "processing", "layers", "حداقل سهم قرارگیری آبجکت متن‌محور داخل ماسک شخص"),
}

ENUMS = {"part_mask_resample": ("nearest", "linear")}
CATEGORIES = {
    "tracker": "ردیابی با نقطه و کادر (بازیگر)",
    "concept": "شناسایی و ردیابی متنی SAM3 (لباس/اجزا)",
    "processing": "کیفیت، وضوح و پالایش ماسک",
    "preview": "پیش‌نمایش تک‌فریم",
}
# These are NOT real public knobs in the HF checkpoint and must not be passed to model config.
VIDEO_CONFIG_MAP = {k: k for k in FIELDS if FIELDS[k][3] == "concept" and k not in ("image_size", "score_threshold", "new_det_threshold")}
VIDEO_CONFIG_MAP.update({"score_threshold": "score_threshold_detection", "new_det_threshold": "new_det_thresh"})


def defaults() -> dict:
    return {key: spec[0] for key, spec in FIELDS.items()}


def effective(base: dict, overrides: dict | None = None) -> dict:
    values = defaults()
    # Include existing YAML config fields (backwards compatibility).
    for key in FIELDS:
        if key in base:
            values[key] = base[key]
    if overrides:
        values.update(overrides)
    validate(values)
    return values


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


def schema() -> dict:
    return {
        "groups": CATEGORIES,
        "fields": [dict(key=k, default=v[0], minimum=v[1], maximum=v[2], group=v[3], effect=v[4], help=v[5],
                        type="boolean" if isinstance(v[0], bool) else "select" if isinstance(v[0], str) else
                        "integer" if isinstance(v[0], int) else "number", options=list(ENUMS[k]) if k in ENUMS else None)
                   for k, v in FIELDS.items()],
    }
