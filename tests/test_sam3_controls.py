"""Fast validation checks for SAM3 studio configuration; no GPU required."""
import math
import pytest
from cloth_swap.studio.sam3_controls import (
    defaults, effective, schema, validate, VIDEO_CONFIG_MAP,
)

def test_schema_covers_controls():
    data = schema()
    assert {f["key"] for f in data["fields"]} == set(defaults())
    assert "score_threshold" in VIDEO_CONFIG_MAP
    assert "tracker_mask_threshold" not in VIDEO_CONFIG_MAP

def test_overlay_keeps_defaults():
    result = effective({"score_threshold": 0.6}, {"part_tracking_max_side": 1440})
    assert result["score_threshold"] == 0.6
    assert result["part_tracking_max_side"] == 1440
    assert defaults()["part_tracking_max_side"] == 960

@pytest.mark.parametrize("bad", [
    {"nonexistent": 1},
    {"part_tracking_max_side": 64},
    {"part_tracking_max_side": True},
    {"part_mask_resample": "cubic"},
    {"tracker_non_overlap": "false"},
    {"tracker_mask_threshold": float("nan")},
    {"init_trk_keep_alive": 200, "max_trk_keep_alive": 20},
])
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        validate(bad)
