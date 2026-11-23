"""Regression test for precomputed CLIP embeddings in Wan Animate.

Run in project root: python -m pytest -q tests/test_wan_reference_inputs.py
These tests mock the Diffusers pipeline and do not require model weights/GPU.
"""

import torch
import pytest

from cloth_swap.animate_runner import wan_animate_with_precomputed_embeddings


class MockWanAnimatePipeline:
    """Models the official validator's image/image_embeds exclusion check."""

    def __init__(self):
        self.actual_inputs = None
        self.checked_inputs = None

    def check_inputs(self, prompt, negative_prompt, image, pose_video, face_video,
                     background_video, mask_video, height, width,
                     prompt_embeds=None, negative_prompt_embeds=None,
                     image_embeds=None, callback_on_step_end_tensor_inputs=None,
                     mode=None, prev_segment_conditioning_frames=None):
        self.checked_inputs = (image, image_embeds)
        if image is not None and image_embeds is not None:
            raise ValueError("Cannot forward both image and image_embeds")
        if image is None and image_embeds is None:
            raise ValueError("Missing conditioning image")
        if height % 16 or width % 16:
            raise ValueError("Invalid dimensions")

    def __call__(self, image, image_embeds, **kwargs):
        self.check_inputs(None, None, image, [], [], [], [], 512, 896,
                          None, None, image_embeds, None, "replace", 1)
        # The real forward pass needs both the PIL image (VAE) and CLIP embeddings.
        self.actual_inputs = (image, image_embeds)
        return True


def test_reference_image_and_embeddings_reach_forward_together():
    class_ = wan_animate_with_precomputed_embeddings(MockWanAnimatePipeline)
    p = class_()
    image = object()
    embeds = torch.zeros((1, 257, 1280))
    assert p(image=image, image_embeds=embeds)
    assert p.checked_inputs == (image, None)  # validator sees only one
    assert p.actual_inputs[0] is image
    assert p.actual_inputs[1] is embeds   # forward keeps both


def test_validator_does_not_ignore_other_validation_errors():
    class_ = wan_animate_with_precomputed_embeddings(MockWanAnimatePipeline)
    p = class_()
    with pytest.raises(ValueError, match="Invalid dimensions"):
        p.check_inputs(None, None, object(), [], [], [], [], 513, 896,
                       image_embeds=torch.zeros((1, 10, 4)))
    with pytest.raises(ValueError, match="Precomputed CLIP"):
        p.check_inputs(None, None, object(), [], [], [], [], 512, 896,
                       image_embeds=torch.zeros((10, 4)))


def test_regular_pipeline_inputs_unchanged():
    class_ = wan_animate_with_precomputed_embeddings(MockWanAnimatePipeline)
    p = class_()
    p.check_inputs(None, None, object(), [], [], [], [], 512, 896)
    assert p.checked_inputs[1] is None
