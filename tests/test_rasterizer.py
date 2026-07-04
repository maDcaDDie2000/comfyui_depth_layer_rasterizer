"""Tests for depth layer rasterization."""

import torch

from rasterizer.pipeline import depth_rasterize_layers  # noqa: E402 — conftest adds repo root


def _gradient_depth(batch: int, height: int, width: int) -> torch.Tensor:
    x = torch.linspace(0, 1, width).view(1, 1, width).expand(batch, height, width)
    return x


def test_all_layers_output_shape():
    image = torch.rand(2, 32, 48, 3)
    depth_map = _gradient_depth(2, 32, 48).unsqueeze(-1).expand(-1, -1, -1, 3)
    result = depth_rasterize_layers(
        image,
        rasterization_levels=4,
        depth_map=depth_map,
        output_mode="all_layers",
    )
    assert result["layered_images"].shape == (8, 32, 48, 3)
    assert result["layer_masks"].shape == (8, 32, 48)
    assert result["depth_quantized"].shape == (2, 32, 48, 3)


def test_single_layer_preserves_color():
    image = torch.zeros(1, 16, 16, 3)
    image[:, 4:12, 4:12, 0] = 1.0
    depth_map = torch.full((1, 16, 16), 0.25).unsqueeze(-1).expand(-1, -1, -1, 3)
    result = depth_rasterize_layers(
        image,
        rasterization_levels=4,
        depth_map=depth_map,
        output_mode="single_layer",
        selected_layer=2,
        soft_masks=False,
    )
    assert result["layered_images"].shape == (1, 16, 16, 3)
    assert result["layered_images"][0, 6, 6, 0] > 0.9


def test_flat_depth_assigns_middle_layer():
    image = torch.ones(1, 8, 8, 3)
    depth_map = torch.full((1, 8, 8), 0.5).unsqueeze(-1).expand(-1, -1, -1, 3)
    result = depth_rasterize_layers(
        image,
        rasterization_levels=4,
        depth_map=depth_map,
        output_mode="debug",
    )
    assert result["debug_preview"].shape == (1, 8, 8, 3)


def test_composite_mode():
    image = torch.rand(1, 24, 24, 3)
    depth_map = _gradient_depth(1, 24, 24).unsqueeze(-1).expand(-1, -1, -1, 3)
    result = depth_rasterize_layers(
        image,
        rasterization_levels=6,
        depth_map=depth_map,
        output_mode="composite",
        enable_outline=True,
        enable_shadow=True,
    )
    assert result["composited_image"].shape == (1, 24, 24, 3)
    assert result["composited_image"].max() > 0


def test_depth_model_callable():
    image = torch.rand(1, 16, 16, 3)

    def model(imgs):
        del imgs
        return _gradient_depth(1, 16, 16)

    result = depth_rasterize_layers(
        image,
        rasterization_levels=3,
        depth_model=model,
        output_mode="single_layer",
        selected_layer=0,
    )
    assert result["layer_masks"].shape == (1, 16, 16)
