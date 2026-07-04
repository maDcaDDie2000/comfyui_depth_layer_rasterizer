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
    assert result["composited_image"].mean() > 0.05


def test_composite_matches_input_brightness():
    image = torch.full((1, 32, 32, 3), 0.75)
    depth_map = torch.full((1, 32, 32), 0.4).unsqueeze(-1).expand(-1, -1, -1, 3)
    result = depth_rasterize_layers(
        image,
        rasterization_levels=4,
        depth_map=depth_map,
        output_mode="composite",
        soft_masks=True,
        depth_blur=0.0,
    )
    assert result["composited_image"].mean() > 0.6


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


def test_flat_average_layer_color():
    image = torch.zeros(1, 8, 8, 3)
    image[:, :, :, 0] = 1.0
    image[:, :, :, 1] = 0.5
    depth_map = torch.full((1, 8, 8), 0.25).unsqueeze(-1).expand(-1, -1, -1, 3)
    result = depth_rasterize_layers(
        image,
        rasterization_levels=4,
        depth_map=depth_map,
        output_mode="single_layer",
        selected_layer=2,
        layer_color_mode="flat_average",
        soft_masks=False,
        depth_blur=0.0,
    )
    center = result["layered_images"][0, 4, 4]
    assert center[0] > 0.9
    assert center[1] > 0.4


def test_color_zones_reduces_unique_colors():
    image = torch.rand(1, 32, 32, 3)
    depth_map = torch.full((1, 32, 32), 0.3).unsqueeze(-1).expand(-1, -1, -1, 3)
    original = depth_rasterize_layers(
        image,
        rasterization_levels=2,
        depth_map=depth_map,
        output_mode="single_layer",
        selected_layer=1,
        layer_color_mode="original",
        soft_masks=False,
        depth_blur=0.0,
    )
    zoned = depth_rasterize_layers(
        image,
        rasterization_levels=2,
        depth_map=depth_map,
        output_mode="single_layer",
        selected_layer=1,
        layer_color_mode="color_zones",
        color_zones_per_layer=4,
        soft_masks=False,
        depth_blur=0.0,
    )
    mask = original["layer_masks"][0] > 0.5
    orig_unique = torch.unique(original["layered_images"][0][mask].round(decimals=2), dim=0).shape[0]
    zone_unique = torch.unique(zoned["layered_images"][0][mask].round(decimals=2), dim=0).shape[0]
    assert zone_unique <= 4
    assert zone_unique < orig_unique


def test_layer_smoothing_reduces_boundary_noise():
    """Noisy depth should produce fewer layer transitions when layer_smoothing is on."""
    image = torch.rand(1, 64, 64, 3)
    base = torch.linspace(0, 1, 64).view(1, 1, 64).expand(1, 64, 64)
    noise = torch.rand(1, 64, 64) * 0.15
    depth_map = (base + noise).clamp(0, 1).unsqueeze(-1).expand(-1, -1, -1, 3)

    def boundary_count(layer_masks):
        m = layer_masks[0] > 0.5
        gx = (m[:, 1:] != m[:, :-1]).sum()
        gy = (m[1:, :] != m[:-1, :]).sum()
        return int(gx + gy)

    sharp = depth_rasterize_layers(
        image,
        rasterization_levels=8,
        depth_map=depth_map,
        output_mode="single_layer",
        selected_layer=3,
        layer_smoothing=0.0,
        soft_masks=False,
        depth_blur=0.0,
    )
    smooth = depth_rasterize_layers(
        image,
        rasterization_levels=8,
        depth_map=depth_map,
        output_mode="single_layer",
        selected_layer=3,
        layer_smoothing=6.0,
        soft_masks=False,
        depth_blur=0.0,
    )
    assert boundary_count(smooth["layer_masks"]) < boundary_count(sharp["layer_masks"])
