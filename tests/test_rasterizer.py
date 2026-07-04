"""Tests for depth layer rasterization."""

import importlib.util
from pathlib import Path

import torch

from rasterizer.pipeline import depth_rasterize_layers  # noqa: E402 - conftest adds repo root

_TOOLTIPS_SPEC = importlib.util.spec_from_file_location(
    "depth_layer_tooltips",
    Path(__file__).resolve().parents[1] / "nodes" / "tooltips.py",
)
_TOOLTIPS = importlib.util.module_from_spec(_TOOLTIPS_SPEC)
assert _TOOLTIPS_SPEC.loader is not None
_TOOLTIPS_SPEC.loader.exec_module(_TOOLTIPS)
build_depth_rasterize_input_types = _TOOLTIPS.build_depth_rasterize_input_types


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


def test_composite_shadow_visible():
    image = torch.rand(1, 64, 64, 3)
    depth_map = torch.linspace(0, 1, 64).view(1, 1, 64).expand(1, 64, 64).unsqueeze(-1).expand(-1, -1, -1, 3)
    without = depth_rasterize_layers(
        image,
        rasterization_levels=8,
        depth_map=depth_map,
        output_mode="composite",
        enable_shadow=False,
        soft_masks=False,
        depth_blur=0.0,
    )
    with_shadow = depth_rasterize_layers(
        image,
        rasterization_levels=8,
        depth_map=depth_map,
        output_mode="composite",
        enable_shadow=True,
        shadow_distance=10.0,
        shadow_opacity=0.6,
        depth_scaled_shadow=False,
        soft_masks=False,
        depth_blur=0.0,
    )
    assert with_shadow["composited_image"].mean() < without["composited_image"].mean()


def test_black_white_layer_color_mode():
    image = torch.zeros(1, 8, 8, 3)
    image[:, :, :4, :] = 0.25
    image[:, :, 4:, :] = 0.75
    depth_map = torch.full((1, 8, 8), 0.3).unsqueeze(-1).expand(-1, -1, -1, 3)

    result = depth_rasterize_layers(
        image,
        rasterization_levels=2,
        depth_map=depth_map,
        output_mode="single_layer",
        selected_layer=1,
        layer_color_mode="black_white",
        black_white_threshold=0.5,
        soft_masks=False,
        depth_blur=0.0,
    )

    layer = result["layered_images"][0]
    assert layer[:, :4].max() == 0
    assert layer[:, 4:].min() == 1


def test_color_adjustments_apply_inside_layer_mask():
    image = torch.full((1, 8, 8, 3), 0.4)
    depth_map = torch.full((1, 8, 8), 0.3).unsqueeze(-1).expand(-1, -1, -1, 3)

    result = depth_rasterize_layers(
        image,
        rasterization_levels=2,
        depth_map=depth_map,
        output_mode="single_layer",
        selected_layer=1,
        layer_color_mode="flat_average",
        color_brightness=0.2,
        soft_masks=False,
        depth_blur=0.0,
    )

    assert torch.allclose(result["layered_images"][0, 4, 4], torch.full((3,), 0.6), atol=1e-5)


def test_manual_depth_range_uses_normalization_mode_only():
    image = torch.ones(1, 8, 8, 3)
    depth_map = torch.full((1, 8, 8), 0.75).unsqueeze(-1).expand(-1, -1, -1, 3)

    result = depth_rasterize_layers(
        image,
        rasterization_levels=4,
        depth_map=depth_map,
        output_mode="single_layer",
        selected_layer=3,
        normalization_mode="manual",
        depth_min=0.0,
        depth_max=1.0,
        soft_masks=False,
        depth_blur=0.0,
    )

    assert result["layer_masks"].sum() == 64


def test_node_ui_hides_internal_cleanup_knobs():
    optional = build_depth_rasterize_input_types()["optional"]
    hidden = {
        "auto_normalize_depth",
        "soft_masks",
        "mask_expand",
        "mask_erode",
        "layer_order",
        "outline_mode",
        "offset_mode",
        "color_zone_space",
    }
    assert hidden.isdisjoint(optional)
    assert "black_white" in optional["layer_color_mode"][0]
    assert "color_brightness" in optional
