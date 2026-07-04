"""Outline, shadow, and layer offset effects."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from .masks import morph_dilate, morph_erode


def _color_tensor(color: Sequence[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rgb = torch.tensor(list(color[:3]), device=device, dtype=dtype)
    return rgb.view(1, 1, 1, 3)


def edge_detect_layer_boundaries(layer_id: torch.Tensor) -> torch.Tensor:
    """Detect pixels whose 4-neighbors belong to a different layer [B, H, W]."""
    left = layer_id != F.pad(layer_id, (1, 0, 0, 0))[:, :, 1:]
    right = layer_id != F.pad(layer_id, (0, 1, 0, 0))[:, :, :-1]
    up = layer_id != F.pad(layer_id, (0, 0, 1, 0))[:, 1:, :]
    down = layer_id != F.pad(layer_id, (0, 0, 0, 1))[:, :-1, :]

    edge = torch.zeros_like(layer_id, dtype=torch.float32)
    edge[:, :, 1:] = edge[:, :, 1:] + left.float()
    edge[:, :, :-1] = edge[:, :, :-1] + right.float()
    edge[:, 1:, :] = edge[:, 1:, :] + up.float()
    edge[:, :-1, :] = edge[:, :-1, :] + down.float()
    return (edge > 0).float()


def create_outline_mask(
    mask: torch.Tensor,
    layer_id: torch.Tensor | None,
    outline_width: int,
    outline_mode: str,
) -> torch.Tensor:
    """Generate outline alpha mask for one layer [B, H, W]."""
    width = max(outline_width, 1)

    if outline_mode == "depth_boundary" and layer_id is not None:
        return edge_detect_layer_boundaries(layer_id) * mask

    expanded = morph_dilate(mask, width)
    eroded = morph_erode(mask, width)

    if outline_mode == "outer":
        return (expanded - mask).clamp(0.0, 1.0)
    if outline_mode == "inner":
        return (mask - eroded).clamp(0.0, 1.0)
    # centered
    return (expanded - eroded).clamp(0.0, 1.0)


def apply_outline_to_layer(
    layer_rgb: torch.Tensor,
    outline: torch.Tensor,
    outline_color: Sequence[float],
    outline_opacity: float,
) -> torch.Tensor:
    """Composite outline color over a layer image [B, H, W, C]."""
    if outline_opacity <= 0:
        return layer_rgb

    color = _color_tensor(outline_color, layer_rgb.device, layer_rgb.dtype)
    alpha = outline.unsqueeze(-1) * outline_opacity
    return layer_rgb * (1.0 - alpha) + color * alpha


def translate_mask(mask: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Shift mask by fractional pixel offsets using grid_sample."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return mask

    b, h, w = mask.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=mask.device, dtype=mask.dtype),
        torch.linspace(-1, 1, w, device=mask.device, dtype=mask.dtype),
        indexing="ij",
    )

    shift_x = (2.0 * dx) / max(w - 1, 1)
    shift_y = (2.0 * dy) / max(h - 1, 1)
    grid = torch.stack((grid_x + shift_x, grid_y + shift_y), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

    sampled = F.grid_sample(mask.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(1).clamp(0.0, 1.0)


def gaussian_blur_mask(mask: torch.Tensor, radius: float) -> torch.Tensor:
    """Blur a mask [B, H, W]."""
    if radius <= 0:
        return mask

    kernel_size = max(3, int(math.ceil(radius * 2)) * 2 + 1)
    sigma = max(radius * 0.5, 1e-3)
    coords = torch.arange(kernel_size, device=mask.device, dtype=mask.dtype) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    x = mask.unsqueeze(1)
    pad = kernel_size // 2
    kh = kernel_1d.view(1, 1, 1, -1)
    kv = kernel_1d.view(1, 1, -1, 1)
    blurred = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kh)
    blurred = F.conv2d(F.pad(blurred, (0, 0, pad, pad), mode="reflect"), kv)
    return blurred.squeeze(1).clamp(0.0, 1.0)


def create_shadow_layer(
    mask: torch.Tensor,
    layer_index: int,
    rasterization_levels: int,
    shadow_distance: float,
    shadow_angle: float,
    shadow_blur: float,
    shadow_opacity: float,
    shadow_color: Sequence[float],
    depth_scaled_shadow: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return straight shadow RGB [B, H, W, 3] and alpha [B, H, W, 1]."""
    angle_rad = math.radians(shadow_angle)
    dx = math.cos(angle_rad) * shadow_distance
    dy = math.sin(angle_rad) * shadow_distance

    if depth_scaled_shadow and rasterization_levels > 1:
        depth_factor = (layer_index + 1) / rasterization_levels
        dx *= depth_factor
        dy *= depth_factor

    shifted = translate_mask(mask, dx, dy)
    blurred = gaussian_blur_mask(shifted, shadow_blur)
    alpha = (blurred * shadow_opacity).clamp(0.0, 1.0).unsqueeze(-1)

    color = _color_tensor(shadow_color, mask.device, mask.dtype)
    b, h, w, _ = alpha.shape
    rgb = color.expand(b, h, w, 3)
    return rgb, alpha


def compute_layer_offset(
    layer_index: int,
    rasterization_levels: int,
    offset_x: float,
    offset_y: float,
    offset_mode: str,
) -> tuple[float, float]:
    """Return pixel offset for parallax shifting."""
    if offset_mode == "uniform":
        return offset_x, offset_y

    if rasterization_levels <= 1:
        return 0.0, 0.0

    depth_factor = layer_index / (rasterization_levels - 1)
    return offset_x * depth_factor, offset_y * depth_factor


def shift_image(image: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Translate an image [B, H, W, C] with zero fill."""
    b, h, w, c = image.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=image.device, dtype=image.dtype),
        torch.linspace(-1, 1, w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    shift_x = (2.0 * dx) / max(w - 1, 1)
    shift_y = (2.0 * dy) / max(h - 1, 1)
    grid = torch.stack((grid_x + shift_x, grid_y + shift_y), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    sampled = F.grid_sample(image.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.permute(0, 2, 3, 1)


def alpha_composite_over(base: torch.Tensor, layer: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Composite layer over base using alpha [B, H, W, 1]."""
    return base * (1.0 - alpha) + layer * alpha
