"""Layer fill modes for paper-cut depth layers."""

from __future__ import annotations

import math

import torch


def _masked_straight_rgb(color_layers: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Convert premultiplied layer RGB to straight RGB for color operations."""
    alpha = masks.unsqueeze(-1)
    return torch.where(
        alpha > 1e-6,
        color_layers / alpha.clamp(min=1e-6),
        torch.zeros_like(color_layers),
    )


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _premultiply(straight_rgb: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return straight_rgb.clamp(0.0, 1.0) * masks.unsqueeze(-1)


def _rgb_cube_zone_ids(rgb: torch.Tensor, zones: int) -> torch.Tensor:
    """Assign each pixel to a 3D RGB cube bin."""
    side = max(2, int(round(zones ** (1.0 / 3.0))))
    r = (rgb[..., 0] * side).long().clamp(0, side - 1)
    g = (rgb[..., 1] * side).long().clamp(0, side - 1)
    b = (rgb[..., 2] * side).long().clamp(0, side - 1)
    return r * side * side + g * side + b


def _luminance_chroma_zone_ids(rgb: torch.Tensor, zones: int) -> torch.Tensor:
    """Assign pixels using luminance and chroma bins."""
    lum_bins = max(2, int(round(math.sqrt(zones))))
    chr_bins = max(1, zones // lum_bins)

    lum = _luminance(rgb)
    max_c = rgb.max(dim=-1).values
    min_c = rgb.min(dim=-1).values
    chroma = (max_c - min_c).clamp(0.0, 1.0)

    lid = (lum * lum_bins).long().clamp(0, lum_bins - 1)
    cid = (chroma * chr_bins).long().clamp(0, chr_bins - 1)
    return lid * chr_bins + cid


def _flatten_zones_in_layer(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    zones: int,
    zone_space: str,
) -> torch.Tensor:
    """Fill each color zone inside one depth layer with its average RGB."""
    flat = torch.zeros_like(rgb)
    active = mask > 0.01
    if not active.any():
        return flat

    if zone_space == "rgb_cube":
        zone_id = _rgb_cube_zone_ids(rgb, zones)
    else:
        zone_id = _luminance_chroma_zone_ids(rgb, zones)

    zone_id = zone_id.clone()
    zone_id[~active] = -1

    max_zone = int(zone_id[active].max().item()) + 1
    pixels = rgb[active]
    ids = zone_id[active].long()

    counts = torch.zeros(max_zone, device=rgb.device, dtype=rgb.dtype)
    sums = torch.zeros(max_zone, 3, device=rgb.device, dtype=rgb.dtype)
    counts.scatter_add_(0, ids, torch.ones(ids.shape[0], device=rgb.device, dtype=rgb.dtype))
    sums.scatter_add_(0, ids.unsqueeze(-1).expand(-1, 3), pixels)

    means = sums / counts.clamp(min=1.0).unsqueeze(-1)
    filled = means[zone_id.clamp(min=0)]
    return filled * active.unsqueeze(-1).float()


def _apply_color_adjustments(
    color_layers: torch.Tensor,
    masks: torch.Tensor,
    color_brightness: float,
    color_contrast: float,
    color_saturation: float,
) -> torch.Tensor:
    if (
        abs(color_brightness) < 1e-6
        and abs(color_contrast - 1.0) < 1e-6
        and abs(color_saturation - 1.0) < 1e-6
    ):
        return color_layers

    straight = _masked_straight_rgb(color_layers, masks)
    gray = _luminance(straight).unsqueeze(-1)
    adjusted = gray + (straight - gray) * float(color_saturation)
    adjusted = (adjusted - 0.5) * float(color_contrast) + 0.5
    adjusted = adjusted + float(color_brightness)
    return _premultiply(adjusted, masks)


def apply_layer_color_mode(
    color_layers: torch.Tensor,
    masks: torch.Tensor,
    image: torch.Tensor,
    layer_color_mode: str,
    color_zones_per_layer: int,
    color_zone_space: str,
    color_brightness: float = 0.0,
    color_contrast: float = 1.0,
    color_saturation: float = 1.0,
    black_white_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Transform extracted color layers according to the selected paper fill style.

    color_layers: [B, L, H, W, C]
    masks: [B, L, H, W]
    image: [B, H, W, C]
    """
    batch_size, levels, height, width, channels = color_layers.shape

    if layer_color_mode == "flat_average":
        weights = masks.unsqueeze(-1)
        sums = (image.unsqueeze(1) * weights).sum(dim=(2, 3), keepdim=True)
        counts = weights.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        averages = sums / counts
        result = averages.expand(batch_size, levels, height, width, channels) * weights

    elif layer_color_mode == "color_zones":
        zones = max(2, color_zones_per_layer)
        result = color_layers.clone()
        for batch_idx in range(batch_size):
            frame_rgb = image[batch_idx]
            for layer_idx in range(levels):
                result[batch_idx, layer_idx] = _flatten_zones_in_layer(
                    frame_rgb,
                    masks[batch_idx, layer_idx],
                    zones,
                    color_zone_space,
                )

    elif layer_color_mode == "grayscale":
        gray = _luminance(image).unsqueeze(-1).expand(batch_size, height, width, channels)
        result = gray.unsqueeze(1) * masks.unsqueeze(-1)

    elif layer_color_mode == "black_white":
        gray = _luminance(image)
        bw = (gray >= float(black_white_threshold)).to(image.dtype)
        bw_rgb = bw.unsqueeze(-1).expand(batch_size, height, width, channels)
        result = bw_rgb.unsqueeze(1) * masks.unsqueeze(-1)

    elif layer_color_mode == "original":
        result = color_layers

    else:
        raise ValueError(f"Unknown layer_color_mode: {layer_color_mode}")

    return _apply_color_adjustments(
        result,
        masks,
        color_brightness,
        color_contrast,
        color_saturation,
    )
