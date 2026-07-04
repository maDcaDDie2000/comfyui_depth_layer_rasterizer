"""Layer fill modes: original pixels, flat average, or intra-layer color zones."""

from __future__ import annotations

import math

import torch


def _rgb_cube_zone_ids(rgb: torch.Tensor, zones: int) -> torch.Tensor:
    """Assign each pixel to a 3D RGB cube bin."""
    side = max(2, int(round(zones ** (1.0 / 3.0))))
    r = (rgb[..., 0] * side).long().clamp(0, side - 1)
    g = (rgb[..., 1] * side).long().clamp(0, side - 1)
    b = (rgb[..., 2] * side).long().clamp(0, side - 1)
    return r * side * side + g * side + b


def _luminance_chroma_zone_ids(rgb: torch.Tensor, zones: int) -> torch.Tensor:
    """Assign pixels using luminance + chroma bins (better hue separation than luminance alone)."""
    lum_bins = max(2, int(round(math.sqrt(zones))))
    chr_bins = max(1, zones // lum_bins)

    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
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
    """
    Posterize colors inside one depth layer: each color zone becomes a flat fill
    of that zone's average RGB (cel-shading / paper-cut look within the slice).
    """
    h, w, _ = rgb.shape
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
    filled = filled * active.unsqueeze(-1).float()
    return filled


def apply_layer_color_mode(
    color_layers: torch.Tensor,
    masks: torch.Tensor,
    image: torch.Tensor,
    layer_color_mode: str,
    color_zones_per_layer: int,
    color_zone_space: str,
) -> torch.Tensor:
    """
    Transform extracted color layers according to fill mode.

    color_layers: [B, L, H, W, C]
    masks: [B, L, H, W]
    image: [B, H, W, C]
    """
    if layer_color_mode == "original":
        return color_layers

    batch_size, levels, height, width, channels = color_layers.shape
    result = color_layers.clone()

    if layer_color_mode == "flat_average":
        weights = masks.unsqueeze(-1)
        sums = (image.unsqueeze(1) * weights).sum(dim=(2, 3), keepdim=True)
        counts = weights.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        averages = sums / counts
        return averages.expand(batch_size, levels, height, width, channels) * weights

    if layer_color_mode == "color_zones":
        zones = max(2, color_zones_per_layer)
        for batch_idx in range(batch_size):
            frame_rgb = image[batch_idx]
            for layer_idx in range(levels):
                layer_mask = masks[batch_idx, layer_idx]
                result[batch_idx, layer_idx] = _flatten_zones_in_layer(
                    frame_rgb,
                    layer_mask,
                    zones,
                    color_zone_space,
                )
        return result

    raise ValueError(f"Unknown layer_color_mode: {layer_color_mode}")
