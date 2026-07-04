"""Depth map extraction, normalization, blur, and temporal smoothing."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def image_to_depth(depth_map: torch.Tensor) -> torch.Tensor:
    """Convert an IMAGE tensor to depth [B, H, W] in float32."""
    if depth_map.ndim == 2:
        depth = depth_map.unsqueeze(0)
    elif depth_map.ndim == 3:
        depth = depth_map
    elif depth_map.ndim == 4:
        depth = depth_map.mean(dim=-1)
    else:
        raise ValueError(f"Unsupported depth_map rank: {depth_map.ndim}")

    return depth.float()


def resize_depth_to_match(depth: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize depth [B, H, W] to target resolution."""
    if depth.shape[-2] == height and depth.shape[-1] == width:
        return depth

    depth_4d = depth.unsqueeze(1)
    resized = F.interpolate(depth_4d, size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(1)


def sanitize_depth(depth: torch.Tensor, fill_value: Optional[float] = None) -> torch.Tensor:
    """Replace NaN/Inf depth values with a fill (median of valid pixels per frame)."""
    depth = depth.clone()
    invalid = ~torch.isfinite(depth)

    if not invalid.any():
        return depth

    if fill_value is not None:
        depth[invalid] = fill_value
        return depth

    for batch_idx in range(depth.shape[0]):
        frame = depth[batch_idx]
        frame_invalid = invalid[batch_idx]
        valid = frame[~frame_invalid]
        replacement = valid.median() if valid.numel() > 0 else torch.tensor(1.0, device=depth.device)
        frame[frame_invalid] = replacement
        depth[batch_idx] = frame

    return depth


def gaussian_blur_depth(depth: torch.Tensor, radius: float) -> torch.Tensor:
    """Apply separable Gaussian blur to depth maps."""
    if radius <= 0:
        return depth

    kernel_size = max(3, int(math.ceil(radius * 2)) * 2 + 1)
    sigma = max(radius * 0.5, 1e-3)

    coords = torch.arange(kernel_size, device=depth.device, dtype=depth.dtype) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    depth_4d = depth.unsqueeze(1)
    pad = kernel_size // 2

    kh = kernel_1d.view(1, 1, 1, -1)
    kv = kernel_1d.view(1, 1, -1, 1)

    blurred = F.conv2d(F.pad(depth_4d, (pad, pad, 0, 0), mode="reflect"), kh)
    blurred = F.conv2d(F.pad(blurred, (0, 0, pad, pad), mode="reflect"), kv)
    return blurred.squeeze(1)


def compute_depth_range(
    depth: torch.Tensor,
    auto_normalize: bool,
    depth_min: Optional[float],
    depth_max: Optional[float],
    normalization_mode: str,
    sequence_depth: Optional[torch.Tensor] = None,
) -> tuple[float, float]:
    """Resolve min/max depth used for normalization."""
    if not auto_normalize and normalization_mode == "manual":
        d_min = 0.0 if depth_min is None else float(depth_min)
        d_max = 1.0 if depth_max is None else float(depth_max)
        return d_min, d_max

    source = sequence_depth if normalization_mode == "whole_sequence" and sequence_depth is not None else depth
    d_min = float(source.min().item())
    d_max = float(source.max().item())
    return d_min, d_max


def normalize_depth(
    depth: torch.Tensor,
    d_min: float,
    d_max: float,
    invert_depth: bool,
    depth_gamma: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Normalize depth to [0, 1] with optional invert and gamma."""
    raw_span = d_max - d_min

    if raw_span < epsilon:
        depth_norm = torch.full_like(depth, 0.5)
    else:
        depth_norm = (depth - d_min) / raw_span
        depth_norm = depth_norm.clamp(0.0, 1.0)

    if invert_depth:
        depth_norm = 1.0 - depth_norm

    if depth_gamma != 1.0:
        depth_norm = depth_norm.clamp(min=0.0).pow(depth_gamma)

    return depth_norm


def temporal_smooth_depth(
    current: torch.Tensor,
    previous: Optional[torch.Tensor],
    temporal_smoothing: float,
) -> torch.Tensor:
    """Exponential blend with previous normalized depth frame."""
    if previous is None or temporal_smoothing <= 0:
        return current

    weight = float(temporal_smoothing)
    return current * (1.0 - weight) + previous * weight


def depth_to_layer_ids(depth_norm: torch.Tensor, rasterization_levels: int) -> torch.Tensor:
    """Quantize normalized depth into discrete layer indices [B, H, W]."""
    levels = max(rasterization_levels, 1)
    layer_id = torch.floor(depth_norm * levels).long()
    return layer_id.clamp(0, levels - 1)


def layer_ids_to_quantized_image(layer_id: torch.Tensor, rasterization_levels: int) -> torch.Tensor:
    """Posterized grayscale depth visualization [B, H, W, 3]."""
    levels = max(rasterization_levels, 1)
    denom = max(levels - 1, 1)
    gray = layer_id.float() / denom
    return gray.unsqueeze(-1).expand(-1, -1, -1, 3)
