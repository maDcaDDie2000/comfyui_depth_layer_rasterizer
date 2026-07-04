"""Layer mask creation, feathering, and cleanup."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def smoothstep(edge0: torch.Tensor | float, edge1: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
    """Standard smoothstep with safe denominator."""
    denom = edge0 - edge1
    if isinstance(denom, torch.Tensor):
        denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
    elif abs(denom) < 1e-8:
        denom = 1.0

    t = ((x - edge1) / denom).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def create_binary_layer_masks(layer_id: torch.Tensor, rasterization_levels: int) -> torch.Tensor:
    """Hard masks [B, L, H, W] where layer_id matches each level."""
    levels = torch.arange(rasterization_levels, device=layer_id.device, dtype=layer_id.dtype)
    return (layer_id.unsqueeze(1) == levels.view(1, -1, 1, 1)).float()


def create_soft_layer_masks(
    depth_norm: torch.Tensor,
    rasterization_levels: int,
    mask_feather: float,
) -> torch.Tensor:
    """Soft falloff masks around each depth bin center [B, L, H, W]."""
    levels = rasterization_levels
    bin_size = 1.0 / levels
    half_bin = bin_size * 0.5
    feather = max(mask_feather, 0.0) * half_bin

    layer_indices = torch.arange(levels, device=depth_norm.device, dtype=depth_norm.dtype)
    bin_centers = (layer_indices + 0.5) * bin_size
    distance = (depth_norm.unsqueeze(1) - bin_centers.view(1, -1, 1, 1)).abs()

    edge0 = half_bin
    edge1 = max(half_bin - feather, 0.0)
    return smoothstep(edge0, edge1, distance)


def create_layer_masks(
    layer_id: torch.Tensor,
    depth_norm: torch.Tensor,
    rasterization_levels: int,
    mask_feather: float,
    soft_masks: bool,
) -> torch.Tensor:
    """Create layer masks using binary or soft mode."""
    if soft_masks and mask_feather > 0:
        masks = create_soft_layer_masks(depth_norm, rasterization_levels, mask_feather)
        binary = create_binary_layer_masks(layer_id, rasterization_levels)
        # Keep soft weights inside assigned bins; zero outside hard assignment.
        return masks * binary

    return create_binary_layer_masks(layer_id, rasterization_levels)


def morph_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Max-pool dilation for masks [B, H, W] or [B, L, H, W]."""
    if radius <= 0:
        return mask

    kernel = radius * 2 + 1
    if mask.ndim == 3:
        x = mask.unsqueeze(1)
        out = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)
        return out.squeeze(1)

    b, levels, h, w = mask.shape
    x = mask.reshape(b * levels, 1, h, w)
    out = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)
    return out.reshape(b, levels, h, w)


def morph_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Min-pool erosion for masks."""
    if radius <= 0:
        return mask
    return 1.0 - morph_dilate(1.0 - mask, radius)


def apply_region_mask(masks: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """Limit masks to a region [B, H, W]."""
    if region.ndim == 2:
        region = region.unsqueeze(0)
    if region.shape[0] == 1 and masks.shape[0] > 1:
        region = region.expand(masks.shape[0], -1, -1)
    return masks * region.unsqueeze(1)


def remove_small_islands(masks: torch.Tensor, min_size: int) -> torch.Tensor:
    """Drop connected components smaller than min_size (OpenCV when available)."""
    if min_size <= 1:
        return masks

    try:
        import cv2
        import numpy as np
    except ImportError:
        return masks

    cleaned = masks.clone()
    b, levels, h, w = cleaned.shape

    for batch_idx in range(b):
        for level_idx in range(levels):
            component = cleaned[batch_idx, level_idx]
            binary = (component > 0.5).cpu().numpy().astype(np.uint8)
            if binary.max() == 0:
                continue

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            keep = np.zeros_like(binary, dtype=np.float32)
            for label_idx in range(1, num_labels):
                if stats[label_idx, cv2.CC_STAT_AREA] >= min_size:
                    keep[labels == label_idx] = 1.0

            cleaned[batch_idx, level_idx] = torch.from_numpy(keep).to(
                device=component.device,
                dtype=component.dtype,
            )

    return cleaned


def process_layer_masks(
    masks: torch.Tensor,
    region_mask: torch.Tensor | None,
    mask_expand: int,
    mask_erode: int,
    remove_islands: bool,
    island_min_size: int,
) -> torch.Tensor:
    """Apply morphological and cleanup operations to layer masks."""
    if region_mask is not None:
        masks = apply_region_mask(masks, region_mask)

    if mask_expand > 0:
        masks = morph_dilate(masks, mask_expand)

    if mask_erode > 0:
        masks = morph_erode(masks, mask_erode)

    if remove_islands:
        masks = remove_small_islands(masks, island_min_size)

    return masks.clamp(0.0, 1.0)


def extract_color_layers(image: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Multiply RGB image by per-layer masks -> [B, L, H, W, C]."""
    mask_expanded = masks.unsqueeze(-1)
    return image.unsqueeze(1) * mask_expanded
