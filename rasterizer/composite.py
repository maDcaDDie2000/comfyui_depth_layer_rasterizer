"""Composite rendering and debug visualization."""

from __future__ import annotations

from typing import Sequence

import torch

from .effects import (
    alpha_composite_over,
    apply_outline_to_layer,
    compute_layer_offset,
    create_outline_mask,
    create_shadow_layer,
    shift_image,
)


def layer_order_indices(rasterization_levels: int, layer_order: str) -> list[int]:
    """Return layer indices in compositing order (far to near on top)."""
    indices = list(range(rasterization_levels))
    if layer_order == "near_to_far":
        return list(reversed(indices))
    return indices


def render_composite_frame(
    color_layers: torch.Tensor,
    masks: torch.Tensor,
    layer_id: torch.Tensor,
    rasterization_levels: int,
    background: torch.Tensor | None,
    enable_outline: bool,
    outline_width: int,
    outline_mode: str,
    outline_color: Sequence[float],
    outline_opacity: float,
    enable_shadow: bool,
    shadow_distance: float,
    shadow_angle: float,
    shadow_blur: float,
    shadow_opacity: float,
    shadow_color: Sequence[float],
    depth_scaled_shadow: bool,
    enable_layer_offset: bool,
    offset_x: float,
    offset_y: float,
    offset_mode: str,
    layer_order: str,
) -> torch.Tensor:
    """Composite all layers for one batch item paths handled per frame in B dim."""
    b, levels, h, w, _ = color_layers.shape

    results = []
    for batch_idx in range(b):
        if background is None:
            canvas = torch.zeros((h, w, 3), device=color_layers.device, dtype=color_layers.dtype)
        else:
            canvas = background[batch_idx if background.shape[0] > 1 else 0].clone()

        order = layer_order_indices(levels, layer_order)
        for layer_index in order:
            mask = masks[batch_idx, layer_index]
            layer_rgb = color_layers[batch_idx, layer_index]
            layer_active = mask > 1e-6
            # Composite uses straight RGB with hard opacity per assigned pixel so soft-mask
            # feathering (for layer exports) does not leave black gaps in the stack.
            alpha = layer_active.float().unsqueeze(-1)
            straight_rgb = torch.where(
                layer_active.unsqueeze(-1),
                layer_rgb / mask.unsqueeze(-1).clamp(min=1e-6),
                torch.zeros_like(layer_rgb),
            )

            if enable_layer_offset:
                dx, dy = compute_layer_offset(
                    layer_index,
                    rasterization_levels,
                    offset_x,
                    offset_y,
                    offset_mode,
                )
                straight_rgb = shift_image(straight_rgb.unsqueeze(0), dx, dy).squeeze(0)
                alpha = shift_image(alpha.unsqueeze(0), dx, dy).squeeze(0)
                mask = alpha.squeeze(-1)

            if enable_shadow:
                shadow_rgb, shadow_alpha = create_shadow_layer(
                    mask.unsqueeze(0),
                    layer_index,
                    rasterization_levels,
                    shadow_distance,
                    shadow_angle,
                    shadow_blur,
                    shadow_opacity,
                    shadow_color,
                    depth_scaled_shadow,
                )
                shadow_rgb = shadow_rgb.squeeze(0)
                shadow_alpha = shadow_alpha.squeeze(0)
                canvas = alpha_composite_over(canvas, shadow_rgb, shadow_alpha)

            if enable_outline:
                outline = create_outline_mask(
                    mask.unsqueeze(0),
                    layer_id[batch_idx : batch_idx + 1] if outline_mode == "depth_boundary" else None,
                    outline_width,
                    outline_mode,
                ).squeeze(0)
                straight_rgb = apply_outline_to_layer(
                    straight_rgb.unsqueeze(0),
                    outline.unsqueeze(0),
                    outline_color,
                    outline_opacity,
                ).squeeze(0)

            canvas = alpha_composite_over(canvas, straight_rgb, alpha)

        results.append(canvas)

    return torch.stack(results, dim=0)


def apply_outlines_to_layers(
    color_layers: torch.Tensor,
    masks: torch.Tensor,
    layer_id: torch.Tensor,
    outline_width: int,
    outline_mode: str,
    outline_color: Sequence[float],
    outline_opacity: float,
) -> torch.Tensor:
    """Apply outlines to each layer in the stack."""
    b, levels, h, w, c = color_layers.shape
    outlined = color_layers.clone()

    for batch_idx in range(b):
        for layer_index in range(levels):
            mask = masks[batch_idx, layer_index]
            outline = create_outline_mask(
                mask.unsqueeze(0),
                layer_id[batch_idx : batch_idx + 1] if outline_mode == "depth_boundary" else None,
                outline_width,
                outline_mode,
            ).squeeze(0)
            outlined[batch_idx, layer_index] = apply_outline_to_layer(
                color_layers[batch_idx, layer_index].unsqueeze(0),
                outline.unsqueeze(0),
                outline_color,
                outline_opacity,
            ).squeeze(0)

    return outlined


def false_color_debug_preview(layer_id: torch.Tensor, rasterization_levels: int) -> torch.Tensor:
    """Map layer IDs to a saturated rainbow [B, H, W, 3]."""
    levels = max(rasterization_levels, 1)
    t = (layer_id.float() / max(levels - 1, 1)).unsqueeze(-1)

    r = (torch.sin(t * 6.28318 + 0.0) * 0.5 + 0.5).clamp(0.0, 1.0)
    g = (torch.sin(t * 6.28318 + 2.09439) * 0.5 + 0.5).clamp(0.0, 1.0)
    b = (torch.sin(t * 6.28318 + 4.18879) * 0.5 + 0.5).clamp(0.0, 1.0)
    return torch.cat([r, g, b], dim=-1)


def flatten_layers_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten [B, L, ...] -> [B*L, ...] preserving frame-major order."""
    b, levels = tensor.shape[:2]
    rest = tensor.shape[2:]
    return tensor.reshape(b * levels, *rest)
