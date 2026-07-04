"""End-to-end depth rasterization pipeline."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .color_layers import apply_layer_color_mode
from .composite import (
    apply_outlines_to_layers,
    false_color_debug_preview,
    flatten_layers_batch,
    render_composite_frame,
)
from .depth import (
    compute_depth_range,
    gaussian_blur_depth,
    image_to_depth,
    layer_ids_to_quantized_image,
    normalize_depth,
    resize_depth_to_match,
    sanitize_depth,
    temporal_smooth_depth,
)
from .masks import (
    create_layer_masks,
    extract_color_layers,
    process_layer_masks,
    smooth_layer_assignments,
)


def depth_rasterize_layers(
    image_batch: torch.Tensor,
    depth_map: torch.Tensor,
    rasterization_levels: int = 16,
    region_mask: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
    invert_depth: bool = False,
    auto_normalize_depth: bool = True,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    normalization_mode: str = "per_frame",
    depth_gamma: float = 1.0,
    depth_blur: float = 1.0,
    temporal_smoothing: float = 0.0,
    layer_smoothing: float = 0.0,
    mask_feather: float = 1.0,
    soft_masks: bool = True,
    mask_expand: int = 0,
    mask_erode: int = 0,
    output_mode: str = "composite",
    selected_layer: int = 0,
    layer_order: str = "near_to_far",
    enable_outline: bool = False,
    outline_width: int = 2,
    outline_opacity: float = 0.75,
    outline_color: Sequence[float] = (0.0, 0.0, 0.0),
    outline_mode: str = "outer",
    enable_shadow: bool = False,
    shadow_distance: float = 8.0,
    shadow_angle: float = 135.0,
    shadow_blur: float = 6.0,
    shadow_opacity: float = 0.35,
    shadow_color: Sequence[float] = (0.0, 0.0, 0.0),
    depth_scaled_shadow: bool = True,
    enable_layer_offset: bool = False,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_mode: str = "depth_scaled",
    layer_color_mode: str = "original",
    color_zones_per_layer: int = 4,
    color_zone_space: str = "luminance_chroma",
    color_brightness: float = 0.0,
    color_contrast: float = 1.0,
    color_saturation: float = 1.0,
    black_white_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Rasterize depth into color-preserved layers.

    Returns dict with keys: layered_images, layer_masks, depth_quantized, composited_image, debug_preview.
    """
    if image_batch.ndim != 4:
        raise ValueError("image_batch must have shape [B, H, W, C]")

    batch_size, height, width, _ = image_batch.shape
    device = image_batch.device
    dtype = image_batch.dtype

    if depth_map is None:
        raise ValueError("depth_map is required")

    depth = image_to_depth(depth_map)
    if depth.shape[0] == 1 and batch_size > 1:
        depth = depth.expand(batch_size, -1, -1)
    elif depth.shape[0] != batch_size:
        raise ValueError(
            f"depth_map batch ({depth.shape[0]}) must match image batch ({batch_size}) or be 1"
        )

    depth = resize_depth_to_match(depth, height, width)
    depth = sanitize_depth(depth)

    if depth_blur > 0:
        depth = gaussian_blur_depth(depth, depth_blur)

    if normalization_mode == "whole_sequence":
        seq_min, seq_max = compute_depth_range(
            depth,
            auto_normalize_depth,
            depth_min,
            depth_max,
            "whole_sequence",
        )
    else:
        seq_min = seq_max = None

    previous_depth: Optional[torch.Tensor] = None
    all_layers: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_layer_ids: list[torch.Tensor] = []
    composites: list[torch.Tensor] = []

    for frame_idx in range(batch_size):
        frame_depth = depth[frame_idx : frame_idx + 1]

        if normalization_mode == "whole_sequence":
            d_min, d_max = seq_min, seq_max
        else:
            d_min, d_max = compute_depth_range(
                frame_depth,
                auto_normalize_depth,
                depth_min,
                depth_max,
                normalization_mode,
            )

        depth_norm = normalize_depth(frame_depth, d_min, d_max, invert_depth, depth_gamma)
        depth_norm = temporal_smooth_depth(depth_norm, previous_depth, temporal_smoothing)
        previous_depth = depth_norm.detach()

        depth_for_layers, layer_id = smooth_layer_assignments(
            depth_norm,
            rasterization_levels,
            layer_smoothing,
        )
        masks = create_layer_masks(
            layer_id,
            depth_for_layers,
            rasterization_levels,
            mask_feather,
            soft_masks,
        )

        if region_mask is not None:
            region = region_mask
            if region.ndim == 2:
                region = region.unsqueeze(0)
            region_frame = region[frame_idx if region.shape[0] > 1 else 0]
        else:
            region_frame = None

        masks = process_layer_masks(
            masks,
            region_frame,
            mask_expand,
            mask_erode,
        )

        frame_image = image_batch[frame_idx : frame_idx + 1]
        color_layers = extract_color_layers(frame_image, masks)
        color_layers = apply_layer_color_mode(
            color_layers,
            masks,
            frame_image,
            layer_color_mode,
            color_zones_per_layer,
            color_zone_space,
            color_brightness,
            color_contrast,
            color_saturation,
            black_white_threshold,
        )

        if enable_outline and output_mode in ("all_layers", "single_layer"):
            color_layers = apply_outlines_to_layers(
                color_layers,
                masks,
                layer_id,
                outline_width,
                outline_mode,
                outline_color,
                outline_opacity,
            )

        composite = None
        if output_mode == "composite":
            bg = background
            if bg is not None and bg.shape[0] == 1 and batch_size > 1:
                bg = bg.expand(batch_size, -1, -1, -1)
            composite = render_composite_frame(
                color_layers,
                masks,
                layer_id,
                rasterization_levels,
                bg[frame_idx : frame_idx + 1] if bg is not None else None,
                enable_outline,
                outline_width,
                outline_mode,
                outline_color,
                outline_opacity,
                enable_shadow,
                shadow_distance,
                shadow_angle,
                shadow_blur,
                shadow_opacity,
                shadow_color,
                depth_scaled_shadow,
                enable_layer_offset,
                offset_x,
                offset_y,
                offset_mode,
                layer_order,
            )
            composites.append(composite)

        all_layers.append(color_layers)
        all_masks.append(masks)
        all_layer_ids.append(layer_id)

    layers_stack = torch.cat(all_layers, dim=0)
    masks_stack = torch.cat(all_masks, dim=0)
    layer_id_stack = torch.cat(all_layer_ids, dim=0)
    depth_quantized = layer_ids_to_quantized_image(layer_id_stack, rasterization_levels)
    debug_preview = false_color_debug_preview(layer_id_stack, rasterization_levels)

    layered_images = flatten_layers_batch(layers_stack)
    layer_masks = flatten_layers_batch(masks_stack)

    if output_mode == "single_layer":
        clamped = max(0, min(selected_layer, rasterization_levels - 1))
        layered_images = layers_stack[:, clamped]
        layer_masks = masks_stack[:, clamped]

    elif output_mode == "masks_only":
        layered_images = torch.zeros(batch_size, height, width, 3, device=device, dtype=dtype)
        layer_masks = masks_stack[:, max(0, min(selected_layer, rasterization_levels - 1))]

    elif output_mode == "debug":
        layered_images = debug_preview
        layer_masks = masks_stack[:, 0]

    if output_mode == "composite":
        if composites:
            composited_image = torch.cat(composites, dim=0)
        else:
            composited_image = torch.zeros(batch_size, height, width, 3, device=device, dtype=dtype)
    else:
        composited_image = layered_images if output_mode == "single_layer" else torch.zeros(
            batch_size, height, width, 3, device=device, dtype=dtype
        )

    return {
        "layered_images": layered_images,
        "layer_masks": layer_masks,
        "depth_quantized": depth_quantized,
        "composited_image": composited_image,
        "debug_preview": debug_preview,
    }
