"""End-to-end depth rasterization pipeline."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch

from .composite import (
    apply_outlines_to_layers,
    false_color_debug_preview,
    flatten_layers_batch,
    render_composite_frame,
)
from .depth import (
    compute_depth_range,
    depth_to_layer_ids,
    gaussian_blur_depth,
    image_to_depth,
    layer_ids_to_quantized_image,
    normalize_depth,
    resize_depth_to_match,
    sanitize_depth,
    temporal_smooth_depth,
)
from .masks import create_layer_masks, extract_color_layers, process_layer_masks


DepthPredictFn = Callable[[torch.Tensor], torch.Tensor]


def predict_depth_from_model(depth_model: Any, images: torch.Tensor) -> torch.Tensor:
    """Invoke a DEPTH_MODEL handle or callable on an image batch."""
    if depth_model is None:
        raise ValueError("depth_model is required when depth_map is not provided")

    if hasattr(depth_model, "predict"):
        depth = depth_model.predict(images)
    elif callable(depth_model):
        depth = depth_model(images)
    elif isinstance(depth_model, dict) and callable(depth_model.get("predict")):
        depth = depth_model["predict"](images)
    else:
        raise TypeError("depth_model must be callable or expose a .predict(images) method")

    if depth.ndim == 4:
        depth = depth.mean(dim=-1)
    return depth.float()


def depth_rasterize_layers(
    image_batch: torch.Tensor,
    rasterization_levels: int = 16,
    depth_model: Any = None,
    depth_map: Optional[torch.Tensor] = None,
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
    mask_feather: float = 1.0,
    soft_masks: bool = True,
    mask_expand: int = 0,
    mask_erode: int = 0,
    remove_small_islands: bool = True,
    island_min_size: int = 32,
    output_mode: str = "all_layers",
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
    manual_offset_x: float = 0.0,
    manual_offset_y: float = 0.0,
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

    if depth_map is not None:
        depth = image_to_depth(depth_map)
        if depth.shape[0] == 1 and batch_size > 1:
            depth = depth.expand(batch_size, -1, -1)
        elif depth.shape[0] != batch_size:
            raise ValueError(
                f"depth_map batch ({depth.shape[0]}) must match image batch ({batch_size}) or be 1"
            )
    else:
        depth = predict_depth_from_model(depth_model, image_batch)

    depth = resize_depth_to_match(depth, height, width)
    depth = sanitize_depth(depth)

    if depth_blur > 0:
        depth = gaussian_blur_depth(depth, depth_blur)

    sequence_source = depth if normalization_mode == "whole_sequence" else None
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
    all_depth_norm: list[torch.Tensor] = []
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

        layer_id = depth_to_layer_ids(depth_norm, rasterization_levels)
        masks = create_layer_masks(
            layer_id,
            depth_norm,
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
            remove_small_islands,
            island_min_size,
        )

        frame_image = image_batch[frame_idx : frame_idx + 1]
        color_layers = extract_color_layers(frame_image, masks)

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
                manual_offset_x,
                manual_offset_y,
                layer_order,
            )
            composites.append(composite)

        all_layers.append(color_layers)
        all_masks.append(masks)
        all_layer_ids.append(layer_id)
        all_depth_norm.append(depth_norm)

    layers_stack = torch.cat(all_layers, dim=0)
    masks_stack = torch.cat(all_masks, dim=0)
    layer_id_stack = torch.cat(all_layer_ids, dim=0)
    depth_norm_stack = torch.cat(all_depth_norm, dim=0)

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
