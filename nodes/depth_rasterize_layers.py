"""ComfyUI node definitions for depth layer rasterization."""

from __future__ import annotations

import torch

from ..rasterizer.pipeline import depth_rasterize_layers
from .tooltips import NODE_DESCRIPTION, OUTPUT_TOOLTIPS, build_depth_rasterize_input_types


def _normalize_optional_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    return mask


class NullDepthModel:
    """Placeholder DEPTH_MODEL when using a precomputed depth_map input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("DEPTH_MODEL",)
    RETURN_NAMES = ("depth_model",)
    FUNCTION = "execute"
    CATEGORY = "depth/layers"
    DESCRIPTION = "Outputs an empty DEPTH_MODEL handle. Connect when your workflow supplies depth via depth_map only."

    def execute(self):
        return (None,)


class DepthRasterizeLayers:
    """Quantize depth into slices with optional flat or posterized color fills."""

    @classmethod
    def INPUT_TYPES(cls):
        return build_depth_rasterize_input_types()

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "layered_images",
        "layer_masks",
        "depth_quantized",
        "composited_image",
        "debug_preview",
    )
    OUTPUT_TOOLTIPS = OUTPUT_TOOLTIPS
    FUNCTION = "process"
    CATEGORY = "depth/layers"
    DESCRIPTION = NODE_DESCRIPTION

    def process(
        self,
        image,
        rasterization_levels,
        output_mode,
        depth_model=None,
        depth_map=None,
        region_mask=None,
        background=None,
        selected_layer=0,
        invert_depth=False,
        auto_normalize_depth=True,
        depth_min=0.0,
        depth_max=1.0,
        normalization_mode="per_frame",
        depth_gamma=1.0,
        depth_blur=1.0,
        temporal_smoothing=0.0,
        layer_smoothing=0.0,
        layer_color_mode="original",
        color_zones_per_layer=4,
        color_zone_space="luminance_chroma",
        mask_feather=1.0,
        soft_masks=True,
        mask_expand=0,
        mask_erode=0,
        remove_small_islands=True,
        island_min_size=32,
        layer_order="near_to_far",
        enable_outline=False,
        outline_width=2,
        outline_opacity=0.75,
        outline_mode="outer",
        outline_color_r=0.0,
        outline_color_g=0.0,
        outline_color_b=0.0,
        enable_shadow=False,
        shadow_distance=8.0,
        shadow_angle=135.0,
        shadow_blur=6.0,
        shadow_opacity=0.35,
        shadow_color_r=0.0,
        shadow_color_g=0.0,
        shadow_color_b=0.0,
        depth_scaled_shadow=True,
        enable_layer_offset=False,
        offset_x=0.0,
        offset_y=0.0,
        offset_mode="depth_scaled",
        manual_offset_x=0.0,
        manual_offset_y=0.0,
    ):
        if depth_map is None and depth_model is None:
            raise ValueError("Provide depth_map and/or depth_model (connect Null Depth Model if using depth_map only)")

        manual_min = None if auto_normalize_depth else depth_min
        manual_max = None if auto_normalize_depth else depth_max
        if normalization_mode == "manual":
            manual_min = depth_min
            manual_max = depth_max

        result = depth_rasterize_layers(
            image_batch=image,
            rasterization_levels=rasterization_levels,
            depth_model=depth_model,
            depth_map=depth_map,
            region_mask=_normalize_optional_mask(region_mask),
            background=background,
            invert_depth=invert_depth,
            auto_normalize_depth=auto_normalize_depth,
            depth_min=manual_min,
            depth_max=manual_max,
            normalization_mode=normalization_mode,
            depth_gamma=depth_gamma,
            depth_blur=depth_blur,
            temporal_smoothing=temporal_smoothing,
            layer_smoothing=layer_smoothing,
            layer_color_mode=layer_color_mode,
            color_zones_per_layer=color_zones_per_layer,
            color_zone_space=color_zone_space,
            mask_feather=mask_feather,
            soft_masks=soft_masks,
            mask_expand=mask_expand,
            mask_erode=mask_erode,
            remove_small_islands=remove_small_islands,
            island_min_size=island_min_size,
            output_mode=output_mode,
            selected_layer=selected_layer,
            layer_order=layer_order,
            enable_outline=enable_outline,
            outline_width=outline_width,
            outline_opacity=outline_opacity,
            outline_color=(outline_color_r, outline_color_g, outline_color_b),
            outline_mode=outline_mode,
            enable_shadow=enable_shadow,
            shadow_distance=shadow_distance,
            shadow_angle=shadow_angle,
            shadow_blur=shadow_blur,
            shadow_opacity=shadow_opacity,
            shadow_color=(shadow_color_r, shadow_color_g, shadow_color_b),
            depth_scaled_shadow=depth_scaled_shadow,
            enable_layer_offset=enable_layer_offset,
            offset_x=offset_x,
            offset_y=offset_y,
            offset_mode=offset_mode,
            manual_offset_x=manual_offset_x,
            manual_offset_y=manual_offset_y,
        )

        return (
            result["layered_images"],
            result["layer_masks"],
            result["depth_quantized"],
            result["composited_image"],
            result["debug_preview"],
        )


NODE_CLASS_MAPPINGS = {
    "DepthRasterizeLayers": DepthRasterizeLayers,
    "NullDepthModel": NullDepthModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthRasterizeLayers": "Depth Rasterize Layers",
    "NullDepthModel": "Null Depth Model",
}
