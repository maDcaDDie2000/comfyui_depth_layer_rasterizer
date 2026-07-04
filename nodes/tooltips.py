"""Widget tooltips and INPUT_TYPES helpers for ComfyUI nodes."""

from __future__ import annotations

TOOLTIPS: dict[str, str] = {
    "image": "RGB input image or video frame batch to turn into depth-cut paper layers.",
    "depth_map": "Aligned grayscale depth map from a depth node such as Depth Anything.",
    "region_mask": "Optional mask limiting where the paper-cut effect is applied.",
    "background": "Optional bottom plate for the composite output.",
    "rasterization_levels": "Number of depth slices to cut from the image. More slices add depth detail.",
    "output_mode": "Return all cut layers, one selected layer, the final composite, masks, or a debug depth preview.",
    "selected_layer": "Layer index for single_layer or masks_only output. 0 is the nearest layer after depth normalization.",
    "invert_depth": "Swap near and far if your depth map is reversed.",
    "depth_min": "Manual minimum depth. Used only when normalization_mode is manual.",
    "depth_max": "Manual maximum depth. Used only when normalization_mode is manual.",
    "normalization_mode": "per_frame adapts each image, whole_sequence stabilizes video batches, manual uses depth_min/depth_max.",
    "depth_gamma": "Curve the normalized depth before slicing. Higher values move more pixels toward near layers.",
    "depth_blur": "Blur the raw depth map before slicing to reduce noisy cut boundaries.",
    "temporal_smoothing": "Blend video depth with the previous frame to reduce flicker.",
    "layer_smoothing": "Smooth tiny depth islands after normalization for cleaner paper shapes.",
    "layer_color_mode": "Paper fill style: original RGB, one average color per layer, color zones, grayscale, or black/white.",
    "color_zones_per_layer": "Number of flat color patches inside each depth slice when using color_zones.",
    "color_brightness": "Brightness adjustment applied after the layer fill mode.",
    "color_contrast": "Contrast multiplier applied after the layer fill mode.",
    "color_saturation": "Saturation multiplier applied after the layer fill mode.",
    "black_white_threshold": "Luminance cutoff for black_white mode.",
    "mask_feather": "Soft edge width at layer boundaries. Set 0 for crisp hard cuts.",
    "enable_outline": "Draw a colored edge around each paper layer.",
    "outline_width": "Outline thickness in pixels.",
    "outline_opacity": "Outline blend strength.",
    "outline_color_r": "Outline red channel from 0 to 1.",
    "outline_color_g": "Outline green channel from 0 to 1.",
    "outline_color_b": "Outline blue channel from 0 to 1.",
    "enable_shadow": "Drop a blurred offset shadow behind each paper layer in composite mode.",
    "shadow_distance": "Shadow offset distance in pixels.",
    "shadow_angle": "Shadow direction in degrees. 135 casts toward the lower-left.",
    "shadow_blur": "Gaussian blur radius for the shadow mask.",
    "shadow_opacity": "Shadow opacity.",
    "shadow_color_r": "Shadow red channel from 0 to 1.",
    "shadow_color_g": "Shadow green channel from 0 to 1.",
    "shadow_color_b": "Shadow blue channel from 0 to 1.",
    "depth_scaled_shadow": "Scale shadow offset by layer depth for a stronger stacked-paper feel.",
    "enable_layer_offset": "Shift layers by depth for a subtle parallax cutout effect.",
    "offset_x": "Horizontal parallax strength in pixels.",
    "offset_y": "Vertical parallax strength in pixels.",
}

OUTPUT_TOOLTIPS = (
    "Layer images: one batch entry per frame x layer, or the selected/debug image depending on output_mode.",
    "Layer masks: alpha masks matching layered_images.",
    "Posterized grayscale depth map showing the cut depth bands.",
    "Final stacked paper-cut composite when output_mode is composite.",
    "False-color debug view of depth layer IDs.",
)

NODE_DESCRIPTION = (
    "Turn an image and depth map into layered paper-cut pieces with flat color styles, "
    "depth shadows, optional outlines, and parallax offsets."
)


def _opts(key: str, **kwargs) -> dict:
    return {**kwargs, "tooltip": TOOLTIPS[key]}


def _combo(options: list[str], key: str, default: str | None = None) -> tuple:
    kwargs = {"default": default} if default is not None else {}
    return (options, _opts(key, **kwargs))


def build_depth_rasterize_input_types() -> dict:
    """Build INPUT_TYPES dict with only the controls needed for the paper-cut effect."""
    return {
        "required": {
            "image": ("IMAGE", _opts("image")),
            "depth_map": ("IMAGE", _opts("depth_map")),
            "rasterization_levels": ("INT", _opts("rasterization_levels", default=12, min=2, max=48, step=1)),
            "output_mode": _combo(
                ["composite", "all_layers", "single_layer", "masks_only", "debug"],
                "output_mode",
                default="composite",
            ),
        },
        "optional": {
            "region_mask": ("MASK", _opts("region_mask")),
            "background": ("IMAGE", _opts("background")),
            "selected_layer": ("INT", _opts("selected_layer", default=0, min=0, max=47, step=1)),
            "invert_depth": ("BOOLEAN", _opts("invert_depth", default=False)),
            "normalization_mode": _combo(["per_frame", "whole_sequence", "manual"], "normalization_mode", default="per_frame"),
            "depth_min": ("FLOAT", _opts("depth_min", default=0.0, min=-1e6, max=1e6, step=0.001)),
            "depth_max": ("FLOAT", _opts("depth_max", default=1.0, min=-1e6, max=1e6, step=0.001)),
            "depth_gamma": ("FLOAT", _opts("depth_gamma", default=1.0, min=0.1, max=4.0, step=0.05)),
            "depth_blur": ("FLOAT", _opts("depth_blur", default=1.0, min=0.0, max=24.0, step=0.5)),
            "temporal_smoothing": ("FLOAT", _opts("temporal_smoothing", default=0.0, min=0.0, max=0.99, step=0.01)),
            "layer_smoothing": ("FLOAT", _opts("layer_smoothing", default=1.0, min=0.0, max=24.0, step=0.5)),
            "mask_feather": ("FLOAT", _opts("mask_feather", default=0.5, min=0.0, max=8.0, step=0.25)),
            "layer_color_mode": _combo(
                ["original", "flat_average", "color_zones", "grayscale", "black_white"],
                "layer_color_mode",
                default="color_zones",
            ),
            "color_zones_per_layer": ("INT", _opts("color_zones_per_layer", default=4, min=2, max=24, step=1)),
            "color_brightness": ("FLOAT", _opts("color_brightness", default=0.0, min=-1.0, max=1.0, step=0.02)),
            "color_contrast": ("FLOAT", _opts("color_contrast", default=1.0, min=0.0, max=3.0, step=0.05)),
            "color_saturation": ("FLOAT", _opts("color_saturation", default=1.0, min=0.0, max=3.0, step=0.05)),
            "black_white_threshold": ("FLOAT", _opts("black_white_threshold", default=0.5, min=0.0, max=1.0, step=0.01)),
            "enable_shadow": ("BOOLEAN", _opts("enable_shadow", default=True)),
            "shadow_distance": ("FLOAT", _opts("shadow_distance", default=8.0, min=0.0, max=96.0, step=1.0)),
            "shadow_angle": ("FLOAT", _opts("shadow_angle", default=135.0, min=0.0, max=360.0, step=1.0)),
            "shadow_blur": ("FLOAT", _opts("shadow_blur", default=6.0, min=0.0, max=48.0, step=1.0)),
            "shadow_opacity": ("FLOAT", _opts("shadow_opacity", default=0.35, min=0.0, max=1.0, step=0.05)),
            "shadow_color_r": ("FLOAT", _opts("shadow_color_r", default=0.0, min=0.0, max=1.0, step=0.01)),
            "shadow_color_g": ("FLOAT", _opts("shadow_color_g", default=0.0, min=0.0, max=1.0, step=0.01)),
            "shadow_color_b": ("FLOAT", _opts("shadow_color_b", default=0.0, min=0.0, max=1.0, step=0.01)),
            "depth_scaled_shadow": ("BOOLEAN", _opts("depth_scaled_shadow", default=True)),
            "enable_outline": ("BOOLEAN", _opts("enable_outline", default=False)),
            "outline_width": ("INT", _opts("outline_width", default=2, min=1, max=16, step=1)),
            "outline_opacity": ("FLOAT", _opts("outline_opacity", default=0.5, min=0.0, max=1.0, step=0.05)),
            "outline_color_r": ("FLOAT", _opts("outline_color_r", default=0.0, min=0.0, max=1.0, step=0.01)),
            "outline_color_g": ("FLOAT", _opts("outline_color_g", default=0.0, min=0.0, max=1.0, step=0.01)),
            "outline_color_b": ("FLOAT", _opts("outline_color_b", default=0.0, min=0.0, max=1.0, step=0.01)),
            "enable_layer_offset": ("BOOLEAN", _opts("enable_layer_offset", default=False)),
            "offset_x": ("FLOAT", _opts("offset_x", default=0.0, min=-128.0, max=128.0, step=1.0)),
            "offset_y": ("FLOAT", _opts("offset_y", default=0.0, min=-128.0, max=128.0, step=1.0)),
        },
    }
