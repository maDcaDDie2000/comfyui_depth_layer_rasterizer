"""Widget tooltips and INPUT_TYPES helpers for ComfyUI nodes."""

from __future__ import annotations

TOOLTIPS: dict[str, str] = {
    "image": "RGB input image or video frame batch. Each frame is split into depth layers independently.",
    "depth_model": "Optional depth estimator (callable with .predict). Ignored when depth_map is connected.",
    "depth_map": "Precomputed depth as grayscale IMAGE. Overrides depth_model when connected.",
    "region_mask": "Optional mask limiting where rasterization runs. Outside areas stay empty on all layers.",
    "background": "Optional background IMAGE used as the bottom plate when output_mode is composite.",
    "rasterization_levels": "How many depth slices to create (2–64). Higher values = finer depth separation.",
    "output_mode": "What to return: all layer images, one layer, stacked composite, masks only, or debug false-color.",
    "selected_layer": "Layer index to output when output_mode is single_layer or masks_only (0 = nearest).",
    "invert_depth": "Swap near/far interpretation after normalization (0=near, 1=far by default).",
    "auto_normalize_depth": "Automatically stretch each frame's depth range to 0–1 before quantization.",
    "depth_min": "Manual minimum depth for normalization. Used when normalization_mode is manual.",
    "depth_max": "Manual maximum depth for normalization. Used when normalization_mode is manual.",
    "normalization_mode": "per_frame: normalize each frame. whole_sequence: one range for the batch. manual: use depth_min/max.",
    "depth_gamma": "Gamma curve on normalized depth. >1 pushes detail to near layers; <1 pushes detail to far layers.",
    "depth_blur": "Gaussian blur radius on depth before quantization. Reduces noisy layer boundaries.",
    "temporal_smoothing": "Blend each frame's depth with the previous (0=off, 0.9=strong). Reduces video flicker.",
    "layer_smoothing": "Smooth layer borders after depth normalization: blurs fine depth texture before slicing, then merges tiny layer specks. Use for hair, foliage, or noisy depth. 0=off.",
    "layer_color_mode": "How layer pixels are filled: original colors, one flat average per layer, or color zones within each layer.",
    "color_zones_per_layer": "When layer_color_mode is color_zones: how many flat color patches to create inside each depth slice.",
    "color_zone_space": "How pixels are grouped into color zones: luminance+chroma (natural) or RGB cube (uniform grid).",
    "mask_feather": "Soft falloff width at depth layer edges. Higher = smoother transitions between depth slices.",
    "soft_masks": "Use soft depth-bin weights instead of hard pixel assignment. Smoother but less crisp.",
    "mask_expand": "Grow each layer mask outward by N pixels (dilation). Helps close small gaps.",
    "mask_erode": "Shrink each layer mask inward by N pixels (erosion). Removes fringe pixels.",
    "remove_small_islands": "Delete tiny disconnected mask specks using connected-component filtering.",
    "island_min_size": "Minimum connected pixel area to keep when remove_small_islands is enabled.",
    "layer_order": "Composite stacking order: near_to_far draws closest layers on top (typical diorama look).",
    "enable_outline": "Draw an outline around each layer before output or compositing.",
    "outline_width": "Outline thickness in pixels.",
    "outline_opacity": "Outline blend strength (0=invisible, 1=solid).",
    "outline_mode": "outer: outside edge. inner: inside edge. centered: both sides. depth_boundary: only where depth layers meet.",
    "outline_color_r": "Outline red channel (0–1).",
    "outline_color_g": "Outline green channel (0–1).",
    "outline_color_b": "Outline blue channel (0–1).",
    "enable_shadow": "Drop a blurred offset shadow behind each layer in composite mode.",
    "shadow_distance": "Shadow offset distance in pixels before angle is applied.",
    "shadow_angle": "Shadow direction in degrees (0=right, 90=down, 135=bottom-left, common default).",
    "shadow_blur": "Gaussian blur radius applied to the shadow mask.",
    "shadow_opacity": "Shadow strength (0–1).",
    "shadow_color_r": "Shadow red channel (0–1).",
    "shadow_color_g": "Shadow green channel (0–1).",
    "shadow_color_b": "Shadow blue channel (0–1).",
    "depth_scaled_shadow": "Scale shadow offset by layer depth so far layers cast longer shadows.",
    "enable_layer_offset": "Shift each layer horizontally/vertically for a parallax cutout effect (composite mode).",
    "offset_x": "Horizontal parallax strength in pixels (sign = direction).",
    "offset_y": "Vertical parallax strength in pixels.",
    "offset_mode": "uniform: same offset all layers. depth_scaled: offset grows with depth. manual: use manual_offset_x/y.",
    "manual_offset_x": "Fixed horizontal offset when offset_mode is manual.",
    "manual_offset_y": "Fixed vertical offset when offset_mode is manual.",
}

OUTPUT_TOOLTIPS = (
    "Layer images: one batch entry per frame×layer (or single layer / debug image depending on output_mode).",
    "Layer masks: alpha masks matching layered_images batch layout.",
    "Posterized grayscale depth map showing discrete depth bands per frame.",
    "Final stacked image with shadows/outlines/offsets when output_mode is composite; otherwise empty or passthrough.",
    "False-color debug view of depth layer IDs (same layout as input frames).",
)

NODE_DESCRIPTION = (
    "Split an image into depth slices, optionally flatten each slice to average or posterized colors, "
    "then export layers or a paper-cut composite with outline, shadow, and parallax."
)


def _opts(key: str, **kwargs) -> dict:
    return {**kwargs, "tooltip": TOOLTIPS[key]}


def _combo(options: list[str], key: str) -> tuple:
    return (options, _opts(key))


def build_depth_rasterize_input_types() -> dict:
    """Build INPUT_TYPES dict with tooltips on every socket and widget."""
    return {
        "required": {
            "image": ("IMAGE", _opts("image")),
            "rasterization_levels": ("INT", _opts("rasterization_levels", default=16, min=2, max=64, step=1)),
            "output_mode": _combo(
                ["all_layers", "single_layer", "composite", "masks_only", "debug"],
                "output_mode",
            ),
        },
        "optional": {
            "depth_model": ("DEPTH_MODEL", _opts("depth_model")),
            "depth_map": ("IMAGE", _opts("depth_map")),
            "region_mask": ("MASK", _opts("region_mask")),
            "background": ("IMAGE", _opts("background")),
            "selected_layer": ("INT", _opts("selected_layer", default=0, min=0, max=63, step=1)),
            "invert_depth": ("BOOLEAN", _opts("invert_depth", default=False)),
            "auto_normalize_depth": ("BOOLEAN", _opts("auto_normalize_depth", default=True)),
            "depth_min": ("FLOAT", _opts("depth_min", default=0.0, min=-1e6, max=1e6, step=0.001)),
            "depth_max": ("FLOAT", _opts("depth_max", default=1.0, min=-1e6, max=1e6, step=0.001)),
            "normalization_mode": _combo(["per_frame", "whole_sequence", "manual"], "normalization_mode"),
            "depth_gamma": ("FLOAT", _opts("depth_gamma", default=1.0, min=0.1, max=4.0, step=0.05)),
            "depth_blur": ("FLOAT", _opts("depth_blur", default=1.0, min=0.0, max=32.0, step=0.5)),
            "temporal_smoothing": ("FLOAT", _opts("temporal_smoothing", default=0.0, min=0.0, max=0.99, step=0.01)),
            "layer_smoothing": ("FLOAT", _opts("layer_smoothing", default=0.0, min=0.0, max=32.0, step=0.5)),
            "layer_color_mode": _combo(["original", "flat_average", "color_zones"], "layer_color_mode"),
            "color_zones_per_layer": ("INT", _opts("color_zones_per_layer", default=4, min=2, max=32, step=1)),
            "color_zone_space": _combo(["luminance_chroma", "rgb_cube"], "color_zone_space"),
            "mask_feather": ("FLOAT", _opts("mask_feather", default=1.0, min=0.0, max=32.0, step=0.5)),
            "soft_masks": ("BOOLEAN", _opts("soft_masks", default=True)),
            "mask_expand": ("INT", _opts("mask_expand", default=0, min=0, max=64, step=1)),
            "mask_erode": ("INT", _opts("mask_erode", default=0, min=0, max=64, step=1)),
            "remove_small_islands": ("BOOLEAN", _opts("remove_small_islands", default=True)),
            "island_min_size": ("INT", _opts("island_min_size", default=32, min=1, max=4096, step=1)),
            "layer_order": _combo(["near_to_far", "far_to_near"], "layer_order"),
            "enable_outline": ("BOOLEAN", _opts("enable_outline", default=False)),
            "outline_width": ("INT", _opts("outline_width", default=2, min=1, max=32, step=1)),
            "outline_opacity": ("FLOAT", _opts("outline_opacity", default=0.75, min=0.0, max=1.0, step=0.05)),
            "outline_mode": _combo(["outer", "inner", "centered", "depth_boundary"], "outline_mode"),
            "outline_color_r": ("FLOAT", _opts("outline_color_r", default=0.0, min=0.0, max=1.0, step=0.01)),
            "outline_color_g": ("FLOAT", _opts("outline_color_g", default=0.0, min=0.0, max=1.0, step=0.01)),
            "outline_color_b": ("FLOAT", _opts("outline_color_b", default=0.0, min=0.0, max=1.0, step=0.01)),
            "enable_shadow": ("BOOLEAN", _opts("enable_shadow", default=False)),
            "shadow_distance": ("FLOAT", _opts("shadow_distance", default=8.0, min=0.0, max=128.0, step=1.0)),
            "shadow_angle": ("FLOAT", _opts("shadow_angle", default=135.0, min=0.0, max=360.0, step=1.0)),
            "shadow_blur": ("FLOAT", _opts("shadow_blur", default=6.0, min=0.0, max=64.0, step=1.0)),
            "shadow_opacity": ("FLOAT", _opts("shadow_opacity", default=0.35, min=0.0, max=1.0, step=0.05)),
            "shadow_color_r": ("FLOAT", _opts("shadow_color_r", default=0.0, min=0.0, max=1.0, step=0.01)),
            "shadow_color_g": ("FLOAT", _opts("shadow_color_g", default=0.0, min=0.0, max=1.0, step=0.01)),
            "shadow_color_b": ("FLOAT", _opts("shadow_color_b", default=0.0, min=0.0, max=1.0, step=0.01)),
            "depth_scaled_shadow": ("BOOLEAN", _opts("depth_scaled_shadow", default=True)),
            "enable_layer_offset": ("BOOLEAN", _opts("enable_layer_offset", default=False)),
            "offset_x": ("FLOAT", _opts("offset_x", default=0.0, min=-256.0, max=256.0, step=1.0)),
            "offset_y": ("FLOAT", _opts("offset_y", default=0.0, min=-256.0, max=256.0, step=1.0)),
            "offset_mode": _combo(["uniform", "depth_scaled", "manual"], "offset_mode"),
            "manual_offset_x": ("FLOAT", _opts("manual_offset_x", default=0.0, min=-256.0, max=256.0, step=1.0)),
            "manual_offset_y": ("FLOAT", _opts("manual_offset_y", default=0.0, min=-256.0, max=256.0, step=1.0)),
        },
    }
