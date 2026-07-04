# ComfyUI Depth Layer Rasterizer

A ComfyUI custom node for turning an image or video frame batch plus a depth map into a layered paper-cut composite. The node slices the depth map into paper layers, fills those layers with stylized color, and can render stacked shadows, outlines, and small parallax offsets without LoRAs or external styling tools.

## Node

| Node | Category | Purpose |
|------|----------|---------|
| **Depth Rasterize Layers** | `depth/layers` | Build layer exports, masks, debug previews, or a final paper-cut composite from image + depth. |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-org/comfyui_depth_layer_rasterizer.git
cd comfyui_depth_layer_rasterizer
pip install -r requirements.txt
```

Restart ComfyUI.

## Quick Workflow

1. Connect `image` from your source image or video.
2. Connect `depth_map` from a depth node such as Depth Anything or Depth Anything V2.
3. Start with `output_mode = composite` and `rasterization_levels = 12`.
4. Use `layer_color_mode` to choose the paper style.
5. Tune shadow distance, blur, and opacity for the stacked-paper depth.

## Color Styles

| Mode | Result | Good for |
|------|--------|----------|
| `original` | Keeps source RGB inside each depth slice. | Photo cutouts, subtle parallax. |
| `flat_average` | Fills each depth layer with one average color. | Minimal paper silhouettes. |
| `color_zones` | Splits each layer into flat average color patches. | Papercraft, cel-shaded diorama looks. |
| `grayscale` | Converts each layer to grayscale. | Monochrome paper studies. |
| `black_white` | Thresholds each layer to pure black or white. | High-contrast stencil and graphic cuts. |

Color adjustments are applied after the selected fill style:

| Control | Default | Description |
|---------|---------|-------------|
| `color_brightness` | 0.0 | Adds or removes brightness. |
| `color_contrast` | 1.0 | Multiplies contrast around mid-gray. |
| `color_saturation` | 1.0 | Multiplies saturation. |
| `black_white_threshold` | 0.5 | Luminance cutoff for `black_white`. |
| `color_zones_per_layer` | 4 | Number of flat patches per depth layer in `color_zones`. |

## Main Controls

| Control | Default | Description |
|---------|---------|-------------|
| `rasterization_levels` | 12 | Number of depth slices. More levels add detail; fewer levels look more graphic. |
| `output_mode` | composite | `composite`, `all_layers`, `single_layer`, `masks_only`, or `debug`. |
| `selected_layer` | 0 | Layer index for `single_layer` and `masks_only`; 0 is nearest. |
| `invert_depth` | false | Swap near/far if the depth map is reversed. |
| `normalization_mode` | per_frame | Use `per_frame` for images, `whole_sequence` for video consistency, or `manual` with min/max. |
| `depth_min` / `depth_max` | 0 / 1 | Manual normalization range when `normalization_mode = manual`. |
| `depth_gamma` | 1.0 | Bias how depth values are distributed into layers. |
| `depth_blur` | 1.0 | Smooth noisy depth before slicing. |
| `temporal_smoothing` | 0.0 | Reduce frame-to-frame depth flicker in video batches. |
| `layer_smoothing` | 1.0 | Merge tiny depth specks into cleaner paper shapes. |
| `mask_feather` | 0.5 | Layer edge softness. Set 0 for crisp hard cuts. |

## Paper Effects

| Control | Default | Description |
|---------|---------|-------------|
| `enable_shadow` | true | Render drop shadows behind layers in `composite` mode. |
| `shadow_distance` | 8.0 | Offset distance in pixels. |
| `shadow_angle` | 135.0 | Shadow direction in degrees. |
| `shadow_blur` | 6.0 | Shadow softness. |
| `shadow_opacity` | 0.35 | Shadow strength. |
| `shadow_color_r/g/b` | 0 / 0 / 0 | Shadow color. |
| `depth_scaled_shadow` | true | Scale shadow offset by depth for a stronger layered-paper look. |
| `enable_outline` | false | Draw an edge around layer shapes. |
| `outline_width` | 2 | Outline thickness in pixels. |
| `outline_opacity` | 0.5 | Outline strength. |
| `outline_color_r/g/b` | 0 / 0 / 0 | Outline color. |
| `enable_layer_offset` | false | Apply a depth-scaled parallax shift to each layer. |
| `offset_x` / `offset_y` | 0 / 0 | Parallax strength in pixels. |

## Outputs

| Output | Description |
|--------|-------------|
| `layered_images` | Layer image batch, or selected/debug images depending on `output_mode`. |
| `layer_masks` | Alpha masks matching `layered_images`. |
| `depth_quantized` | Posterized grayscale view of the depth slices. |
| `composited_image` | Final paper-cut render when `output_mode = composite`. |
| `debug_preview` | False-color layer ID preview. |

## Presets

**Papercraft diorama**

```text
output_mode = composite
rasterization_levels = 12
layer_color_mode = color_zones
color_zones_per_layer = 4
enable_shadow = true
shadow_distance = 8
shadow_blur = 6
```

**Minimal flat paper**

```text
output_mode = composite
rasterization_levels = 8
layer_color_mode = flat_average
mask_feather = 0
enable_shadow = true
```

**Black and white stencil**

```text
output_mode = composite
layer_color_mode = black_white
black_white_threshold = 0.5
enable_outline = false
```

**Export every cut layer**

```text
output_mode = all_layers
rasterization_levels = 16
layer_color_mode = original
```

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

## License

MIT
