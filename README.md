# ComfyUI Depth Layer Rasterizer

ComfyUI custom nodes that quantize depth into discrete layers for paper-cut, parallax, and diorama effects. Each depth slice can keep original colors, fill with one flat average, or posterize into **color zones** (flat patches within the slice).

## Nodes

| Node | Category | Purpose |
|------|----------|---------|
| **Depth Rasterize Layers** | `depth/layers` | Main processing node |
| **Null Depth Model** | `depth/layers` | Placeholder when using `depth_map` only |

Hover any widget or socket on the node for a short tooltip. Full reference below.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-org/comfyui_depth_layer_rasterizer.git
cd comfyui_depth_layer_rasterizer
pip install -r requirements.txt
```

Restart ComfyUI.

## Quick workflow

1. Connect **image** (RGB batch).
2. Connect **depth_map** (grayscale IMAGE from Depth Anything, etc.) or a **depth_model**.
3. Set **rasterization_levels** (default 16).
4. Pick **layer_color_mode** for the look you want (see below).
5. Choose **output_mode** (`all_layers`, `composite`, etc.).

If you only use `depth_map`, connect **Null Depth Model** to `depth_model` when your ComfyUI build requires that socket.

---

## Layer color modes (depth + color rasterization)

Depth splits the image by distance. **Layer color mode** controls how pixels inside each depth slice are filled:

| Mode | Effect | Best for |
|------|--------|----------|
| **original** | Keeps per-pixel RGB inside each depth mask | Realistic cutouts, parallax |
| **flat_average** | Entire depth slice filled with one average color | Minimal paper-cut / silhouette layers |
| **color_zones** | Within each depth slice, group similar colors and flatten each group to its average (posterization / cel-shading) | Stylized diorama, comic/paper craft |

### Color zones (intra-layer posterization)

When `layer_color_mode = color_zones`, each depth layer is subdivided by color—not by depth again. Example with 4 depth layers and 4 color zones per layer:

```text
Depth layer 0 (near)
  ├── color zone 0 → flat fill (avg of dark greens in that slice)
  ├── color zone 1 → flat fill (avg of skin tones)
  └── …

Depth layer 1
  ├── color zone 0 → flat fill
  └── …
```

Think of it as **two-stage rasterization**: first by depth, then by color within each slice. Similar hues merge into flat patches (sky, foliage, clothing) while depth separation is preserved between layers.

**color_zones_per_layer** — how many color patches per depth slice (2–32, default 4). More zones = finer color detail; fewer = bolder poster look.

**color_zone_space** — how pixels are grouped:

- **luminance_chroma** (default) — groups by brightness + saturation. Usually looks more natural (sky vs trees vs shadows).
- **rgb_cube** — uniform 3D grid in RGB space. More predictable but can merge unlike hues.

---

## Parameter reference

### Inputs (sockets)

| Input | Type | Description |
|-------|------|-------------|
| **image** | IMAGE | RGB input image or video frame batch. |
| **depth_model** | DEPTH_MODEL | Optional estimator with `.predict()`. Ignored if `depth_map` is connected. |
| **depth_map** | IMAGE | Precomputed depth (grayscale). Overrides `depth_model`. |
| **region_mask** | MASK | Limit rasterization to masked areas. |
| **background** | IMAGE | Bottom plate for `composite` output mode. |

### Core

| Parameter | Default | Description |
|-----------|---------|-------------|
| **rasterization_levels** | 16 | Number of depth slices (2–64). |
| **output_mode** | all_layers | `all_layers`, `single_layer`, `composite`, `masks_only`, `debug`. |
| **selected_layer** | 0 | Layer index for `single_layer` / `masks_only` (0 = nearest). |

### Depth processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| **invert_depth** | false | Swap near/far after normalization. |
| **auto_normalize_depth** | true | Stretch depth to 0–1 automatically. |
| **depth_min** / **depth_max** | 0 / 1 | Manual range when `normalization_mode = manual`. |
| **normalization_mode** | per_frame | `per_frame`, `whole_sequence`, `manual`. |
| **depth_gamma** | 1.0 | Bias depth toward near (>1) or far (<1) layers. |
| **depth_blur** | 1.0 | Blur raw depth before normalization (large-scale noise). |
| **temporal_smoothing** | 0.0 | Video: blend depth with previous frame (reduces flicker). |
| **layer_smoothing** | 0.0 | Smooth layer borders after normalization — blurs fine depth texture before slicing and merges tiny layer specks (hair, foliage, noisy depth). |

### Layer color

| Parameter | Default | Description |
|-----------|---------|-------------|
| **layer_color_mode** | original | `original`, `flat_average`, `color_zones`. |
| **color_zones_per_layer** | 4 | Color patches per depth slice (color_zones mode). |
| **color_zone_space** | luminance_chroma | `luminance_chroma` or `rgb_cube`. |

### Mask controls

| Parameter | Default | Description |
|-----------|---------|-------------|
| **mask_feather** | 1.0 | Soft edge width between depth layers. |
| **soft_masks** | true | Soft depth-bin weights vs hard pixels. |
| **mask_expand** | 0 | Dilate layer masks (pixels). |
| **mask_erode** | 0 | Erode layer masks (pixels). |
| **remove_small_islands** | true | Remove tiny speck masks. |
| **island_min_size** | 32 | Minimum connected area to keep. |

### Composite / effects

| Parameter | Default | Description |
|-----------|---------|-------------|
| **layer_order** | near_to_far | Stack order for composite (near on top). |
| **enable_outline** | false | Draw layer outlines. |
| **outline_width** | 2 | Outline thickness (px). |
| **outline_opacity** | 0.75 | Outline strength. |
| **outline_mode** | outer | `outer`, `inner`, `centered`, `depth_boundary`. |
| **outline_color_r/g/b** | 0 | Outline RGB (0–1). |
| **enable_shadow** | false | Drop shadow in composite mode. |
| **shadow_distance** | 8.0 | Shadow offset (px). |
| **shadow_angle** | 135.0 | Shadow direction (degrees). |
| **shadow_blur** | 6.0 | Shadow blur radius. |
| **shadow_opacity** | 0.35 | Shadow strength. |
| **shadow_color_r/g/b** | 0 | Shadow RGB (0–1). |
| **depth_scaled_shadow** | true | Far layers cast longer shadows. |
| **enable_layer_offset** | false | Parallax shift per layer (composite). |
| **offset_x** / **offset_y** | 0 | Parallax strength (px). |
| **offset_mode** | depth_scaled | `uniform`, `depth_scaled`, `manual`. |
| **manual_offset_x/y** | 0 | Fixed offset when mode is manual. |

### Outputs

| Output | Description |
|--------|-------------|
| **layered_images** | Batch of layer images (`frames × layers` in `all_layers` mode). |
| **layer_masks** | Matching mask batch. |
| **depth_quantized** | Posterized grayscale depth per frame. |
| **composited_image** | Stacked composite when `output_mode = composite` (otherwise empty/black — switch output_mode to composite to use this output). |
| **debug_preview** | False-color depth layer visualization. |

**Smooth layer borders (hair, grass, noisy depth)**

```text
layer_smoothing = 4
mask_feather = 1.5
soft_masks = true
depth_blur = 1.0
```

---

## Example presets

**16-layer realistic cutout**

```text
rasterization_levels = 16
layer_color_mode = original
output_mode = all_layers
```

**Minimal flat paper layers**

```text
rasterization_levels = 8
layer_color_mode = flat_average
output_mode = composite
enable_outline = true
enable_shadow = true
```

**Stylized poster / cel-shaded diorama**

```text
rasterization_levels = 12
layer_color_mode = color_zones
color_zones_per_layer = 6
output_mode = composite
enable_outline = true
outline_width = 2
```

**Debug depth separation**

```text
output_mode = debug
rasterization_levels = 16
```

---

## DEPTH_MODEL integration

Any callable or object with `.predict(images)` returning `[B,H,W]` or `[B,H,W,C]` depth works. In practice, connecting a **depth_map** IMAGE from Depth Anything / Depth Anything V2 is the usual path.

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

## License

MIT
