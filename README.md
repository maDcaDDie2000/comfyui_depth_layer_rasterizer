# ComfyUI Depth Layer Rasterizer

ComfyUI custom nodes that quantize depth into discrete color-preserved layers for paper-cut, parallax, and diorama effects.

## Nodes

### Depth Rasterize Layers

Main node: takes RGB images (or video batches), depth from a model or precomputed map, and outputs per-layer color slices, masks, posterized depth, optional composite, and debug preview.

**Category:** `depth/layers`

### Null Depth Model

Placeholder `DEPTH_MODEL` output for workflows that supply depth via the `depth_map` input (e.g. from Depth Anything or other depth nodes).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-org/comfyui_depth_layer_rasterizer.git
cd comfyui_depth_layer_rasterizer
pip install -r requirements.txt
```

Restart ComfyUI.

## Workflow

1. Load image or video frames (`IMAGE` batch).
2. Connect a depth map (`IMAGE`, grayscale) **or** a `DEPTH_MODEL` callable.
   - If using only `depth_map`, connect **Null Depth Model** to the `depth_model` socket (optional socket can be left disconnected if your ComfyUI build allows it).
3. Set `rasterization_levels` (2–64, default 16).
4. Choose `output_mode`:
   - `all_layers` — batch of `frames × layers` images and masks
   - `single_layer` — one selected slice
   - `composite` — stacked result with optional outline/shadow/offset
   - `masks_only` — mask for selected layer
   - `debug` — false-color layer visualization

## DEPTH_MODEL integration

Any object with a `.predict(images)` method (or callable) that returns `[B, H, W]` or `[B, H, W, C]` depth is supported. Precomputed depth as `IMAGE` is the most common path in ComfyUI.

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

## License

MIT
