# Run DinoV2_EarthRover on Google Colab (GPU)

This guide shows how to run the DINOv2 video search pipeline on Google Colab with a GPU for faster processing and live progress/ETA.

## 1) Start a GPU Runtime

- Open https://colab.research.google.com/
- Runtime > Change runtime type > Hardware accelerator: GPU > Save

(Optional) Verify GPU:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
```

## 2) Clone the repo

```bash
!git clone https://github.com/Ashish-Reddy-T/DinoV2_EarthRover.git
%cd DinoV2_EarthRover
```

## 3) Install dependencies

Colab includes recent PyTorch + CUDA by default. Install package in editable mode so CLI is available and code edits re-load instantly.

```bash
# Optional: upgrade pip first
!pip install -U pip

# Install this project (regular + dev deps)
!pip install -e ".[dev]"
```

Notes:
- If you see warnings about xFormers, it's optional. You can install it for speed-ups on GPU:
  ```bash
  !pip install xformers
  ```
  If it fails, you can skip it; the pipeline still works.

## 4) (Optional) Generate the demo assets

The repo ships with a demo generator. You can also use your own files.

```bash
!python scripts/make_demo.py --out assets --frames 24 --fps 6 --insert-every 6 --seed 1234
!ls -lh assets
```
This writes:
- `assets/demo_video.mp4`
- `assets/demo_target.png`
- `assets/demo_manifest.json`

## 5) Warm up the model (downloads/caches weights)

```bash
!dinov2-search warmup --model-name dinov2_vits14
```

If the CLI command is not found (rare in Colab), use the module entry point:
```bash
!python -m dinov2_pipeline.cli warmup dinov2_vits14
```

## 6) Run a search (with live progress + ETA)

Example on the demo assets:
```bash
!dinov2-search search \
  assets/demo_video.mp4 \
  assets/demo_target.png \
  --output /content/results \
  --frame-stride 2 \
  --window-size 350 \
  --frame-threshold 0.35 \
  --patch-threshold 0.6 \
  --half
```
Notes:
- The `--half` flag uses half precision on CUDA to speed up throughput and reduce memory.
- During the run you'll see a progress bar with FPS, elapsed time and ETA.
- Outputs:
  - Annotated frames in `/content/results/`
  - A summary JSON at `/content/results/results.json` (includes elapsed_seconds and processing_fps)

If the CLI is unavailable for any reason, use the module form:
```bash
!python -m dinov2_pipeline.cli search \
  assets/demo_video.mp4 \
  assets/demo_target.png \
  --output /content/results \
  --frame-stride 2 \
  --window-size 350 \
  --frame-threshold 0.35 \
  --patch-threshold 0.6 \
  --half
```

## 7) Use your own video/target (Drive)

Mount Google Drive to access your files:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Then run the search pointing to your files:
```bash
!dinov2-search search \
  /content/drive/MyDrive/path/to/your_video.mp4 \
  /content/drive/MyDrive/path/to/your_target.png \
  --output /content/results \
  --frame-stride 2 \
  --window-size 350 \
  --frame-threshold 0.35 \
  --patch-threshold 0.6 \
  --half
```

## Performance tips on Colab

- Enable GPU runtime (see step 1).
- Use `--half` to leverage half precision on CUDA.
- Increase `--frame-stride` (e.g., 3–5) to process fewer frames.
- Increase `--window-size` to reduce the number of sliding windows per frame.
- For very high-res (4K) videos, consider downscaling the video before search to speed up processing.

## Troubleshooting

- CLI not found: use the module variant `python -m dinov2_pipeline.cli ...`.
- Typer/Click mismatch: this repo pins compatible versions; reinstall with `!pip install -e ".[dev]"`.
- xFormers install fails: it’s optional; skip it.
- OpenCV/video codec errors: ensure the video path exists; Colab ships with ffmpeg and OpenCV.

## Example: quick sanity check

```bash
# Warmup
!dinov2-search warmup --model-name dinov2_vits14

# Run demo
!dinov2-search search assets/demo_video.mp4 assets/demo_target.png --output /content/results --frame-stride 2 --half

# Inspect summary
!cat /content/results/results.json
```
