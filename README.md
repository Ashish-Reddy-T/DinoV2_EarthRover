# DINOv2 Video Target Search

This project provides an end‑to‑end pipeline that lets you point a DINOv2 model at a video and ask whether a smaller **target image** appears anywhere inside. It handles feature extraction, sliding‑window search, optional frame annotation, and produces a machine‑readable JSON summary.

The code is packaged so you can run it from a CLI (`dinov2-search`) or import the `VideoSearcher` utility in your own notebooks.

---

## 1. Environment Setup

1. **Create the conda environment (Python 3.10 + PyTorch 2.2 CPU/GPU builds):**
   ```bash
   conda env create -f environment.yml
   conda activate dinov2-sim
   ```
   *If you want CUDA acceleration, install the matching `pytorch-cuda` package after activation (see: https://pytorch.org/get-started/locally/).*

2. **Install the project in editable mode (brings in the CLI and pytest):**
   ```bash
   pip install -e ".[dev]"
   ```

3. **(Optional) Warm up model weights once to avoid first-run latency:**
   ```bash
   dinov2-search warmup --model-name dinov2_vits14
   ```

---

## 2. CLI Usage

Run a search with a target image and a video:

```bash
dinov2-search search \
  /path/to/video.mp4 \
  /path/to/target.png \
  --output /path/to/output_dir \
  --frame-stride 2 \
  --window-size 350 \
  --frame-threshold 0.35 \
  --patch-threshold 0.6
```

Example using the bundled demo assets:

```bash
dinov2-search search assets/demo_video.mp4 assets/demo_target.png
```

What you get:
- Annotated frames written to `output_dir` (one image per matching frame).
- `output_dir/results.json` with frame indices, timestamps, frame scores, and patch detections.
- A console table summarising the matches.

**Helpful flags**
- `--frame-stride`: skip frames for faster scans.
- `--window-size` & `--stride-ratio`: control sliding-window granularity (smaller = more precise, slower).
- `--patch-threshold`: higher → fewer detections; lower → more sensitive.
- `--half`: run the model in FP16 when CUDA is available.

You can always invoke the CLI without prior installation by using:
```bash
python -m dinov2_pipeline.cli search ...
```

---

## 3. Library API

```python
from pathlib import Path

from dinov2_pipeline import DinoFeatureExtractor, VideoSearcher

extractor = DinoFeatureExtractor(model_name="dinov2_vits14")
searcher = VideoSearcher(extractor, frame_stride=2)
results = searcher.search_video(
    "video.mp4",
    Path("target.jpg"),
    output_dir="outputs",
    frame_threshold=0.3,
    patch_threshold=0.55,
)
```

Each `SearchResult` contains the processed `frame_index`, `timestamp`, `frame_score`, and a list of `Detection` entries with bounding boxes and similarity scores.

---

## 4. Running Tests

Tests fabricate a synthetic video, download the DINOv2 weights via `torch.hub`, and assert the detector can recover the implanted target:

```bash
conda activate dinov2-sim
pytest -q
```

The first run can take about a minute while weights (~300 MB) download. Subsequent runs hit the local cache (`~/.cache/torch/hub/`).

---

## 5. Demo Data (Optional)

To try the pipeline without your own assets, you can generate a quick synthetic example:

```bash
python scripts/make_demo.py
dinov2-search search assets/demo_video.mp4 assets/demo_target.png
```

See `scripts/make_demo.py` for tunable parameters (frame size, insertion frames, etc.).

---

## 6. Tips for Real Footage

- Pre-trim videos to the region of interest to keep runtimes manageable.
- Start with a looser `--patch-threshold` (≈0.45) and inspect `results.json`; tighten once you confirm matches.
- If the target appears at wildly different scales, widen `--scales` in `VideoSearcher` (e.g. `(0.5, 0.8, 1.0, 1.3)`).
- Lighting changes can affect similarity. Consider histogram-equalising frames beforehand if needed.
- For live webcam support, feed frames from OpenCV capture into `VideoSearcher._scan_frame` in a loop; the existing code is agnostic to the source.

---

## 7. Repository Layout

- `environment.yml` – reproducible conda environment (Python 3.10 + PyTorch 2.2).
- `pyproject.toml` – package metadata and CLI entry point.
- `src/dinov2_pipeline/` – core Python modules.
- `tests/` – pytest suite that fabricates a synthetic video and asserts detections.
- `assets/` – place your videos/targets or generated demo data.
- `scripts/make_demo.py` – helper to fabricate demo assets (run once, keep results).

---

## 8. Next Steps

- Integrate a tracker (e.g. Norfair/ByteTrack) to stabilise detections across frames.
- Export annotated results as an `mp4` via `imageio.get_writer(...)`.
- Build a simple web UI (Gradio/Streamlit) around the CLI for interactive use.
