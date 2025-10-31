from __future__ import annotations

import json
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw

from dinov2_pipeline.extractor import DinoFeatureExtractor
from dinov2_pipeline.video_search import VideoSearcher


def _make_target_image(size: int = 224) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([size // 8, size // 8, size - size // 8, size - size // 8], outline=(220, 30, 30), width=8)
    draw.ellipse([size // 3, size // 3, size - size // 3, size - size // 3], fill=(30, 220, 30))
    draw.line([0, size // 2, size, size // 2], fill=(30, 30, 220), width=6)
    return img


def _make_video(path: Path, target: Image.Image, frames: int = 8, insert_frames: tuple[int, ...] = (3, 6)) -> None:
    rng = np.random.default_rng(42)
    target_small = target.resize((160, 160))
    with imageio.get_writer(path, fps=4) as writer:
        for idx in range(frames):
            base = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
            frame = Image.fromarray(base)
            if idx in insert_frames:
                frame.paste(target_small, (240, 160))
            writer.append_data(np.array(frame))


def test_search_pipeline(tmp_path: Path) -> None:
    target = _make_target_image()
    video_path = tmp_path / "demo.mp4"
    _make_video(video_path, target)
    target_path = tmp_path / "target.png"
    target.save(target_path)

    extractor = DinoFeatureExtractor(model_name="dinov2_vits14")
    searcher = VideoSearcher(
        extractor=extractor,
        frame_stride=1,
        window_size=220,
        stride_ratio=0.5,
        min_window=150,
        scales=(0.8, 1.0, 1.2),
        batch_size=16,
    )

    output_dir = tmp_path / "results"
    results = searcher.search_video(
        video_path=video_path,
        target_image=target_path,
        output_dir=output_dir,
        frame_threshold=0.2,
        patch_threshold=0.65,
        topk=3,
    )

    assert results, "No matches returned by the search pipeline."

    matched_frames = {res.frame_index for res in results if res.detections}
    assert matched_frames.issuperset({3, 6}), f"Expected detections on frames 3 and 6, got {matched_frames}"

    json_path = output_dir / "results.json"
    assert json_path.exists(), "results.json was not generated."
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert "results" in data and data["results"], "JSON summary missing detection results."
