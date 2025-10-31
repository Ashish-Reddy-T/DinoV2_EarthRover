from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import imageio
import numpy as np
from PIL import Image, ImageDraw


def _make_target(size: int = 256) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(25, 25, 25))
    draw = ImageDraw.Draw(img)
    draw.rectangle([size * 0.1, size * 0.1, size * 0.9, size * 0.9], outline=(255, 60, 60), width=8)
    draw.ellipse([size * 0.35, size * 0.3, size * 0.75, size * 0.7], fill=(60, 255, 120))
    draw.line([0, size // 2, size, size // 3], fill=(60, 120, 255), width=10)
    return img


def _place_target(rng: np.random.Generator, frame: Image.Image, target: Image.Image, margin: int) -> None:
    fw, fh = frame.size
    tw, th = target.size
    x = int(rng.integers(margin, max(margin + 1, fw - tw - margin + 1)))
    y = int(rng.integers(margin, max(margin + 1, fh - th - margin + 1)))
    frame.paste(target, (x, y))


def build_demo(
    out_dir: Path,
    width: int,
    height: int,
    frames: int,
    fps: int,
    insert_every: int,
    margin: int,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = _make_target()
    target_path = out_dir / "demo_target.png"
    target.save(target_path)

    rng = np.random.default_rng(seed)
    inserted_frames: list[int] = []
    video_path = out_dir / "demo_video.mp4"
    target_small = target.resize((min(width, height) // 3, min(width, height) // 3))

    with imageio.get_writer(video_path, fps=fps) as writer:
        for idx in range(frames):
            background = (rng.random((height, width, 3)) * 255).astype(np.uint8)
            frame = Image.fromarray(background)
            if insert_every and idx % insert_every == 0:
                _place_target(rng, frame, target_small, margin)
                inserted_frames.append(idx)
            writer.append_data(np.array(frame))

    manifest = {
        "video_path": str(video_path.resolve()),
        "target_path": str(target_path.resolve()),
        "inserted_frames": inserted_frames,
        "fps": fps,
    }
    manifest_path = out_dir / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic demo video + target pair.")
    parser.add_argument("--out", type=Path, default=Path("assets"), help="Directory where demo files will be written.")
    parser.add_argument("--width", type=int, default=640, help="Video width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Video height in pixels.")
    parser.add_argument("--frames", type=int, default=24, help="Number of frames to synthesise.")
    parser.add_argument("--fps", type=int, default=6, help="Frame rate for the output video.")
    parser.add_argument("--insert-every", type=int, default=6, help="Insert the target every N frames (0 disables).")
    parser.add_argument("--margin", type=int, default=40, help="Minimum margin from frame edges when inserting target.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_demo(
        out_dir=args.out,
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        insert_every=args.insert_every,
        margin=args.margin,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
