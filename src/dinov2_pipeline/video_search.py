from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .extractor import DinoFeatureExtractor


@dataclass
class Detection:
    frame_index: int
    box: Tuple[int, int, int, int]
    score: float
    scale: float

    def as_dict(self) -> dict:
        data = asdict(self)
        data["box"] = {"x1": self.box[0], "y1": self.box[1], "x2": self.box[2], "y2": self.box[3]}
        return data


@dataclass
class SearchResult:
    frame_index: int
    timestamp: float
    frame_score: float
    detections: List[Detection]

    def as_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "frame_score": self.frame_score,
            "detections": [det.as_dict() for det in self.detections],
        }


class VideoSearcher:
    """Searches for an instance of a target image inside a video using DINOv2 features."""

    def __init__(
        self,
        extractor: DinoFeatureExtractor,
        frame_stride: int = 1,
        window_size: int = 350,
        stride_ratio: float = 0.5,
        min_window: int = 196,
        scales: Sequence[float] = (0.8, 1.0, 1.2),
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if not 0 < stride_ratio <= 1:
            raise ValueError("stride_ratio must be in (0, 1].")
        self.extractor = extractor
        self.frame_stride = max(1, frame_stride)
        self.window_size = window_size
        self.stride_ratio = stride_ratio
        self.min_window = min_window
        self.scales = scales
        self.batch_size = batch_size
        self.device = device or extractor.device

    def search_video(
        self,
        video_path: str | Path,
        target_image: str | Path | np.ndarray | Image.Image,
        output_dir: str | Path | None = None,
        frame_threshold: float = 0.3,
        patch_threshold: float = 0.5,
        topk: int = 5,
        annotate: bool = True,
    ) -> List[SearchResult]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_feature = self.extractor.extract(target_image)
        target_feature = target_feature.to(self.extractor.device)

        results: List[SearchResult] = []
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        frame_index = 0
        processed_frames = 0
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if frame_index % self.frame_stride != 0:
                    frame_index += 1
                    continue
                processed_frames += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_feature = self.extractor.extract(frame_pil)
                frame_score = float(torch.dot(frame_feature, target_feature).cpu())

                detections = self._scan_frame(frame_rgb, target_feature, patch_threshold, frame_index)

                if frame_score >= frame_threshold or detections:
                    detections = sorted(detections, key=lambda d: d.score, reverse=True)[:topk]
                    timestamp = frame_index / fps if fps else 0.0
                    results.append(
                        SearchResult(
                            frame_index=frame_index,
                            timestamp=timestamp,
                            frame_score=frame_score,
                            detections=detections,
                        )
                    )
                    if output_path and annotate:
                        annotated = self._draw_detections(frame_rgb, detections, score=frame_score)
                        outfile = output_path / f"frame_{frame_index:06d}.jpg"
                        cv2.imwrite(str(outfile), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                frame_index += 1
        finally:
            cap.release()

        if output_path:
            metadata = {
                "video_path": str(video_path),
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "frame_stride": self.frame_stride,
                "frame_threshold": frame_threshold,
                "patch_threshold": patch_threshold,
                "results": [result.as_dict() for result in results],
            }
            with (output_path / "results.json").open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)

        return results

    def _scan_frame(
        self,
        frame_rgb: np.ndarray,
        target_feature: torch.Tensor,
        threshold: float,
        frame_index: int,
    ) -> List[Detection]:
        h, w, _ = frame_rgb.shape
        windows = self._generate_windows(w, h)
        if not windows:
            return []

        patches: List[Image.Image] = [Image.fromarray(frame_rgb[y1:y2, x1:x2]) for (x1, y1, x2, y2, _) in windows]
        features = self.extractor.extract_batch(patches, batch_size=self.batch_size)
        scores = torch.matmul(features, target_feature.to(features.device))
        scores = scores.detach().cpu().numpy()

        detections: List[Detection] = []
        for idx, score in enumerate(scores):
            if score < threshold:
                continue
            x1, y1, x2, y2, scale = windows[idx]
            detections.append(
                Detection(
                    frame_index=frame_index,
                    box=(x1, y1, x2, y2),
                    score=float(score),
                    scale=scale,
                )
            )
        return detections

    def _generate_windows(self, width: int, height: int) -> List[Tuple[int, int, int, int, float]]:
        boxes: List[Tuple[int, int, int, int, float]] = []
        max_dim = min(width, height)
        base_window = min(self.window_size, max_dim)
        for scale in self.scales:
            win_size = int(base_window * scale)
            if win_size < self.min_window:
                continue
            stride = max(32, int(win_size * self.stride_ratio))
            if stride <= 0:
                stride = win_size
            scale_boxes: List[Tuple[int, int, int, int, float]] = []
            for y in range(0, max(1, height - win_size + 1), stride):
                for x in range(0, max(1, width - win_size + 1), stride):
                    x2 = min(width, x + win_size)
                    y2 = min(height, y + win_size)
                    scale_boxes.append((x, y, x2, y2, scale))
            if not scale_boxes:
                scale_boxes.append((0, 0, width, height, scale))
            else:
                # Explicitly cover right and bottom edges
                if scale_boxes[-1][2] < width:
                    scale_boxes.append((max(0, width - win_size), 0, width, min(height, win_size), scale))
                if scale_boxes[-1][3] < height:
                    scale_boxes.append((0, max(0, height - win_size), min(width, win_size), height, scale))
                # Bottom-right corner
                scale_boxes.append(
                    (max(0, width - win_size), max(0, height - win_size), width, height, scale)
                )
            boxes.extend(scale_boxes)
        # Fallback: include full frame
        if not boxes:
            boxes.append((0, 0, width, height, 1.0))
        return boxes

    @staticmethod
    def _draw_detections(
        frame_rgb: np.ndarray,
        detections: Sequence[Detection],
        score: float,
    ) -> np.ndarray:
        annotated = frame_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.score:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            f"frame score: {score:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated
