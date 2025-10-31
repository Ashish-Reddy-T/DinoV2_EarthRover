from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .extractor import DinoFeatureExtractor
from .video_search import VideoSearcher

console = Console()
app = typer.Typer(add_completion=False, help="Search for a target image inside a video using DINOv2.")


@app.command()
def search(
    video: Path = typer.Argument(..., exists=True, readable=True, help="Input video file (mp4, mov, etc.)."),
    target: Path = typer.Argument(..., exists=True, readable=True, help="Target image to locate."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Directory to store annotated frames and results."),
    frame_stride: int = typer.Option(1, "--frame-stride", help="Process every Nth frame."),
    window_size: int = typer.Option(350, "--window-size", help="Base sliding window size in pixels."),
    stride_ratio: float = typer.Option(0.5, "--stride-ratio", help="Stride as a fraction of window size."),
    min_window: int = typer.Option(196, "--min-window", help="Smallest window that will be considered."),
    frame_threshold: float = typer.Option(0.3, "--frame-threshold", help="Similarity threshold for full-frame matches."),
    patch_threshold: float = typer.Option(0.5, "--patch-threshold", help="Similarity threshold for window matches."),
    topk: int = typer.Option(5, "--topk", help="Number of detections saved per frame."),
    model_name: str = typer.Option("dinov2_vits14", "--model-name", help="Model variant to load."),
    use_half: bool = typer.Option(False, "--half", help="Use half precision when running on CUDA."),
):
    """Run a full search over the provided video."""
    console.log("Loading DINOv2 model…")
    extractor = DinoFeatureExtractor(model_name=model_name, half=use_half)
    searcher = VideoSearcher(
        extractor=extractor,
        frame_stride=frame_stride,
        window_size=window_size,
        stride_ratio=stride_ratio,
        min_window=min_window,
    )

    output_dir = output or (video.parent / f"{video.stem}_search")
    console.log(f"Searching video {video} with target {target}…")
    results = searcher.search_video(
        video_path=video,
        target_image=target,
        output_dir=output_dir,
        frame_threshold=frame_threshold,
        patch_threshold=patch_threshold,
        topk=topk,
    )

    if not results:
        console.print("[yellow]No matching frames detected above the provided thresholds.[/yellow]")
        return

    table = Table(title="Detection Summary")
    table.add_column("Frame")
    table.add_column("Timestamp (s)")
    table.add_column("Frame Score")
    table.add_column("Matches")

    for result in results:
        boxes = ", ".join(f"{det.score:.2f}" for det in result.detections) or "-"
        table.add_row(str(result.frame_index), f"{result.timestamp:.2f}", f"{result.frame_score:.2f}", boxes)

    console.print(table)
    console.print(f"[green]Results written to[/green] {output_dir}")


@app.command()
def warmup(model_name: str = typer.Option("dinov2_vits14", "--model-name", help="Model variant to cache.")):
    """Load the model once to cache weights locally."""
    console.log(f"Caching model {model_name}…")
    extractor = DinoFeatureExtractor(model_name=model_name)
    _ = extractor.model


if __name__ == "__main__":
    app()
