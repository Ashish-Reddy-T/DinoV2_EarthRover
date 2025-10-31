from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PreprocessConfig:
    resize_size: int = 518
    crop_size: int = 518
    center_crop: bool = True
    interpolation: InterpolationMode = InterpolationMode.BICUBIC


ImageInput = Union[Image.Image, np.ndarray, str, Path]


class DinoFeatureExtractor:
    """Wraps a DINOv2 model and handles feature extraction."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: str | None = None,
        preprocess: PreprocessConfig | None = None,
        cache_model: bool = True,
        half: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess_config = preprocess or PreprocessConfig()
        self.cache_model = cache_model
        self.half = half if half is not None else (self.device.startswith("cuda"))
        self._model: torch.nn.Module | None = None
        self._transform = self._build_transform(self.preprocess_config)

    def _build_transform(self, config: PreprocessConfig) -> transforms.Compose:
        ops = [
            transforms.Resize(config.resize_size, interpolation=config.interpolation),
        ]
        if config.center_crop:
            ops.append(transforms.CenterCrop(config.crop_size))
        else:
            ops.append(transforms.Resize((config.crop_size, config.crop_size), interpolation=config.interpolation))
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return transforms.Compose(ops)

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        model = torch.hub.load("facebookresearch/dinov2", self.model_name)  # type: ignore[attr-defined]
        model.eval()
        model.to(self.device)
        if self.half and self.device.startswith("cuda"):
            model.half()
        if self.cache_model:
            self._model = model
        return model

    @staticmethod
    def _ensure_image(image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(str(image)).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return Image.fromarray(image)
        raise TypeError(f"Unsupported image type: {type(image)}")

    def preprocess(self, image: ImageInput) -> torch.Tensor:
        pil_img = self._ensure_image(image)
        tensor = self._transform(pil_img).unsqueeze(0)
        if self.half and self.device.startswith("cuda"):
            tensor = tensor.half()
        return tensor.to(self.device)

    @property
    def model(self) -> torch.nn.Module:
        return self._load_model()

    def extract(self, image: ImageInput, normalize: bool = True) -> torch.Tensor:
        model = self._load_model()
        with torch.inference_mode():
            feats = model(self.preprocess(image))
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats.squeeze(0)

    def extract_batch(
        self,
        images: Sequence[ImageInput],
        normalize: bool = True,
        batch_size: int = 16,
    ) -> torch.Tensor:
        if not images:
            raise ValueError("No images provided for batch extraction.")
        model = self._load_model()
        all_feats: List[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start : start + batch_size]
            stacked = torch.cat([self.preprocess(img) for img in batch_imgs], dim=0)
            with torch.inference_mode():
                feats = model(stacked)
            if normalize:
                feats = F.normalize(feats, dim=-1)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)


@lru_cache(maxsize=1)
def get_default_extractor() -> DinoFeatureExtractor:
    return DinoFeatureExtractor()
