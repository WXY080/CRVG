"""Preprocessing for the original OpenGVLab InternVL3 checkpoints.

Dynamic tiling follows OpenGVLab's MIT-licensed model-card implementation:
https://huggingface.co/OpenGVLab/InternVL3-9B
Copyright (c) 2024 OpenGVLab.
"""
import numpy as np
from PIL import Image
import torch


def image_tiles(image, image_size=448, max_tiles=12, use_thumbnail=True):
    """Keep aspect ratio with a bounded tile grid and an optional thumbnail."""
    if image_size < 1 or max_tiles < 1:
        raise ValueError("image_size and max_tiles must be positive")
    image = image.convert("RGB")
    width, height = image.size
    ratios = sorted({(w, h) for w in range(1, max_tiles + 1)
                     for h in range(1, max_tiles + 1) if w * h <= max_tiles},
                    key=lambda ratio: ratio[0] * ratio[1])
    best, distance = (1, 1), float("inf")
    for columns, rows in ratios:
        difference = abs(width / height - columns / rows)
        if difference < distance or (difference == distance and
                width * height > .5 * image_size ** 2 * columns * rows):
            best, distance = (columns, rows), difference
    columns, rows = best
    resized = image.resize((columns * image_size, rows * image_size), Image.Resampling.BICUBIC)
    tiles = [resized.crop((x * image_size, y * image_size,
                          (x + 1) * image_size, (y + 1) * image_size))
             for y in range(rows) for x in range(columns)]
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    return tiles


def pixel_batch(images, config, max_tiles=12):
    """Return ImageNet-normalized tiles and per-image tile counts."""
    image_size = config.force_image_size or config.vision_config.image_size
    dynamic = getattr(config, "dynamic_image_size", True)
    thumbnail = getattr(config, "use_thumbnail", True)
    mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(3, 1, 1)
    pixels, counts = [], []
    for image in images:
        tiles = image_tiles(image, image_size, max_tiles if dynamic else 1, thumbnail)
        counts.append(len(tiles))
        for tile in tiles:
            tensor = torch.from_numpy(np.array(tile)).permute(2, 0, 1).float() / 255.
            pixels.append((tensor - mean) / std)
    return torch.stack(pixels), counts
