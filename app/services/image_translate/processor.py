"""Image pre-processing: resize with white background, composite-image splitting.

Extracted from the original app/services/image_service.py
and app/services/image_split_service.py – zero coupling to the main app.
"""

from pathlib import Path

from PIL import Image, ImageOps

from .config import ModuleConfig
from .schemas import SplitInfo


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------


def resize_with_white_background(
    image_path: Path,
    output_path: Path,
    size: tuple[int, int] = (660, 900),
    quality: int = 95,
) -> Path:
    """Resize *image_path* to fit inside *size*, centering on a white canvas.

    Returns *output_path* so it can be chained.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", size, "white")
        left = (size[0] - image.width) // 2
        top = (size[1] - image.height) // 2
        canvas.paste(image, (left, top))
        canvas.save(output_path, format="JPEG", quality=quality, subsampling=0, optimize=True)

    return output_path


# ---------------------------------------------------------------------------
# composite-image splitting  (horizontal white-gutter detection)
# ---------------------------------------------------------------------------


def _find_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, val in enumerate(flags):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def _merge_close_runs(
    runs: list[tuple[int, int]],
    max_gap: int,
) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _should_keep_split_region(
    crop_bbox: tuple[int, int, int, int],
    source_size: tuple[int, int],
) -> tuple[bool, str]:
    left, top, right, bottom = crop_bbox
    source_width, source_height = source_size
    crop_width = max(0, right - left)
    crop_height = max(0, bottom - top)
    source_area = source_width * source_height
    crop_area = crop_width * crop_height
    area_ratio = crop_area / source_area if source_area else 0
    width_ratio = crop_width / source_width if source_width else 0
    height_ratio = crop_height / source_height if source_height else 0

    if area_ratio < 0.25:
        return False, "split_area_ratio_too_small"
    if crop_width < 320:
        return False, "split_width_too_small"
    if crop_height < 320:
        return False, "split_height_too_small"
    if width_ratio < 0.30 and height_ratio < 0.30:
        return False, "split_dimensions_too_small"
    return True, ""


def split_composite_image(
    image_path: Path,
    output_dir: Path,
    min_parts: int = 2,
    max_parts: int = 6,
) -> list[Path]:
    """Split side-by-side product panels separated by white gutters.

    Returns the original image path wrapped in a list when no split is detected.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size

        if width < 240 or height < 160:
            return [image_path]

        pixels = image.load()
        column_flags: list[bool] = []
        threshold = 248
        min_occupied_pixels = max(4, int(height * 0.08))

        for x in range(width):
            occupied = 0
            for y in range(height):
                r, g, b = pixels[x, y]
                if r < threshold or g < threshold or b < threshold:
                    occupied += 1
            column_flags.append(occupied >= min_occupied_pixels)

        runs = _find_runs(column_flags)
        runs = _merge_close_runs(runs, max_gap=max(2, int(width * 0.015)))
        min_width = max(80, int(width * 0.18))
        runs = [(start, end) for start, end in runs if end - start >= min_width]

        # Filter tiny gift/detail insets
        filtered: list[tuple[int, int]] = []
        for start, end in runs:
            keep, _ = _should_keep_split_region(
                (start, 0, end, height), (width, height),
            )
            if keep:
                filtered.append((start, end))
        runs = filtered

        if not min_parts <= len(runs) <= max_parts:
            return [image_path]

        split_paths: list[Path] = []
        padding = max(2, int(width * 0.01))
        for idx, (start, end) in enumerate(runs, start=1):
            left = max(0, start - padding)
            right = min(width, end + padding)
            crop = image.crop((left, 0, right, height))
            split_path = output_dir / f"{image_path.stem}_part_{idx:02d}.jpg"
            crop.save(split_path, format="JPEG", quality=95, optimize=True)
            split_paths.append(split_path)

        return split_paths


def preprocess_images(
    image_path: Path,
    output_dir: Path,
    config: ModuleConfig,
    apply_split: bool | None = None,
    apply_resize: bool | None = None,
    resize_size: tuple[int, int] | None = None,
) -> list[tuple[Path, SplitInfo | None]]:
    """Run resize and/or split according to *config*.

    Returns a list of ``(image_path, split_info_or_none)`` tuples.
    """
    should_split = config.auto_split_composite if apply_split is None else apply_split

    if should_split:
        split_dir = output_dir / "splits"
        split_paths = split_composite_image(
            image_path, split_dir,
            min_parts=config.split_min_parts,
            max_parts=config.split_max_parts,
        )
    else:
        split_paths = [image_path]

    is_split = len(split_paths) > 1
    results: list[tuple[Path, SplitInfo | None]] = []

    for idx, sp in enumerate(split_paths, start=1):
        si = SplitInfo(
            source_path=str(image_path),
            part_index=idx,
            total_parts=len(split_paths),
        ) if is_split else None

        should_resize = config.resize_enabled if apply_resize is None else apply_resize
        if should_resize:
            size = resize_size or config.resize_target_size
            out_name = f"{image_path.stem}_{idx:02d}_resized.jpg"
            out = output_dir / out_name
            resize_with_white_background(sp, out, size=size, quality=config.resize_quality)
            results.append((out, si))
        else:
            results.append((sp, si))

    return results
