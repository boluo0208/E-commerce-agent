from pathlib import Path

from PIL import Image, ImageOps


def _find_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None

    for index, value in enumerate(flags):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
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
        previous_start, previous_end = merged[-1]
        if start - previous_end <= max_gap:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))

    return merged


def should_keep_split_region(
    crop_bbox: tuple[int, int, int, int],
    source_size: tuple[int, int],
) -> tuple[bool, str]:
    """Return (keep, reason).  Use this to reject tiny gift/inset regions."""
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

    Returns the original image path when no reliable split is detected.
    Skips regions that are too small (likely inset/gift images, not product panels).
    """

    MIN_SPLIT_AREA_RATIO = 0.25
    MIN_SPLIT_WIDTH_RATIO = 0.30
    MIN_SPLIT_HEIGHT_RATIO = 0.30
    MIN_SPLIT_ABSOLUTE_WIDTH = 320
    MIN_SPLIT_ABSOLUTE_HEIGHT = 320

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

        # Strict size check — skip tiny gift / detail insets that would
        # become blurry when scaled to 660×900.
        total_area = width * height
        filtered: list[tuple[int, int]] = []
        for start, end in runs:
            crop_w = end - start
            crop_h = height  # column split → full height
            keep, _reason = should_keep_split_region(
                (start, 0, end, height),
                (width, height),
            )
            if keep:
                filtered.append((start, end))
        runs = filtered

        if not min_parts <= len(runs) <= max_parts:
            return [image_path]

        split_paths: list[Path] = []
        padding = max(2, int(width * 0.01))
        for index, (start, end) in enumerate(runs, start=1):
            left = max(0, start - padding)
            right = min(width, end + padding)
            crop = image.crop((left, 0, right, height))
            split_path = output_dir / f"{image_path.stem}_part_{index:02d}.jpg"
            crop.save(split_path, format="JPEG", quality=95, optimize=True)
            split_paths.append(split_path)

        return split_paths
