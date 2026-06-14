from pathlib import Path

from PIL import Image, ImageOps


def resize_with_white_background(
    image_path: Path,
    output_path: Path,
    size: tuple[int, int] = (660, 900),
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", size, "white")
        left = (size[0] - image.width) // 2
        top = (size[1] - image.height) // 2
        canvas.paste(image, (left, top))
        canvas.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=True)

    return output_path
