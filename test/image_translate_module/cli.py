#!/usr/bin/env python3
"""Command-line entry point for the image-translate module.

Usage::

    # Basic: translate Chinese text on an image to English (mock mode)
    python -m image_translate_module.cli input.jpg -o output.jpg

    # With real API key
    python -m image_translate_module.cli input.jpg -o output.jpg \\
        --translate-api-key sk-xxx --translate-model deepseek-chat

    # Also resize to 660x900
    python -m image_translate_module.cli input.jpg -o output.jpg --resize

    # Also split composite images
    python -m image_translate_module.cli input.jpg -o output_dir/ --split

    # Use environment variables instead of flags
    TRANSLATE_API_KEY=sk-xxx python -m image_translate_module.cli input.jpg -o output.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect Chinese text on a product image, translate to English, "
        "erase the original text, and draw English in its place.",
    )
    parser.add_argument("image", type=Path, help="Input image path (JPG/PNG/WEBP)")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output path (image file or directory when using --split)")

    # Pre-processing
    parser.add_argument("--resize", action="store_true",
                        help="Resize to 660×900 with white background before translation")
    parser.add_argument("--split", action="store_true",
                        help="Auto-detect and split side-by-side composite product images")

    # API config
    parser.add_argument("--translate-api-key", default="",
                        help="API key for the translation LLM (default: $TRANSLATE_API_KEY)")
    parser.add_argument("--translate-base-url", default="",
                        help="Base URL for the translation LLM (default: $TRANSLATE_BASE_URL)")
    parser.add_argument("--translate-model", default="",
                        help="Model name for translation (default: $TRANSLATE_MODEL)")
    parser.add_argument("--no-mock", action="store_true",
                        help="Fail when no API key is set (default: use mock/identity translations)")

    # Output
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Also print the RegionDebug info as JSON to stdout")

    args = parser.parse_args()

    # --- late import so --help is fast ---------------------------------------
    from .config import ModuleConfig
    from .pipeline import process_image

    # Build config: CLI flags override env vars.
    config = ModuleConfig.from_env()

    if args.translate_api_key:
        config.translate_api_key = args.translate_api_key
    if args.translate_base_url:
        config.translate_base_url = args.translate_base_url
    if args.translate_model:
        config.translate_model = args.translate_model
    if args.no_mock:
        config.mock_translate_when_no_key = False
        config.mock_vision_when_no_key = False

    config.resize_enabled = args.resize
    config.auto_split_composite = args.split

    # --- determine output paths ----------------------------------------------
    image_path: Path = args.image.resolve()
    if not image_path.is_file():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    output: Path = args.output.resolve()

    if args.split or args.output.is_dir() or args.output.suffix == "":
        output_dir = output if output.is_dir() or output.suffix == "" else output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Single output file – wrap in a temp output dir.
        output_dir = output.parent
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- run -----------------------------------------------------------------
    async def _run() -> None:
        results = await process_image(
            image_path, output_dir, config,
            apply_resize=args.resize,
            apply_split=args.split,
        )

        for i, r in enumerate(results, start=1):
            status = "OK" if r.success else f"ERRORS: {'; '.join(r.errors)}"
            print(f"[{i}] {r.processed_path}  ({status})")

            regions_with_text = [rd for rd in r.regions if rd.translated and rd.replaced]
            regions_skipped = [rd for rd in r.regions if rd.skip_reason]
            print(f"    translated & rendered: {len(regions_with_text)}")
            print(f"    skipped:               {len(regions_skipped)}")
            for rd in regions_skipped:
                print(f"      - [{rd.skip_reason}] \"{rd.original_text[:40]}\"")

            if args.json_output:
                print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))

        if not results:
            print("No images processed.", file=sys.stderr)
            sys.exit(1)

    asyncio.run(_run())


if __name__ == "__main__":
    _main()
