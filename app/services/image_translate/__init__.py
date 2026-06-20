"""Image Translate Module — independent OCR + translate + render pipeline.

Usage::

    import asyncio
    from pathlib import Path
    from image_translate_module import ModuleConfig, PipelineInput, run_pipeline

    config = ModuleConfig(
        translate_api_key="sk-...",
        translate_model="deepseek-chat",
        mock_translate_when_no_key=True,   # use mock when no key
    )

    input_ = PipelineInput(
        image_path=Path("input.jpg"),
        output_path=Path("output.jpg"),
    )

    result = asyncio.run(run_pipeline(input_, config))
    print(result.to_dict())
"""

from .config import ModuleConfig, get_config, set_config
from .ocr import merge_ocr_lines_into_paragraphs
from .pipeline import process_image, run_pipeline
from .renderer import draw_merged_paragraph_translations, erase_and_draw_merged_paragraphs
from .schemas import (
    ConfigError,
    ImageError,
    MergedParagraph,
    ModuleError,
    OCRLine,
    OCRError,
    PipelineInput,
    PipelineOutput,
    RegionDebug,
    SplitInfo,
    TranslateError,
)
from .seedream_erase import erase_chinese_text_with_seedream

__all__ = [
    # config
    "ModuleConfig",
    "get_config",
    "set_config",
    # pipeline
    "run_pipeline",
    "process_image",
    # ocr
    "merge_ocr_lines_into_paragraphs",
    # renderer
    "erase_and_draw_merged_paragraphs",
    "draw_merged_paragraph_translations",
    # seedream
    "erase_chinese_text_with_seedream",
    # schemas
    "PipelineInput",
    "PipelineOutput",
    "RegionDebug",
    "OCRLine",
    "MergedParagraph",
    "SplitInfo",
    # exceptions
    "ModuleError",
    "ConfigError",
    "ImageError",
    "OCRError",
    "TranslateError",
]
