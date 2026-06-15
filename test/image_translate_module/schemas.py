"""Unified data structures and exceptions for the image-translate module."""

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModuleError(Exception):
    """Base exception for all module errors."""


class ConfigError(ModuleError):
    """Configuration is missing or invalid."""


class ImageError(ModuleError):
    """Image cannot be read or processed."""


class TranslateError(ModuleError):
    """Translation API call failed."""


class OCRError(ModuleError):
    """OCR engine failed or returned unusable results."""


# ---------------------------------------------------------------------------
# Input / output structures
# ---------------------------------------------------------------------------


@dataclass
class RegionDebug:
    """Per-region debug record – mirrors the original _make_debug dict."""

    original_text: str = ""
    full_translation: str = ""
    image_translation: str = ""
    bbox: list[int] = field(default_factory=list)  # [left, top, right, bottom]
    confidence: float = 0.0
    confidence_threshold: float = 0.0
    estimated_original_font_size: int = 0
    used_font_size: int = 0
    final_estimated_text_height: float = 0.0
    background_luminance: float = 0.0
    text_color: str = ""
    detected_by: str = "rapidocr"
    translated: bool = False
    replaced: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "full_translation": self.full_translation,
            "image_translation": self.image_translation,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "confidence_threshold": self.confidence_threshold,
            "estimated_original_font_size": self.estimated_original_font_size,
            "used_font_size": self.used_font_size,
            "final_estimated_text_height": self.final_estimated_text_height,
            "background_luminance": self.background_luminance,
            "text_color": self.text_color,
            "detected_by": self.detected_by,
            "translated": self.translated,
            "replaced": self.replaced,
            "skip_reason": self.skip_reason,
        }


@dataclass
class OCRLine:
    """One line of OCR-detected text with its bounding box and metadata."""

    text: str = ""
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # (left, top, right, bottom)
    confidence: float = 0.0
    estimated_font_size: int = 0
    background_luminance: float = 0.0
    is_light_background: bool = True
    detected_by: str = "rapidocr"


@dataclass
class MergedParagraph:
    """Multiple OCR lines merged into one logical paragraph.

    The *merged_text* is the concatenated Chinese text (with spaces removed).
    Each original line is preserved in *lines* for debugging.
    """

    merged_text: str = ""
    lines: list[OCRLine] = field(default_factory=list)
    merged_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # (min_x, min_y, max_x, max_y)
    estimated_font_size: int = 0
    merge_reason: str = ""  # e.g. "x_aligned,line_spacing,font_size,text_continuity"

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def original_texts(self) -> list[str]:
        return [line.text for line in self.lines]


@dataclass
class SplitInfo:
    """Record of a composite-image split operation."""

    source_path: str = ""
    part_index: int = 0
    total_parts: int = 0


@dataclass
class PipelineInput:
    """What goes into the pipeline."""

    image_path: Path
    output_path: Path
    # Optional pre-processing
    apply_resize: bool | None = None  # None → use config default
    apply_split: bool | None = None
    resize_size: tuple[int, int] | None = None


@dataclass
class PipelineOutput:
    """What comes out of the pipeline."""

    processed_path: Path                     # path to the final output image
    regions: list[RegionDebug] = field(default_factory=list)
    split_info: SplitInfo | None = None     # set when the image was split from a composite
    errors: list[str] = field(default_factory=list)
    original_size: tuple[int, int] = (0, 0)
    processed_size: tuple[int, int] = (0, 0)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "processed_path": str(self.processed_path),
            "regions": [r.to_dict() for r in self.regions],
            "split_info": {
                "source_path": self.split_info.source_path,
                "part_index": self.split_info.part_index,
                "total_parts": self.split_info.total_parts,
            } if self.split_info else None,
            "errors": self.errors,
            "original_size": list(self.original_size),
            "processed_size": list(self.processed_size),
            "success": self.success,
        }
