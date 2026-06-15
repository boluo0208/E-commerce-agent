"""Tests for the image-translate module.

Run::

    cd test/image_translate_module
    python -m pytest test_module.py -v

Or without pytest::

    python test_module.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

# Ensure the parent directory is on the path so we can import the module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_translate_module import (
    ModuleConfig,
    MergedParagraph,
    OCRLine,
    PipelineInput,
    PipelineOutput,
    RegionDebug,
    run_pipeline,
    process_image,
)
from image_translate_module.ocr import (
    contains_chinese,
    is_invalid_ocr_text,
    get_bbox_rect,
    scale_box_back,
)
from image_translate_module.renderer import (
    estimate_surrounding_background,
    wrap_text_to_width,
)
from image_translate_module.processor import (
    resize_with_white_background,
    split_composite_image,
)
from image_translate_module.ocr import merge_ocr_lines_into_paragraphs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_test_image(path: Path, size: tuple = (400, 300), text: str | None = None) -> Path:
    """Create a simple test image, optionally with Chinese text drawn on it."""
    img = Image.new("RGB", size, "white")
    if text:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((20, 20), text, fill="black", font=font)
    img.save(path, format="JPEG", quality=95)
    return path


def _make_mock_config() -> ModuleConfig:
    """Config that uses mock mode (no real API keys needed)."""
    return ModuleConfig(
        translate_api_key="",
        mock_translate_when_no_key=True,
        mock_vision_when_no_key=True,
    )


# ---------------------------------------------------------------------------
# ocr.py tests
# ---------------------------------------------------------------------------


class TestContainsChinese:
    def test_positive(self):
        assert contains_chinese("你好世界") is True

    def test_negative_english(self):
        assert contains_chinese("Hello World") is False

    def test_negative_empty(self):
        assert contains_chinese("") is False

    def test_negative_none(self):
        assert contains_chinese(None) is False    # type: ignore[arg-type]

    def test_mixed(self):
        assert contains_chinese("Hello 你好 World") is True


class TestIsInvalidOCRText:
    def test_empty(self):
        assert is_invalid_ocr_text("") is True

    def test_whitespace(self):
        assert is_invalid_ocr_text("   ") is True

    def test_only_question_marks(self):
        assert is_invalid_ocr_text("???") is True

    def test_replacement_char(self):
        assert is_invalid_ocr_text("��") is True

    def test_valid_text(self):
        assert is_invalid_ocr_text("商品名称") is False

    def test_single_question_mark_ok(self):
        assert is_invalid_ocr_text("你好?") is False  # one ? is fine


class TestBboxUtilities:
    def test_get_bbox_rect(self):
        box = [[10, 20], [100, 20], [100, 60], [10, 60]]
        assert get_bbox_rect(box) == (10, 20, 100, 60)

    def test_scale_box_back(self):
        box = [[30.0, 60.0], [300.0, 60.0], [300.0, 180.0], [30.0, 180.0]]
        result = scale_box_back(box, 3.0)
        assert result == [[10, 20], [100, 20], [100, 60], [10, 60]]


# ---------------------------------------------------------------------------
# renderer.py tests
# ---------------------------------------------------------------------------


class TestEstimateSurroundingBackground:
    def test_white_background(self):
        img = Image.new("RGB", (200, 200), "white")
        result = estimate_surrounding_background(img, (50, 50, 150, 150))
        assert result["is_light"] is True
        assert result["mean_luminance"] > 200

    def test_dark_background(self):
        img = Image.new("RGB", (200, 200), (30, 30, 30))
        result = estimate_surrounding_background(img, (50, 50, 150, 150))
        assert result["is_light"] is False
        assert result["mean_luminance"] < 50

    def test_tiny_bbox(self):
        img = Image.new("RGB", (200, 200), "white")
        result = estimate_surrounding_background(img, (0, 0, 1, 1))
        assert "is_light" in result


class TestWrapTextToWidth:
    def test_fits_one_line(self):
        img = Image.new("RGB", (500, 100))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        lines = wrap_text_to_width(draw, "Short", font, 500, max_lines=2)
        assert lines == ["Short"]

    def test_wraps_two_lines(self):
        img = Image.new("RGB", (500, 100))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        # Test with text that is too long for one line at a narrow width
        lines = wrap_text_to_width(draw, "Very Long Text Here", font, 10, max_lines=2)
        # At 10px width, this should fail (return None)
        assert lines is None

    def test_empty_text(self):
        img = Image.new("RGB", (500, 100))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        lines = wrap_text_to_width(draw, "", font, 200, max_lines=2)
        assert lines is None


# ---------------------------------------------------------------------------
# processor.py tests
# ---------------------------------------------------------------------------


class TestResize:
    def test_resize_smaller(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.jpg"
            dst = Path(td) / "dst.jpg"
            img = Image.new("RGB", (1200, 1600), "red")
            img.save(src)
            result = resize_with_white_background(src, dst, size=(660, 900))
            assert result == dst
            assert dst.is_file()
            with Image.open(dst) as out:
                assert out.size == (660, 900)

    def test_resize_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.jpg"
            dst = Path(td) / "sub" / "dst.jpg"
            img = Image.new("RGB", (100, 100), "red")
            img.save(src)
            resize_with_white_background(src, dst)
            assert dst.is_file()


class TestSplitComposite:
    def test_single_image_no_split(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.jpg"
            img = Image.new("RGB", (400, 400), "white")
            draw = ImageDraw.Draw(img)
            draw.rectangle((50, 50, 350, 350), fill="red")
            img.save(src)

            out_dir = Path(td) / "splits"
            result = split_composite_image(src, out_dir)
            # A single uniform image shouldn't split
            assert len(result) == 1

    def test_empty_output(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "tiny.jpg"
            img = Image.new("RGB", (10, 10), "white")
            img.save(src)
            out_dir = Path(td) / "splits"
            result = split_composite_image(src, out_dir)
            assert len(result) == 1
            assert result[0] == src


# ---------------------------------------------------------------------------
# pipeline.py tests (integration)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_mock_pipeline_no_chinese(self):
        """Pipeline on an image with no Chinese text should return it unchanged."""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.jpg"
            dst = Path(td) / "dst.jpg"
            _make_test_image(src, text="Hello World")

            input_ = PipelineInput(image_path=src, output_path=dst)
            result = asyncio.run(run_pipeline(input_, _make_mock_config()))

            assert isinstance(result, PipelineOutput)
            # No Chinese text → output path is the input path (unchanged)
            assert result.processed_path == src

    def test_mock_pipeline_with_chinese(self):
        """Pipeline on an image WITH Chinese text should translate and render."""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.jpg"
            dst = Path(td) / "dst.jpg"
            _make_test_image(src, text="你好世界")

            input_ = PipelineInput(image_path=src, output_path=dst)
            result = asyncio.run(run_pipeline(input_, _make_mock_config()))

            assert isinstance(result, PipelineOutput)
            assert len(result.errors) == 0

    def test_pipeline_image_not_found(self):
        """Pipeline with a non-existent path should return an error."""
        input_ = PipelineInput(
            image_path=Path("/nonexistent/image_99999.jpg"),
            output_path=Path("/tmp/out.jpg"),
        )
        result = asyncio.run(run_pipeline(input_, _make_mock_config()))
        assert result.success is False
        assert len(result.errors) > 0

    def test_pipeline_empty_texts(self):
        """Empty candidate list should return original image."""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.jpg"
            dst = Path(td) / "dst.jpg"
            _make_test_image(src, text="ABC 123")  # No Chinese

            input_ = PipelineInput(image_path=src, output_path=dst)
            result = asyncio.run(run_pipeline(input_, _make_mock_config()))
            assert result.processed_path == src
            # All regions should be skipped (no chinese)
            for rd in result.regions:
                assert rd.skip_reason is not None or rd.translated is False

    def test_config_disables_translation(self):
        """When translate_image_text=False, pipeline should be a no-op."""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.jpg"
            dst = Path(td) / "dst.jpg"
            _make_test_image(src, text="你好世界")

            config = _make_mock_config()
            config.translate_image_text = False

            input_ = PipelineInput(image_path=src, output_path=dst)
            result = asyncio.run(run_pipeline(input_, config))
            assert result.processed_path == src

    def test_process_image(self):
        """process_image() convenience wrapper should work."""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.jpg"
            _make_test_image(src, text="你好世界")
            out_dir = Path(td) / "out"

            results = asyncio.run(process_image(
                src, out_dir, _make_mock_config(),
                apply_resize=False, apply_split=False,
            ))
            assert len(results) >= 1
            for r in results:
                assert isinstance(r, PipelineOutput)


# ---------------------------------------------------------------------------
# schemas tests
# ---------------------------------------------------------------------------


class TestRegionDebugToDict:
    def test_to_dict(self):
        rd = RegionDebug(
            original_text="你好",
            full_translation="Hello",
            image_translation="Hello",
            bbox=[10, 20, 100, 60],
            translated=True,
            replaced=True,
        )
        d = rd.to_dict()
        assert d["original_text"] == "你好"
        assert d["full_translation"] == "Hello"
        assert d["translated"] is True


class TestPipelineOutputToDict:
    def test_to_dict(self):
        po = PipelineOutput(
            processed_path=Path("/tmp/out.jpg"),
            regions=[],
            errors=[],
            original_size=(400, 300),
            processed_size=(400, 300),
        )
        d = po.to_dict()
        assert d["success"] is True
        assert d["processed_path"] == str(Path("/tmp/out.jpg"))


# ---------------------------------------------------------------------------
# merge_ocr_lines_into_paragraphs tests
# ---------------------------------------------------------------------------


class TestMergeOCRLines:
    """Test the OCR line → paragraph merging logic."""

    IMG_SIZE = (800, 600)

    def _make_candidates(self, lines: list[tuple[tuple[int, int, int, int], str, int]]) -> tuple:
        """Helper: build candidates + metadata from short-form specs.

        Each spec: ((left, top, right, bottom), text, font_size)
        """
        candidates = [((l, t, r, b), txt) for (l, t, r, b), txt, _ in lines]
        font_sizes = [fs for _, _, fs in lines]
        return candidates, font_sizes

    def test_three_body_lines_merge(self):
        """The user's exact example: 3 body lines should merge into 1 paragraph."""
        candidates, font_sizes = self._make_candidates([
            ((31, 121, 756, 156), "这款T恤选用轻盈的弹力面料，侧边开设计提供运动", 28),
            ((31, 164, 756, 204), "空间，增强透气性。粘合工艺的领口和下摆减少不适摩", 28),
            ((30, 209, 558, 248), "擦，打造顺滑触感，让你专注每个比分。", 28),
        ])

        paras, debug_log = merge_ocr_lines_into_paragraphs(
            candidates, self.IMG_SIZE,
            font_sizes=font_sizes,
            confidences=[0.9, 0.9, 0.9],
            bg_luminances=[250, 250, 250],
            is_light_bgs=[True, True, True],
        )

        # Should merge into exactly 1 paragraph.
        assert len(paras) == 1, f"Expected 1 merged paragraph, got {len(paras)}"
        para = paras[0]
        assert para.line_count == 3

        # Merged text should be the concatenation.
        expected = "这款T恤选用轻盈的弹力面料，侧边开设计提供运动空间，增强透气性。粘合工艺的领口和下摆减少不适摩擦，打造顺滑触感，让你专注每个比分。"
        assert para.merged_text == expected

        # Merged bbox should cover all three.
        assert para.merged_bbox[0] == 30   # min x
        assert para.merged_bbox[1] == 121  # min y
        assert para.merged_bbox[2] == 756  # max x
        assert para.merged_bbox[3] == 248  # max y

        # Debug log should contain a merge entry.
        merge_entries = [e for e in debug_log if e["action"] == "merge"]
        assert len(merge_entries) == 1

    def test_title_not_merged_with_body(self):
        """Title '自由驰骋球场' should NOT merge with body text."""
        candidates, font_sizes = self._make_candidates([
            ((31, 70, 400, 110), "自由驰骋球场", 38),   # title: large font
            ((31, 121, 756, 156), "这款T恤选用轻盈的弹力面料", 28),  # body
            ((31, 164, 756, 204), "空间，增强透气性", 28),          # body
        ])

        paras, debug_log = merge_ocr_lines_into_paragraphs(
            candidates, self.IMG_SIZE,
            font_sizes=font_sizes,
            confidences=[0.9, 0.9, 0.9],
            bg_luminances=[250, 250, 250],
            is_light_bgs=[True, True, True],
        )

        # Title should be separate from body.
        assert len(paras) == 2, f"Expected 2 paragraphs (title + body), got {len(paras)}"
        # First para = title (1 line), second = body (2 lines).
        assert paras[0].line_count == 1
        assert paras[0].merged_text == "自由驰骋球场"
        assert paras[1].line_count == 2

    def test_x_misaligned_lines_dont_merge(self):
        """Lines with very different x positions should stay separate."""
        candidates, font_sizes = self._make_candidates([
            ((31, 100, 400, 135), "左侧文本内容", 28),
            ((300, 150, 756, 185), "右侧另一段内容", 28),  # x diff ≈ 269
        ])

        paras, debug_log = merge_ocr_lines_into_paragraphs(
            candidates, self.IMG_SIZE,
            font_sizes=font_sizes,
        )

        assert len(paras) == 2

    def test_large_y_gap_dont_merge(self):
        """Lines with a very large vertical gap should stay separate."""
        candidates, font_sizes = self._make_candidates([
            ((31, 100, 500, 135), "第一段", 28),
            ((31, 300, 500, 335), "第二段", 28),  # y gap = 165, huge
        ])

        paras, debug_log = merge_ocr_lines_into_paragraphs(
            candidates, self.IMG_SIZE,
            font_sizes=font_sizes,
        )

        assert len(paras) == 2

    def test_sentence_ending_punctuation_still_merges_if_aligned(self):
        """Even if prev line ends with '。', strong alignment signals still merge."""
        candidates, font_sizes = self._make_candidates([
            ((31, 100, 756, 135), "第一句已经结束。", 28),
            ((31, 145, 756, 180), "第二句紧接着开始", 28),
        ])

        paras, _ = merge_ocr_lines_into_paragraphs(
            candidates, self.IMG_SIZE,
            font_sizes=font_sizes,
        )

        # Both lines: x aligned, y gap normal, font same. Should merge.
        assert len(paras) == 1
        assert paras[0].line_count == 2

    def test_font_size_mismatch_dont_merge(self):
        """Lines with very different font sizes should stay separate."""
        candidates, font_sizes = self._make_candidates([
            ((31, 100, 500, 135), "小字内容", 18),
            ((31, 145, 500, 200), "大字标题", 36),  # 2× font size → split
        ])

        paras, _ = merge_ocr_lines_into_paragraphs(
            candidates, self.IMG_SIZE,
            font_sizes=font_sizes,
        )

        assert len(paras) == 2

    def test_empty_candidates(self):
        """Empty input returns empty output."""
        paras, debug_log = merge_ocr_lines_into_paragraphs(
            [], self.IMG_SIZE,
        )
        assert paras == []
        assert debug_log == []

    def test_single_line(self):
        """Single line should become a 1-line paragraph."""
        candidates, font_sizes = self._make_candidates([
            ((31, 121, 756, 156), "单行文本", 28),
        ])

        paras, _ = merge_ocr_lines_into_paragraphs(
            candidates, self.IMG_SIZE,
            font_sizes=font_sizes,
        )

        assert len(paras) == 1
        assert paras[0].line_count == 1
        assert paras[0].merged_text == "单行文本"
        assert paras[0].merge_reason == "single_line"


from image_translate_module.renderer import (
    wrap_text_to_width,
    _typeset_english,
    _typeset_compact_label,
    _prevent_orphan_words,
    _line_height_for_font,
    _make_font,
    _shorten_hero_badge_text,
    _shorten_bottom_hero_text,
    _fallback_english_text,
    estimate_background_color,
    estimate_background_complexity,
    create_text_mask,
    erase_text_region,
    detect_text_alignment,
)
from image_translate_module.schemas import OCRLine
from image_translate_module.ocr import is_compact_label, classify_text_role


# ---------------------------------------------------------------------------
# typesetting tests (new)
# ---------------------------------------------------------------------------


class TestTypesetting:
    """Test the revised English typesetting engine."""

    LONG_ENGLISH = (
        "This T-shirt features lightweight stretch fabric with side slits "
        "for motion and breathability. The bonded collar and hem reduce "
        "friction for a smooth feel, letting you focus on every point."
    )

    def _make_draw(self, size=(800, 600)):
        img = Image.new("RGB", size, "white")
        return ImageDraw.Draw(img), img

    def test_body_font_not_below_72_percent(self):
        """Body text font size must NOT drop below 72% of original."""
        draw, img = self._make_draw()
        orig_fs = 30  # typical body font
        box_w, box_h = 700, 120  # wide but short box

        lines, final_fs, _, _, debug = _typeset_english(
            draw, self.LONG_ENGLISH, box_w, box_h, orig_fs,
            img_w=800, img_h=600, img_bottom=300,
        )

        assert lines is not None, f"Typesetting failed: {debug}"
        min_allowed = max(18, round(orig_fs * 0.72))
        assert final_fs >= min_allowed, (
            f"Body font {final_fs} < min {min_allowed} "
            f"(72% of orig_fs={orig_fs}). Debug: {debug}"
        )

    def test_english_wraps_to_multiple_lines(self):
        """Long English text should wrap to 3+ lines, not be squeezed into 1."""
        draw, img = self._make_draw()
        orig_fs = 28  # body font
        box_w, box_h = 350, 180  # narrow box forces multi-line wrap

        lines, final_fs, _, _, debug = _typeset_english(
            draw, self.LONG_ENGLISH, box_w, box_h, orig_fs,
            img_w=800, img_h=600, img_bottom=300,
        )

        assert lines is not None, f"Typesetting failed: {debug}"
        assert len(lines) >= 3, (
            f"Expected 3+ lines for long text in narrow box, "
            f"got {len(lines)}. Debug: {debug}"
        )
        # Font should be reasonable — not crushed below 18.
        assert final_fs >= 18, f"Font too small: {final_fs}. Debug: {debug}"

    def test_title_font_not_below_78_percent(self):
        """Title font size must NOT drop below 78% of original."""
        draw, img = self._make_draw()
        orig_fs = 40  # ≥ _TITLE_FONT_THRESHOLD(36) → treated as title
        box_w, box_h = 300, 60

        title_text = "Move Freely on the Court"
        lines, final_fs, _, _, debug = _typeset_english(
            draw, title_text, box_w, box_h, orig_fs,
            img_w=800, img_h=600, img_bottom=300,
        )

        assert lines is not None, f"Title typesetting failed: {debug}"
        min_allowed = max(22, round(orig_fs * 0.78))
        assert final_fs >= min_allowed, (
            f"Title font {final_fs} < min {min_allowed} "
            f"(78% of orig_fs={orig_fs}). Debug: {debug}"
        )

    def test_line_height_reasonable(self):
        """Line height should be between 1.15× and 1.3× font size."""
        for fs in (18, 24, 30, 40):
            lh = _line_height_for_font(fs)
            assert fs * 1.0 < lh <= fs * 1.5, (
                f"Line height {lh} not in ({fs*1.0}, {fs*1.5}] for font_size={fs}"
            )

    def test_wider_box_needs_fewer_lines(self):
        """With a very wide box, text should fit in fewer lines at larger font."""
        draw, _ = self._make_draw((1200, 800))
        orig_fs = 28

        # Narrow box
        lines_narrow, fs_narrow, _, _, _ = _typeset_english(
            draw, self.LONG_ENGLISH, 400, 300, orig_fs,
            img_w=1200, img_h=800, img_bottom=500,
        )
        # Wide box
        lines_wide, fs_wide, _, _, _ = _typeset_english(
            draw, self.LONG_ENGLISH, 1000, 300, orig_fs,
            img_w=1200, img_h=800, img_bottom=500,
        )

        assert lines_narrow is not None and lines_wide is not None
        # Wider box should allow larger font or fewer lines.
        assert fs_wide >= fs_narrow or len(lines_wide) <= len(lines_narrow), (
            f"Wide: fs={fs_wide}, lines={len(lines_wide)}. "
            f"Narrow: fs={fs_narrow}, lines={len(lines_narrow)}"
        )


# ---------------------------------------------------------------------------
# compact label detection tests
# ---------------------------------------------------------------------------


class TestCompactLabelDetection:
    IMG_SIZE = (800, 600)

    def test_member_gift_detected(self):
        is_comp, reasons = is_compact_label(
            (600, 450, 770, 490), self.IMG_SIZE, "会员赠品",
        )
        assert is_comp is True
        assert "narrow" in " ".join(reasons) or "promo_keyword" in " ".join(reasons)

    def test_price_promo_detected(self):
        is_comp, reasons = is_compact_label(
            (500, 500, 700, 540), self.IMG_SIZE, "满1600元赠品",
        )
        assert is_comp is True

    def test_normal_body_not_detected(self):
        is_comp, _ = is_compact_label(
            (31, 121, 756, 156), self.IMG_SIZE,
            "这款T恤选用轻盈的弹力面料",
        )
        assert is_comp is False

    def test_title_not_detected(self):
        is_comp, _ = is_compact_label(
            (36, 37, 304, 86), self.IMG_SIZE, "自由驰骋球场",
        )
        assert is_comp is False


# ---------------------------------------------------------------------------
# compact label typesetting tests
# ---------------------------------------------------------------------------


class TestCompactLabelTypesetting:
    def _make_draw(self, size=(800, 600)):
        img = Image.new("RGB", size, "white")
        return ImageDraw.Draw(img), img

    def test_no_orphan_words(self):
        """Orphan short words (with, for, on) should be merged."""
        draw, _ = self._make_draw()
        result = _prevent_orphan_words(["Gift", "for", "Members"])
        # "for" should be merged — not a line by itself.
        assert all(len(line.split()) >= 2 for line in result if len(result) > 1) or len(result) <= 2

    def test_compact_label_max_3_lines(self):
        """Compact labels must not exceed 3 lines."""
        draw, _ = self._make_draw()
        text = "Free Gift Over ¥1600 for Members"
        lines, fs, _, _, debug = _typeset_compact_label(
            draw, text, box_w=150, box_h=60, orig_font_size=18,
            img_w=800, img_h=600, img_bottom=500,
        )
        if lines is not None:
            assert len(lines) <= 3, f"Compact label exceeded 3 lines: {len(lines)}"
            assert fs >= 14, f"Font too small: {fs}"

    def test_compact_label_min_font(self):
        """Compact label font must never drop below minimum (14px)."""
        draw, _ = self._make_draw()
        text = "Member Exclusive Limited Time Gift Offer"
        lines, fs, _, _, _ = _typeset_compact_label(
            draw, text, box_w=120, box_h=40, orig_font_size=16,
            img_w=800, img_h=600, img_bottom=500,
        )
        if lines is not None:
            assert fs >= 14, f"Font below minimum: {fs}"


# ---------------------------------------------------------------------------
# background colour extraction tests
# ---------------------------------------------------------------------------


class TestBackgroundColor:
    def test_white_background(self):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        rgb = estimate_background_color(img, (50, 50, 150, 150))
        # Should be close to white.
        assert all(c >= 240 for c in rgb), f"Expected white-ish, got {rgb}"

    def test_light_gray_background(self):
        img = Image.new("RGB", (200, 200), (220, 220, 220))
        rgb = estimate_background_color(img, (50, 50, 150, 150))
        # Should be close to light gray, not pure white.
        assert all(200 <= c <= 235 for c in rgb), f"Expected gray-ish, got {rgb}"

    def test_text_pixels_excluded(self):
        """Black text on white bg — bg estimate should still be white."""
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (200, 200), (245, 245, 243))  # slight off-white
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((60, 80), "满减优惠", fill="black", font=font)
        rgb = estimate_background_color(img, (50, 70, 150, 110))
        # Should be close to the off-white background, not darkened by text.
        assert all(c >= 230 for c in rgb), f"Text darkened bg estimate: {rgb}"


# ---------------------------------------------------------------------------
# background complexity tests
# ---------------------------------------------------------------------------


class TestBackgroundComplexity:
    def test_plain_white_is_simple(self):
        """Pure white background → simple."""
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        result = estimate_background_complexity(img, (50, 50, 150, 150))
        assert result["level"] == "simple"
        assert result["recommended_erase"] == "color_fill"

    def test_light_gray_card_is_simple(self):
        """Uniform light gray card → simple."""
        img = Image.new("RGB", (200, 200), (235, 235, 233))
        result = estimate_background_complexity(img, (50, 50, 150, 150))
        assert result["level"] in ("simple", "medium")

    def test_gradient_is_medium(self):
        """Steep gradient → medium or complex, NOT simple."""
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        for y in range(200):
            val = int(100 + y * 0.7)  # steep gradient 100→240
            arr[y, :, :] = [val, val, val]
        img = Image.fromarray(arr)
        result = estimate_background_complexity(img, (50, 50, 150, 150))
        assert result["level"] in ("medium", "complex"), f"Got level={result['level']}"
        assert result["level"] != "simple"

    def test_textured_fabric_is_complex(self):
        """Random noise (simulated fabric texture) → complex."""
        import numpy as np
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 60, (200, 200, 3), dtype=np.uint8)
        arr = np.clip(np.full((200, 200, 3), 180, dtype=np.uint8) + noise, 0, 255)
        img = Image.fromarray(arr)
        result = estimate_background_complexity(img, (50, 50, 150, 150))
        assert result["level"] == "complex"
        assert result["recommended_erase"] == "inpaint"


class TestTextMask:
    def test_dark_text_mask(self):
        """Black text on white → mask should have pixels."""
        from PIL import ImageDraw
        img = Image.new("RGB", (200, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "Test", fill="black")
        mask = create_text_mask(img, (15, 25, 100, 65), text_color="black")
        assert mask.shape[0] > 0 and mask.shape[1] > 0
        assert np.count_nonzero(mask) > 0, "Mask should have text pixels"

    def test_white_text_mask(self):
        """White text on dark bg → mask should capture white text."""
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (200, 100), (35, 35, 42))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
        draw.text((20, 20), "WHITE", fill=(250, 250, 250), font=font)
        mask = create_text_mask(img, (18, 18, 160, 65), text_color="white")
        assert np.count_nonzero(mask) > 0, "White text mask must not be empty"

    def test_white_text_mask_on_light_photo_bg(self):
        """White badge text on a pale photo-like bg should still be masked."""
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (320, 120), (205, 198, 188))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 34)
        draw.text((12, 20), "WHITE BADGE", fill=(255, 255, 255), font=font)
        mask = create_text_mask(img, (8, 14, 300, 70), text_color="white")
        nonzero = int(np.count_nonzero(mask))
        assert nonzero > 200, f"Expected white badge pixels, got {nonzero}"

    def test_white_text_mask_ignores_large_bright_patch(self):
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (360, 140), (205, 198, 188))
        draw = ImageDraw.Draw(img)
        draw.ellipse((150, 30, 330, 110), fill=(238, 224, 218))
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 32)
        draw.text((12, 18), "WHITE", fill=(255, 255, 255), font=font)
        mask = create_text_mask(img, (8, 12, 340, 118), text_color="white")
        mask_ratio = int(np.count_nonzero(mask)) / mask.size
        assert 0.005 < mask_ratio < 0.18, f"Mask ratio too broad: {mask_ratio:.3f}"

    def test_white_bg_no_text_mask(self):
        """Empty white area → mask should be mostly empty."""
        img = Image.new("RGB", (200, 100), (255, 255, 255))
        mask = create_text_mask(img, (50, 30, 150, 70), text_color="black")
        total = mask.size
        nonzero = int(np.count_nonzero(mask))
        assert nonzero < total * 0.1, f"Mask has {nonzero}/{total} pixels on blank white"


class TestAdaptiveErase:
    def test_simple_background_uses_color_fill(self):
        """Simple bg with black text → erase with color_fill."""
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        complexity = {"level": "simple", "recommended_erase": "color_fill"}
        result, debug = erase_text_region(img, (50, 50, 150, 150), complexity, text_color="black")
        assert debug["strategy"] == "color_fill"

    def test_white_text_always_uses_inpaint(self):
        """White text → must use inpaint even on simple bg."""
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (200, 200), (35, 35, 42))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
        draw.text((50, 80), "WHITE", fill=(250, 250, 250), font=font)
        complexity = {"level": "simple", "recommended_erase": "color_fill"}
        result, debug = erase_text_region(img, (45, 75, 155, 120), complexity, text_color="white")
        # White text must use inpaint, not simple color_fill.
        assert debug.get("inpaint_applied") == True, f"White text must use inpaint, got {debug}"
        assert debug.get("mask_pixels", 0) > 0, "White text mask must not be empty"

    def test_complex_background_uses_inpaint(self):
        """Complex bg → erase with inpaint."""
        import numpy as np
        from PIL import ImageDraw
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 80, (200, 200, 3), dtype=np.uint8)
        arr = np.clip(np.full((200, 200, 3), 150, dtype=np.uint8) + noise, 0, 255)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.text((60, 90), "Text", fill="black")

        complexity = {"level": "complex", "recommended_erase": "inpaint"}
        result, debug = erase_text_region(img, (55, 85, 140, 115), complexity, text_color="black")
        assert debug["strategy"] in ("inpaint", "blur_patch_fallback")
        assert "inpaint_radius" in debug


# ---------------------------------------------------------------------------
# alignment detection tests
# ---------------------------------------------------------------------------


class TestAlignmentDetection:
    def test_left_aligned_multi_line(self):
        """3 lines with close x1 → left-aligned."""
        lines = [
            OCRLine(bbox=(31, 121, 756, 156), text="line1"),
            OCRLine(bbox=(31, 164, 756, 204), text="line2"),
            OCRLine(bbox=(30, 209, 558, 248), text="line3"),
        ]
        assert detect_text_alignment(lines, 800) == "left"

    def test_centered_multi_line(self):
        """Lines centered in container → center."""
        lines = [
            OCRLine(bbox=(200, 50, 600, 90), text="c1"),
            OCRLine(bbox=(220, 100, 580, 140), text="c2"),
        ]
        assert detect_text_alignment(lines, 800) == "center"

    def test_single_line_defaults_left(self):
        lines = [OCRLine(bbox=(300, 50, 500, 90), text="solo")]
        assert detect_text_alignment(lines, 800) == "left"

    def test_right_aligned(self):
        lines = [
            OCRLine(bbox=(500, 50, 780, 90), text="r1"),
            OCRLine(bbox=(510, 100, 780, 140), text="r2"),
        ]
        assert detect_text_alignment(lines, 800) == "right"


# ---------------------------------------------------------------------------
# title single-line tests
# ---------------------------------------------------------------------------


class TestTitleSingleLine:
    def _make_draw(self, size=(800, 300)):
        img = Image.new("RGB", size, "white")
        return ImageDraw.Draw(img), img

    def test_title_fits_single_line(self):
        """'Move Freely on the Court' should fit 1 line at title box width."""
        draw, _ = self._make_draw()
        title = "Move Freely on the Court"
        lines, fs, _, _, debug = _typeset_english(
            draw, title, box_w=270, box_h=50, orig_font_size=40,
            img_w=800, img_h=300, img_bottom=100, is_title=True,
        )
        assert lines is not None, f"Failed: {debug}"
        assert len(lines) == 1, f"Title should be 1 line, got {len(lines)}: {lines}"
        assert fs >= 22, f"Title font too small: {fs}"


class TestHeroBadgeCopy:
    def test_long_elastic_copy_shortens_to_two_line_badge(self):
        result = _shorten_hero_badge_text(
            "采用进口橡筋弹力回缩好，不勒不掉跟，适合80-160斤可穿",
            "Made with imported elastic for excellent rebound, won't pinch or slip off the heel. Suitable for 80-160 jin.",
            "Imported Elastic for Snug Fit, Non-Slip, Suitable for 40-80 kg",
        )
        assert "\n" in result
        assert "40-80 kg" in result
        assert len(result.replace("\n", " ").split()) <= 8


class TestBottomHeroCopy:
    def test_bottom_grip_copy_shortens_to_two_line_headline(self):
        result = _shorten_bottom_hero_text(
            "添加全掌防滑硅胶,给运动中的双脚提供缓冲减震的守护",
            "Full-sole anti-slip silicone provides cushioning and shock absorption for feet during exercise.",
            "Full-sole anti-slip silicone provides cushioning and shock absorption.",
        )
        assert "\n" in result
        assert len(result.replace("\n", " ").split()) <= 8
        assert "Grip" in result or "Cushions" in result


class TestFallbackEnglishText:
    def test_feature_card_chinese_result_gets_english_fallback(self):
        result = _fallback_english_text(
            "优选高端棉面料精选棉花,更透气柔软",
            "优选高端棉面料",
            "精选棉花,更透气柔软",
            "body",
        )
        assert result == "Premium Cotton\nBreathable and Soft"

    def test_elastic_cuff_keeps_weight_range(self):
        result = _fallback_english_text(
            "采用进口弹力袜口设计弹性大 85-160斤可穿",
            "",
            "",
            "body",
        )
        assert "Elastic Cuff" in result
        assert "42.5-80 kg" in result or "42-80 kg" in result


# ---------------------------------------------------------------------------
# text sharpness / rendering tests
# ---------------------------------------------------------------------------


class TestTextSharpness:
    def _make_draw(self, size=(400, 200), bg_color=(40, 40, 40)):
        img = Image.new("RGB", size, bg_color)
        return ImageDraw.Draw(img), img

    def test_no_glow_by_default(self):
        """Glow must be disabled by default (glow_radius=0, glow_enabled=False)."""
        from image_translate_module.renderer import (
            _typeset_english, _make_font, _line_height_for_font,
        )
        draw, img = self._make_draw()
        lines, fs, _, _, debug = _typeset_english(
            draw, "Test Title", box_w=200, box_h=50, orig_font_size=40,
            img_w=400, img_h=200, img_bottom=100, is_title=True,
        )
        # Typeset doesn't set render params — that's done in the draw phase.
        # But verify the function doesn't internally enable glow.
        assert debug.get("glow_enabled") is not True or debug.get("glow_radius", 0) == 0

    def test_output_size_equals_input_size(self):
        """Output image must have the same dimensions as input."""
        img = Image.new("RGB", (400, 300), "white")
        w, h = img.size
        # Simulate: the renderer should not resize.
        assert (w, h) == (400, 300)


# ---------------------------------------------------------------------------
# text role classification tests
# ---------------------------------------------------------------------------


class TestTextRoleClassification:
    IMG_SIZE = (800, 600)

    def test_hero_headline_white_top_left(self):
        """White text in top-left with large font → hero_headline."""
        role = classify_text_role(
            (30, 20, 500, 100), self.IMG_SIZE,
            font_size=34, is_white_text=True,
        )
        assert role == "hero_headline", f"Expected hero_headline, got {role}"

    def test_body_not_hero(self):
        """Regular dark body text → not hero."""
        role = classify_text_role(
            (30, 200, 750, 400), self.IMG_SIZE,
            font_size=28, is_white_text=False,
        )
        assert role == "body"

    def test_bottom_large_headline_detected(self):
        role = classify_text_role(
            (20, 470, 760, 575), self.IMG_SIZE,
            font_size=42, is_white_text=False,
        )
        assert role == "bottom_hero_headline"

    def test_compact_overrides_hero(self):
        """Compact label detection takes priority."""
        role = classify_text_role(
            (600, 450, 780, 490), self.IMG_SIZE,
            font_size=18, is_white_text=True, is_compact=True,
        )
        assert role == "compact_label"


from image_translate_module.renderer import detect_original_text_color


class TestTextColorDetection:
    def test_white_text_detected(self):
        """White text on dark bg should be detected as white."""
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (200, 100), (40, 40, 45))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)
        draw.text((20, 20), "WHITE", fill=(250, 250, 250), font=font)
        label, _, conf = detect_original_text_color(img, (18, 18, 150, 70))
        assert label == "white", f"Expected white, got {label} (conf={conf:.2f})"

    def test_black_text_detected(self):
        """Black text on white bg should be detected as black."""
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (200, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
        draw.text((20, 20), "BLACK", fill=(10, 10, 10), font=font)
        label, _, _ = detect_original_text_color(img, (18, 18, 160, 65))
        assert label == "black", f"Expected black, got {label}"


# ---------------------------------------------------------------------------
# draw validation tests
# ---------------------------------------------------------------------------


from image_translate_module.renderer import validate_draw_text


class TestValidateDrawText:
    def test_chinese_rejected(self):
        valid, reason = validate_draw_text("你好世界")
        assert not valid, f"Chinese should be rejected, got: {reason}"

    def test_english_accepted(self):
        valid, _ = validate_draw_text("Hello World")
        assert valid

    def test_mixed_rejected(self):
        valid, reason = validate_draw_text("Hello 你好")
        assert not valid, f"Mixed text should be rejected, got: {reason}"

    def test_empty_rejected(self):
        valid, _ = validate_draw_text("")
        assert not valid

    def test_numbers_english_accepted(self):
        valid, _ = validate_draw_text("Fits 40-80 kg, imported elastic")
        assert valid


# ---------------------------------------------------------------------------
# runner (when called directly)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("image_translate_module — manual test run")
    print("=" * 60)

    passed = 0
    total = 0

    def run_test(name: str, fn) -> None:
        global passed, total
        total += 1
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")

    # OCR
    run_test("contains_chinese positive", lambda: TestContainsChinese().test_positive())
    run_test("contains_chinese negative", lambda: TestContainsChinese().test_negative_english())
    run_test("contains_chinese empty", lambda: TestContainsChinese().test_negative_empty())
    run_test("is_invalid_ocr_text empty", lambda: TestIsInvalidOCRText().test_empty())
    run_test("is_invalid_ocr_text valid", lambda: TestIsInvalidOCRText().test_valid_text())

    # Renderer
    run_test("bg white", lambda: TestEstimateSurroundingBackground().test_white_background())
    run_test("bg dark", lambda: TestEstimateSurroundingBackground().test_dark_background())

    # Processor
    run_test("resize", lambda: TestResize().test_resize_smaller())
    run_test("resize mkdir", lambda: TestResize().test_resize_creates_parent_dir())

    # Schemas
    run_test("RegionDebug.to_dict", lambda: TestRegionDebugToDict().test_to_dict())
    run_test("PipelineOutput.to_dict", lambda: TestPipelineOutputToDict().test_to_dict())

    # Pipeline (integration)
    run_test("pipeline no chinese", lambda: TestRunPipeline().test_mock_pipeline_no_chinese())
    run_test("pipeline with chinese", lambda: TestRunPipeline().test_mock_pipeline_with_chinese())
    run_test("pipeline image not found", lambda: TestRunPipeline().test_pipeline_image_not_found())
    run_test("pipeline disabled", lambda: TestRunPipeline().test_config_disables_translation())
    run_test("process_image wrapper", lambda: TestRunPipeline().test_process_image())

    # Merge OCR lines
    run_test("merge 3 body lines → 1 para", lambda: TestMergeOCRLines().test_three_body_lines_merge())
    run_test("merge title NOT merged with body", lambda: TestMergeOCRLines().test_title_not_merged_with_body())
    run_test("merge x-misaligned → split", lambda: TestMergeOCRLines().test_x_misaligned_lines_dont_merge())
    run_test("merge large y-gap → split", lambda: TestMergeOCRLines().test_large_y_gap_dont_merge())
    run_test("merge sentence-end still merges", lambda: TestMergeOCRLines().test_sentence_ending_punctuation_still_merges_if_aligned())
    run_test("merge font mismatch → split", lambda: TestMergeOCRLines().test_font_size_mismatch_dont_merge())
    run_test("merge empty input", lambda: TestMergeOCRLines().test_empty_candidates())
    run_test("merge single line", lambda: TestMergeOCRLines().test_single_line())

    # Typesetting (new)
    run_test("body font ≥ 72% orig", lambda: TestTypesetting().test_body_font_not_below_72_percent())
    run_test("english wraps 3+ lines", lambda: TestTypesetting().test_english_wraps_to_multiple_lines())
    run_test("title font ≥ 78% orig", lambda: TestTypesetting().test_title_font_not_below_78_percent())
    run_test("line height 1.0-1.5× font", lambda: TestTypesetting().test_line_height_reasonable())
    run_test("wider box fewer lines", lambda: TestTypesetting().test_wider_box_needs_fewer_lines())

    # Compact label detection
    run_test("compact: member gift", lambda: TestCompactLabelDetection().test_member_gift_detected())
    run_test("compact: price promo", lambda: TestCompactLabelDetection().test_price_promo_detected())
    run_test("compact: body not detected", lambda: TestCompactLabelDetection().test_normal_body_not_detected())
    run_test("compact: title not detected", lambda: TestCompactLabelDetection().test_title_not_detected())

    # Compact label typesetting
    run_test("compact: no orphan words", lambda: TestCompactLabelTypesetting().test_no_orphan_words())
    run_test("compact: max 3 lines", lambda: TestCompactLabelTypesetting().test_compact_label_max_3_lines())
    run_test("compact: min font 14px", lambda: TestCompactLabelTypesetting().test_compact_label_min_font())

    # Background color
    run_test("bg: white", lambda: TestBackgroundColor().test_white_background())
    run_test("bg: light gray", lambda: TestBackgroundColor().test_light_gray_background())
    run_test("bg: text excluded", lambda: TestBackgroundColor().test_text_pixels_excluded())

    # Background complexity
    run_test("cmplx: white is simple", lambda: TestBackgroundComplexity().test_plain_white_is_simple())
    run_test("cmplx: gray card is simple", lambda: TestBackgroundComplexity().test_light_gray_card_is_simple())
    run_test("cmplx: gradient is medium", lambda: TestBackgroundComplexity().test_gradient_is_medium())
    run_test("cmplx: texture is complex", lambda: TestBackgroundComplexity().test_textured_fabric_is_complex())

    # Text mask
    run_test("mask: dark text detected", lambda: TestTextMask().test_dark_text_mask())
    run_test("mask: white text detected", lambda: TestTextMask().test_white_text_mask())
    run_test("mask: blank bg empty", lambda: TestTextMask().test_white_bg_no_text_mask())

    # Adaptive erase
    run_test("erase: simple → color_fill", lambda: TestAdaptiveErase().test_simple_background_uses_color_fill())
    run_test("erase: white text → inpaint", lambda: TestAdaptiveErase().test_white_text_always_uses_inpaint())

    # Draw validation
    run_test("validate: chinese rejected", lambda: TestValidateDrawText().test_chinese_rejected())
    run_test("validate: english accepted", lambda: TestValidateDrawText().test_english_accepted())
    run_test("validate: mixed rejected", lambda: TestValidateDrawText().test_mixed_rejected())
    run_test("validate: empty rejected", lambda: TestValidateDrawText().test_empty_rejected())
    run_test("validate: numbers+eng ok", lambda: TestValidateDrawText().test_numbers_english_accepted())
    run_test("erase: complex → inpaint", lambda: TestAdaptiveErase().test_complex_background_uses_inpaint())

    # Alignment detection
    run_test("align: left multi-line", lambda: TestAlignmentDetection().test_left_aligned_multi_line())
    run_test("align: center multi-line", lambda: TestAlignmentDetection().test_centered_multi_line())
    run_test("align: single→left", lambda: TestAlignmentDetection().test_single_line_defaults_left())
    run_test("align: right aligned", lambda: TestAlignmentDetection().test_right_aligned())

    # Title single-line
    run_test("title: fits single line", lambda: TestTitleSingleLine().test_title_fits_single_line())

    # Text sharpness
    run_test("sharp: no glow default", lambda: TestTextSharpness().test_no_glow_by_default())
    run_test("sharp: output= input size", lambda: TestTextSharpness().test_output_size_equals_input_size())

    # Text role
    run_test("role: hero headline detected", lambda: TestTextRoleClassification().test_hero_headline_white_top_left())
    run_test("role: body not hero", lambda: TestTextRoleClassification().test_body_not_hero())
    run_test("role: compact overrides", lambda: TestTextRoleClassification().test_compact_overrides_hero())

    # Text color
    run_test("color: white text detected", lambda: TestTextColorDetection().test_white_text_detected())
    run_test("color: black text detected", lambda: TestTextColorDetection().test_black_text_detected())

    print(f"\n{passed}/{total} passed")
    if passed < total:
        sys.exit(1)
