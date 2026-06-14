from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

from app.schemas.product import ProductExportRow


HEADERS = [
    "中文标题",
    "英文标题",
    "阿拉伯语标题",
    "总体颜色",
    "中文描述",
    "英文描述",
    "图片文件",
]


def export_to_excel(rows: list[ProductExportRow], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "products"
    sheet.append(HEADERS)

    for row in rows:
        sheet.append(
            [
                row.chinese_title,
                row.english_title,
                row.arabic_title,
                row.overall_color,
                row.chinese_description,
                row.english_description,
                row.image_file,
            ]
        )

    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    widths = [28, 42, 42, 20, 60, 60, 28]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[chr(64 + index)].width = width

    for row_cells in sheet.iter_rows(min_row=2):
        for cell in row_cells:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    workbook.save(output_path)
    return output_path


def create_export_zip(
    excel_path: Path,
    image_paths: list[Path],
    output_path: Path,
    extra_files: list[Path] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_path, "w", ZIP_DEFLATED) as zip_file:
        zip_file.write(excel_path, "products.xlsx")
        for image_path in image_paths:
            zip_file.write(image_path, f"images/{image_path.name}")
        for extra_path in extra_files or []:
            zip_file.write(extra_path, extra_path.name)

    return output_path
