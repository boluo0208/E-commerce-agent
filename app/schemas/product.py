from pydantic import BaseModel, Field


class VisionResult(BaseModel):
    category: str = Field(description="Recognized product category")
    color: str = Field(description="Visible product color")
    color_label: str = Field(
        default="unknown",
        description="Single normalized product color label, ignoring background, model, and accessories.",
    )
    style: str = Field(description="Visible product style")
    visible_features: list[str] = Field(default_factory=list)
    image_width: int
    image_height: int
    note: str = ""


class ProductContent(BaseModel):
    english_title: str
    arabic_title: str
    chinese_description: str
    english_description: str
    safety_notes: list[str] = Field(default_factory=list)


class ProductExportRow(BaseModel):
    chinese_title: str
    english_title: str
    arabic_title: str
    overall_color: str = ""
    color_label: str = ""
    chinese_description: str
    english_description: str
    image_file: str
