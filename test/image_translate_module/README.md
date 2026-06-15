# Image Translate Module

独立模块：**OCR 识别图片中的中文 → 调用 LLM 翻译成英文 → 擦除原图中文 → 绘制英文**。

从 `E-commerce-agent` 主项目提取，零侵入——不依赖原项目的任何代码，可独立运行、独立测试。

## 目录结构

```
image_translate_module/
├── __init__.py        # 公开 API
├── config.py          # ModuleConfig（独立配置，通过构造函数/env 注入）
├── schemas.py         # PipelineInput, PipelineOutput, RegionDebug, 异常类
├── processor.py       # 图片预处理：resize（660×900 白底）+ split 水平拆分
├── ocr.py             # RapidOCR 本地 OCR + 中文检测 + bbox 工具
├── translator.py      # DeepSeek/OpenAI-compatible API 中文→英文翻译
├── renderer.py        # 擦除原图中文 + 绘制英文排版（背景检测、字号估算、换行）
├── pipeline.py        # 总管道：预处理→OCR→翻译→渲染（含 Mimo 可选回退）
├── cli.py             # 命令行入口
├── api_server.py      # FastAPI 测试壳（可选）
├── test_module.py     # 单元测试 + 集成测试
└── README.md          # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install Pillow rapidocr-onnxruntime httpx numpy
# 可选：CLI 测试
pip install fastapi uvicorn python-multipart
```

### 2. 命令行使用

```bash
# Mock 模式（不需要 API Key，返回原文/不做真实翻译）
cd test/image_translate_module
python cli.py input.jpg -o output.jpg

# 使用真实 API
python cli.py input.jpg -o output.jpg \
    --translate-api-key sk-xxx \
    --translate-model deepseek-chat

# 查看调试 JSON
python cli.py input.jpg -o output.jpg --json

# 同时做 resize + split
python cli.py product.jpg -o output/ --resize --split
```

### 3. 代码调用

```python
import asyncio
from pathlib import Path
from image_translate_module import (
    ModuleConfig, PipelineInput, run_pipeline
)

config = ModuleConfig(
    translate_api_key="sk-xxx",
    translate_model="deepseek-chat",
    mock_translate_when_no_key=True,
)

input_ = PipelineInput(
    image_path=Path("input.jpg"),
    output_path=Path("output.jpg"),
)

result = asyncio.run(run_pipeline(input_, config))
print(result.to_dict())
```

### 4. FastAPI 测试服务

```bash
cd test/image_translate_module
python api_server.py
# 打开 http://127.0.0.1:8001/docs
```

### 5. 运行测试

```bash
cd test/image_translate_module
python test_module.py
# 或
python -m pytest test_module.py -v
```

## 配置项

所有配置通过 `ModuleConfig` 传入，或从环境变量读取：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `TRANSLATE_API_KEY` | `""` | LLM API Key（也读 `DEEPSEEK_API_KEY`） |
| `TRANSLATE_BASE_URL` | `https://api.deepseek.com` | LLM 接口地址 |
| `TRANSLATE_MODEL` | `deepseek-chat` | 模型名称 |
| `TRANSLATE_TEMPERATURE` | `0.05` | 翻译温度 |
| `MIMO_API_KEY` | `""` | Mimo 视觉 API Key（可选回退） |
| `MIMO_BASE_URL` | `""` | Mimo 接口地址 |
| `MIMO_MODEL` | `""` | Mimo 模型名 |
| `MOCK_TRANSLATE_WHEN_NO_KEY` | `true` | 无 Key 时用 mock（返回原文） |
| `TRANSLATE_IMAGE_TEXT` | `true` | 是否执行翻译管道 |
| `OCR_SCALE_FACTOR` | `3` | OCR 时的放大倍数 |
| `RESIZE_ENABLED` | `false` | 是否默认 resize |
| `AUTO_SPLIT_COMPOSITE` | `false` | 是否默认拆分复合图 |

## 数据流

```
PipelineInput(image_path, output_path)
        │
        ▼
  [processor]  resize + split（可选）
        │
        ▼
  [ocr]        RapidOCR on 3× upscaled image (→ bg check → Mimo 回退)
        │
        ▼
  [translator] batch LLM translate: Chinese → English
        │
        ▼
  [renderer]   erase Chinese + draw English (background detection, typesetting)
        │
        ▼
PipelineOutput(processed_path, regions, errors, sizes)
```

## 合并回主项目的方案

参见本模块测试稳定后：

1. 将 `test/image_translate_module/` 整体移动/复制到 `app/services/image_translate/`
2. 修改 `app/api/product.py`:
   ```python
   # 替换原来的 import
   from app.services.image_translate import ModuleConfig, PipelineInput, run_pipeline

   # 从主项目 Settings 构建 ModuleConfig
   config = ModuleConfig(
       translate_api_key=settings.deepseek_api_key,
       translate_base_url=settings.deepseek_base_url,
       translate_model=settings.deepseek_model,
       mimo_api_key=settings.mimo_api_key,
       mimo_base_url=settings.mimo_base_url,
       mimo_model=settings.mimo_model,
       mock_translate_when_no_key=settings.mock_llm_when_no_key,
       mock_vision_when_no_key=settings.mock_vision_when_no_key,
       translate_image_text=settings.translate_image_text,
       auto_split_composite=settings.auto_split_composite_images,
       resize_enabled=True,
   )
   ```
3. 删除旧的 `image_service.py`, `image_text_service.py`, `image_split_service.py` 中已迁移的函数
4. 旧的 `deepseek_service.py` 中的 `translate_texts_to_english_with_deepseek` 也可以删除
5. 原项目的 `MimoVisionService` 中的 `detect_chinese_text_regions_with_mimo` 和 `recognize_text_in_region` 在新模块中已内置
