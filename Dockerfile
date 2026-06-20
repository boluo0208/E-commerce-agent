FROM python:3.10-slim

WORKDIR /app

# Use Aliyun mirror for apt
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# System deps: OpenCV, Pillow, fonts for text rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libfontconfig1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer — frontend changes won't trigger reinstall)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# Application code (copied last for layer caching)
COPY app/ ./app/
COPY frontend/dist/ ./frontend/dist/

# Config template
COPY .env.example ./

RUN mkdir -p uploads outputs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
