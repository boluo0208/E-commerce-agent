FROM python:3.11-slim

WORKDIR /app

# System deps: OpenCV libs, Pillow, fonts for text rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libfontconfig1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app/ ./app/
COPY frontend/dist/ ./frontend/dist/

# Config template — copy .env.example to .env if not mounted.
COPY .env.example ./

# Create data directories
RUN mkdir -p uploads outputs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
