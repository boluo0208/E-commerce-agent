#!/bin/bash
set -e

# ============================================================
#  E-commerce Agent — server deployment script (Ubuntu)
#  Usage:  chmod +x deploy.sh && ./deploy.sh
# ============================================================

PROJECT_DIR="$HOME/ecommerce-agent"
REPO_URL="https://github.com/boluo0208/E-commerce-agent.git"
IMAGE_NAME="ecommerce-agent"
CONTAINER_NAME="ecommerce-agent"

echo "==> Pulling latest code..."
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    git pull origin master
else
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

echo "==> Building frontend..."
cd "$PROJECT_DIR/frontend"
npm install
npm run build

echo "==> Building Docker image..."
cd "$PROJECT_DIR"
docker build -t "$IMAGE_NAME:latest" .

echo "==> Stopping old container (if any)..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "==> Ensuring .env exists..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "    ⚠  Created .env from .env.example — please edit it with real API keys!"
    echo "       vi $PROJECT_DIR/.env"
    echo "       Then re-run: ./deploy.sh"
    exit 1
fi

echo "==> Starting container..."
docker run -d \
    --name "$CONTAINER_NAME" \
    -p 8000:8000 \
    -v "$PROJECT_DIR/.env:/app/.env:ro" \
    -v "$PROJECT_DIR/uploads:/app/uploads" \
    -v "$PROJECT_DIR/outputs:/app/outputs" \
    --restart unless-stopped \
    "$IMAGE_NAME:latest"

echo ""
echo "✔  Deploy complete!  Check:  curl http://localhost:8000/health"
echo "   Logs: docker logs -f $CONTAINER_NAME"
