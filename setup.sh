#!/bin/bash

# Setup script for Nomic embedding service on your VPS
echo "🚀 Setting up Nomic Embedding Service..."

# Create embedding directory
mkdir -p ~/embedding-service
cd ~/embedding-service

# Create models directory
mkdir -p models

echo "📁 Files should be copied to ~/embedding-service/"
echo "   - app.py"
echo "   - requirements.txt"
echo "   - Dockerfile"
echo "   - docker-compose.yml"
echo ""

# Check if files exist
files=("app.py" "requirements.txt" "Dockerfile" "docker-compose.yml")
missing_files=()

for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ Missing files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Please copy all required files first!"
    exit 1
fi

echo "✅ All files found!"
echo ""

# Stop any existing container
echo "🛑 Stopping existing containers..."
docker stop nomic-embedding-service 2>/dev/null || true
docker rm nomic-embedding-service 2>/dev/null || true

# Build the image
echo "🔨 Building Docker image..."
docker build -t nomic-embedding-api .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

# Run the container directly
echo "🚀 Starting Nomic embedding service..."
docker run -d \
  --name nomic-embedding-service \
  --restart unless-stopped \
  -p 5000:5000 \
  -v "$(pwd)/models:/app/models" \
  -e FLASK_ENV=production \
  -e PYTHONUNBUFFERED=1 \
  nomic-embedding-api

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container!"
    exit 1
fi

echo "⏳ Waiting for service to be ready..."
sleep 30

# Check service status
echo "📊 Service Status:"
docker ps --filter "name=nomic-embedding-service"

echo ""
echo "📝 Container logs:"
docker logs nomic-embedding-service --tail 10

echo ""
echo "🧪 Testing the API..."

# Test health endpoint
echo "Health check:"
curl -s http://localhost:5000/health | jq . || echo "Health check failed"

echo ""
echo "Embedding test:"
curl -X POST http://localhost:5000/embed \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world test with Nomic"}' \
     -s | jq '.dimensions, .model, .processing_time' || echo "Embedding test failed"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 Your Nomic API Endpoints:"
echo "   Health Check: GET  http://$(hostname -I | awk '{print $1}'):5000/health"
echo "   Single Embed: POST http://$(hostname -I | awk '{print $1}'):5000/embed"
echo "   Batch Embed:  POST http://$(hostname -I | awk '{print $1}'):5000/embed/batch"
echo ""
echo "🎯 Model Info:"
echo "   - Model: nomic-embed-text-v1"
echo "   - Dimensions: 768"
echo "   - Normalized embeddings: Yes"
echo ""
echo "📝 To monitor logs:"
echo "   docker logs -f nomic-embedding-service"
echo ""
echo "🔄 To restart:"
echo "   docker restart nomic-embedding-service"
echo ""
echo "🛑 To stop:"
echo "   docker stop nomic-embedding-service"
echo ""
echo "🗑️ To remove:"
echo "   docker stop nomic-embedding-service && docker rm nomic-embedding-service"
