#!/bin/bash
# Quick deployment script for Google Cloud Run
# Usage: ./QUICK_DEPLOY.sh [PROJECT_ID]

set -e

PROJECT_ID=${1:-$(gcloud config get-value project 2>/dev/null)}

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: Project ID not provided"
    echo "Usage: ./QUICK_DEPLOY.sh [PROJECT_ID]"
    echo "Or set default: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "🚀 Deploying to Google Cloud Run"
echo "Project ID: $PROJECT_ID"
echo "================================"
echo ""

# Build and submit
echo "📦 Building Docker image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/hack-the-track-backend:latest

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy hack-the-track-backend \
  --image gcr.io/$PROJECT_ID/hack-the-track-backend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    SERVICE_URL=$(gcloud run services describe hack-the-track-backend --region us-central1 --format 'value(status.url)' 2>/dev/null)
    echo "🌐 Service URL: $SERVICE_URL"
    echo ""
    echo "Test health endpoint:"
    echo "curl $SERVICE_URL/api/health"
else
    echo "❌ Deployment failed!"
    exit 1
fi

