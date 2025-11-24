# Google Cloud Run Deployment Guide

This guide explains how to deploy the backend to Google Cloud Run with compressed model files.

## Prerequisites

1. **Google Cloud SDK** installed and configured
2. **Docker** installed (for local testing)
3. **gcloud CLI** authenticated: `gcloud auth login`
4. **Project ID** set: `gcloud config set project YOUR_PROJECT_ID`

## Step 1: Prepare Models for Deployment

The model files are compressed to reduce size from ~51MB to ~14MB for GitHub storage.

### Option A: Use Pre-compressed Models (Recommended)

The compressed file `models/models_compressed.tar.gz` is already created and will be automatically decompressed on startup.

### Option B: Compress Models Manually

If you need to recreate the compressed file:

```bash
cd models
./compress_models.sh
```

## Step 2: Build and Push Docker Image

### Build the Docker image:

```bash
# Build the image
docker build -t gcr.io/YOUR_PROJECT_ID/hack-the-track-backend:latest .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/hack-the-track-backend:latest
```

Or use Cloud Build:

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/hack-the-track-backend:latest
```

## Step 3: Deploy to Cloud Run

### Basic Deployment:

```bash
gcloud run deploy hack-the-track-backend \
  --image gcr.io/YOUR_PROJECT_ID/hack-the-track-backend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10
```

### Advanced Deployment with Environment Variables:

```bash
gcloud run deploy hack-the-track-backend \
  --image gcr.io/YOUR_PROJECT_ID/hack-the-track-backend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars "PORT=8080"
```

### Important Configuration:

- **Memory**: Minimum 2Gi (models need memory to load)
- **CPU**: 2 CPUs recommended for faster startup
- **Timeout**: 300s (5 minutes) to allow model decompression and loading
- **Max Instances**: Adjust based on expected traffic

## Step 4: Verify Deployment

After deployment, Cloud Run will provide a URL like:
```
https://hack-the-track-backend-xxxxx-uc.a.run.app
```

Test the health endpoint:
```bash
curl https://YOUR_SERVICE_URL/api/health
```

Test model loading:
```bash
curl https://YOUR_SERVICE_URL/api/predictive/tracks
```

## How Model Decompression Works

The `start.sh` script automatically:
1. Checks if `models/models_compressed.tar.gz` exists
2. Checks if model files (.pkl, .parquet) are already extracted
3. If compressed file exists but models aren't extracted, decompresses them
4. Starts the FastAPI server

This happens automatically on every Cloud Run instance startup.

## Troubleshooting

### Models Not Loading

1. **Check logs**: `gcloud run logs read hack-the-track-backend --limit 50`
2. **Verify compressed file exists**: Check Docker image contains `models/models_compressed.tar.gz`
3. **Check memory**: Models need at least 2Gi RAM

### Startup Timeout

- Increase Cloud Run timeout: `--timeout 600` (10 minutes)
- Models decompress in ~10-30 seconds
- Model loading takes additional 30-60 seconds

### Out of Memory

- Increase memory allocation: `--memory 4Gi`
- Check logs for memory errors

## Local Testing

Test the Docker image locally before deploying:

```bash
# Build
docker build -t hack-the-track-backend:local .

# Run
docker run -p 8080:8080 \
  -e PORT=8080 \
  hack-the-track-backend:local

# Test
curl http://localhost:8080/api/health
```

## Continuous Deployment

### Using Cloud Build (Recommended):

Create `cloudbuild.yaml`:

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/hack-the-track-backend:$SHORT_SHA', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/hack-the-track-backend:$SHORT_SHA']
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'hack-the-track-backend'
      - '--image=gcr.io/$PROJECT_ID/hack-the-track-backend:$SHORT_SHA'
      - '--region=us-central1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--memory=2Gi'
      - '--cpu=2'
      - '--timeout=300'
```

Trigger build:
```bash
gcloud builds submit --config cloudbuild.yaml
```

## Cost Optimization

- **Min Instances**: Set to 0 for cost savings (cold starts acceptable)
- **Max Instances**: Limit based on expected traffic
- **Memory**: Start with 2Gi, increase if needed
- **CPU**: 1 CPU works but slower startup

## Security

- Use `--no-allow-unauthenticated` for private APIs
- Set up IAM roles for service access
- Use secrets for API keys: `gcloud run services update --update-secrets`

## Monitoring

View logs:
```bash
gcloud run logs read hack-the-track-backend --limit 100
```

Monitor in Cloud Console:
- Go to Cloud Run → hack-the-track-backend
- View metrics, logs, and revisions

