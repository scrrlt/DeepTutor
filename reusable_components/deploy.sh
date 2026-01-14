#!/bin/bash
# Deployment Script for CHIMERA Platform

set -e

PROJECT_ID="gen-lang-client-0460359034"
REGION="us-central1"

echo "🚀 Deploying CHIMERA Platform..."

# Deploy Gateway Service
echo "📦 Deploying Gateway Service..."
gcloud run deploy chimera-gateway \
  --source services/gateway \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1

# Deploy Embeddings Service
echo "📦 Deploying Embeddings Service..."
gcloud run deploy chimera-embeddings \
  --source services/research \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 2

# Deploy Data Service
echo "📦 Deploying Data Service..."
gcloud run deploy chimera-data \
  --source services/data \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1

echo "✅ Deployment complete!"
echo "Gateway URL: https://chimera-gateway-<hash>-uc.a.run.app"