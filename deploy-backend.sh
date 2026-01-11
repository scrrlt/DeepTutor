#!/bin/bash
# DeepTutor Backend Deployment Script for Google Cloud Run
# Usage: ./deploy-backend.sh [project-id]

set -euo pipefail

# Configuration
PROJECT_ID="${1:-your-project-id}"
SERVICE_NAME="deeptutor-backend"
REGION="us-central1"
REPO_NAME="deeptutor"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/backend"

echo "🚀 Deploying DeepTutor Backend to Google Cloud Run"
echo "=================================================="
echo "Project ID: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}"
echo ""

# Check if gcloud is authenticated
echo "🔍 Checking gcloud authentication..."
if ! gcloud auth list --filter="status:ACTIVE" --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run: gcloud auth login"
    exit 1
fi

# Set the project
echo "🔧 Setting project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create Artifact Registry Docker repository if it doesn't exist
echo "📦 Ensuring Artifact Registry repository exists..."
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location="${REGION}" >/dev/null 2>&1; then
    gcloud artifacts repositories create "${REPO_NAME}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="DeepTutor Docker images"
fi

# Build and push Docker image
echo "🏗️ Building and pushing Docker image..."
gcloud builds submit --tag "${IMAGE_NAME}:latest" .

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}:latest" \
    --platform managed \
    --region "${REGION}" \
    --allow-unauthenticated \
    --port 8001 \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --timeout 300 \
    --concurrency 80 \
    --set-env-vars "ENVIRONMENT=production"

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo ""
echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "📝 Next steps:"
echo "1. Copy the service URL above"
echo "2. Set NEXT_PUBLIC_API_BASE in Vercel to: ${SERVICE_URL}"
echo "3. Run: ./deploy-frontend.sh"
echo ""
echo "🔧 To check logs: gcloud logs read --service=${SERVICE_NAME} --region=${REGION}"
