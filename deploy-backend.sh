#!/bin/bash
# DeepTutor Backend Deployment Script for Google Cloud Run
# Usage: ./deploy-backend.sh [project-id]

set -e

# Configuration
PROJECT_ID=${1:-"your-project-id"}

# Validate PROJECT_ID
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Error: Please provide a valid Google Cloud Project ID."
    echo "Usage: $0 <project-id>"
    echo "Example: $0 my-gcp-project"
    exit 1
fi

SERVICE_NAME="deeptutor-backend"
REGION="us-central1"
REPOSITORY="deeptutor-repo"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}"

echo "🚀 Deploying DeepTutor Backend to Google Cloud Run"
echo "=================================================="
echo "Project ID: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}"
echo ""

# Check if gcloud is authenticated
echo "🔍 Checking gcloud authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run: gcloud auth login"
    exit 1
fi

# Set the project
echo "🔧 Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo "📦 Creating Artifact Registry repository..."
gcloud artifacts repositories create ${REPOSITORY} \
    --repository-format=docker \
    --location=${REGION} \
    --description="Docker repository for DeepTutor" \
    --async || echo "Repository may already exist, continuing..."

# Configure Docker to use Artifact Registry
echo "🔧 Configuring Docker for Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push Docker image
echo "🏗️ Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME}:latest .

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8001 \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --timeout 300 \
    --concurrency 80 \
    --set-env-vars "ENVIRONMENT=production,API_KEY=REPLACE_ME,OPENAI_API_KEY=REPLACE_ME,PROVIDER=REPLACE_ME,LLM_MODEL=REPLACE_ME,EMBEDDING_MODEL=REPLACE_ME,VECTOR_DB_URL=REPLACE_ME"

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
echo "🔧 To check logs: gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}\" --limit=50"
