#!/bin/bash
# DeepTutor Full Deployment Script
# Usage: ./deploy.sh [project-id]

set -e

# Configuration
PROJECT_ID=${1:-"your-project-id"}

echo "🚀 DeepTutor Full Deployment"
echo "============================"
echo "Project ID: ${PROJECT_ID}"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install Google Cloud SDK:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if vercel is installed
if ! command -v vercel &> /dev/null; then
    print_error "Vercel CLI not found. Please install it:"
    echo "  npm install -g vercel"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker:"
    echo "  https://docs.docker.com/get-docker/"
    exit 1
fi

print_status "Prerequisites check passed"

# Make scripts executable
chmod +x deploy-backend.sh
chmod +x deploy-frontend.sh

# Deploy backend first
echo ""
echo "🔧 Step 1: Deploying Backend to Google Cloud Run"
echo "=================================================="
if ./deploy-backend.sh "${PROJECT_ID}"; then
    print_status "Backend deployment completed"

    # Extract the service URL from the output
    SERVICE_URL=$(gcloud run services describe deeptutor-backend --region=us-central1 --format="value(status.url)" 2>/dev/null || echo "")

    if [ -n "$SERVICE_URL" ]; then
        print_status "Backend URL: $SERVICE_URL"
    else
        print_warning "Could not retrieve service URL automatically"
        echo "Please check your Google Cloud Console for the service URL"
        read -p "Enter the Cloud Run service URL: " SERVICE_URL
    fi
else
    print_error "Backend deployment failed"
    exit 1
fi

# Deploy frontend
echo ""
echo "🔧 Step 2: Deploying Frontend to Vercel"
echo "========================================"
if ./deploy-frontend.sh "${SERVICE_URL}"; then
    print_status "Frontend deployment completed"
else
    print_error "Frontend deployment failed"
    exit 1
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo "===================================="
echo ""
echo "🌐 Your DeepTutor application is now live!"
echo ""
echo "📋 Summary:"
echo "   • Backend API: ${SERVICE_URL}"
echo "   • Frontend: Check Vercel dashboard for URL"
echo ""
echo "🔧 Useful commands:"
echo "   • Check backend logs: gcloud logs read --service=deeptutor-backend --region=us-central1"
echo "   • Check frontend logs: vercel logs"
echo "   • Redeploy backend: ./deploy-backend.sh ${PROJECT_ID}"
echo "   • Redeploy frontend: ./deploy-frontend.sh ${SERVICE_URL}"
echo ""
echo "📚 Documentation:"
echo "   • Google Cloud Run: https://cloud.google.com/run/docs"
echo "   • Vercel: https://vercel.com/docs"
echo ""
print_status "Happy deploying! 🚀"
