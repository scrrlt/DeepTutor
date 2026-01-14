#!/bin/bash
# DeepTutor Frontend Deployment Script for Vercel
# Usage: ./deploy-frontend.sh [api-base-url]

set -e

# Configuration
API_BASE_URL=${1:-"https://deeptutor-backend-abc123-uc.a.run.app"}

echo "🚀 Deploying DeepTutor Frontend to Vercel"
echo "=========================================="
echo "API Base URL: ${API_BASE_URL}"
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Please install it:"
    echo "npm install -g vercel"
    exit 1
fi

# Check if authenticated with Vercel
if ! vercel whoami &> /dev/null; then
    echo "❌ Not authenticated with Vercel. Please run: vercel login"
    exit 1
fi

# Navigate to web directory
cd web

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build the application
echo "🏗️ Building application..."
npm run build

# Deploy to Vercel with environment variables
echo "🚀 Deploying to Vercel..."
DEPLOY_OUTPUT=$(vercel --prod \
    --env NEXT_PUBLIC_API_BASE="${API_BASE_URL}" \
    --yes)

# Get the deployment URL from the output
DEPLOYMENT_URL=$(echo "$DEPLOY_OUTPUT" | grep -o 'https://[^ ]*\.vercel\.app')

echo ""
echo "✅ Frontend deployment completed successfully!"
echo "🌐 Frontend URL: ${DEPLOYMENT_URL}"
echo ""
echo "🎉 Your DeepTutor application is now live!"
echo "   Frontend: ${DEPLOYMENT_URL}"
echo "   Backend: ${API_BASE_URL}"
echo ""
echo "🔧 Useful commands:"
echo "   Check Vercel logs: vercel logs"
echo "   Redeploy: ./deploy-frontend.sh ${API_BASE_URL}"
