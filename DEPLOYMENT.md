# DeepTutor Deployment Guide

This guide will help you deploy DeepTutor to **Vercel** (frontend) and **Google Cloud Run** (backend).

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Vercel Account** (free tier works)
3. **Google Cloud SDK** installed and authenticated
4. **Vercel CLI** installed
5. **Docker** installed

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd DeepTutor

# Copy environment file and configure
cp .env.example .env
# Edit .env with your API keys
```

### 2. Authenticate Services

```bash
# Authenticate with Google Cloud
gcloud auth login

# Authenticate with Vercel
vercel login
```

### 3. Deploy Everything

```bash
# One-command deployment
./deploy.sh your-google-project-id
```

## Manual Deployment

### Backend Deployment (Google Cloud Run)

```bash
# Make script executable
chmod +x deploy-backend.sh

# Deploy backend
./deploy-backend.sh your-google-project-id
```

This will:
- Build Docker image
- Push to Google Container Registry
- Deploy to Cloud Run
- Return the service URL

### Frontend Deployment (Vercel)

```bash
# Make script executable
chmod +x deploy-frontend.sh

# Deploy frontend (replace with your backend URL)
./deploy-frontend.sh https://your-backend-url.a.run.app
```

## Environment Variables

### Backend (.env)

```bash
# Required
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key
EMBEDDING_API_KEY=your-embedding-key

# Optional
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### Frontend (Vercel Environment)

Set in Vercel dashboard or via CLI:
```
NEXT_PUBLIC_API_BASE=https://your-backend-url.a.run.app
```

## Architecture

```
User Browser
    ↓
Vercel (Next.js Frontend)
    ↓ API calls
Google Cloud Run (FastAPI Backend)
    ↓
LLM Providers (OpenAI, Anthropic, etc.)
```

## Cost Estimation

- **Vercel**: Free tier (~$0/month)
- **Google Cloud Run**: ~$0-5/month for low traffic
- **LLM APIs**: Pay per usage

## Troubleshooting

### Backend Issues

```bash
# Check Cloud Run logs
gcloud logs read --service=deeptutor-backend --region=us-central1

# Check service status
gcloud run services describe deeptutor-backend --region=us-central1
```

### Frontend Issues

```bash
# Check Vercel logs
vercel logs

# Redeploy
vercel --prod
```

### Common Issues

1. **CORS errors**: Check CORS_ORIGINS in backend .env
2. **API connection failed**: Verify NEXT_PUBLIC_API_BASE URL
3. **Build failures**: Check that all dependencies are in requirements.txt

## Security Notes

- Never commit .env files
- Use environment variables for all secrets
- Rotate API keys regularly
- Enable Cloud Run authentication if needed

## Support

For issues, check:
1. Google Cloud Console logs
2. Vercel dashboard logs
3. Application logs in Cloud Run
