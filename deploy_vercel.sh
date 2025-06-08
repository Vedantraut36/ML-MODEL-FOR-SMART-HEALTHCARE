#!/bin/bash

# Healthcare Monitoring API - Vercel Deployment Script
echo "🚀 Deploying Healthcare Monitoring API to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Login to Vercel (if not already logged in)
echo "🔐 Checking Vercel authentication..."
vercel whoami || vercel login

# Deploy to Vercel
echo "📦 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo ""
echo "📝 Next steps:"
echo "1. Your API will be available at the Vercel URL provided"
echo "2. Update your React frontend's API_BASE_URL to point to the Vercel URL"
echo "3. Test the API endpoints using the /docs endpoint"
echo ""
echo "🔗 Available endpoints:"
echo "  - GET  /         - Health check"
echo "  - GET  /health   - System status"
echo "  - POST /predict  - Make health predictions"
echo "  - POST /batch-predict - Batch predictions"
echo "  - GET  /metrics  - Model performance"
echo "  - GET  /alerts   - Recent alerts"
echo "  - GET  /docs     - API documentation"
