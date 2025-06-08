# Healthcare Monitoring API - Vercel Deployment

This directory contains the Vercel deployment configuration for the ML-Based Healthcare Monitoring System API.

## 🚀 Quick Deployment

### Prerequisites
1. Node.js and npm installed
2. Vercel account (free tier available)
3. Vercel CLI installed globally

### Deployment Steps

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy using the script**:
   ```bash
   ./deploy_vercel.sh
   ```

   Or manually:
   ```bash
   vercel --prod
   ```

4. **Update Frontend Configuration**:
   After deployment, update your React frontend's `API_BASE_URL` in `frontend/src/App.js`:
   ```javascript
   const API_BASE_URL = 'https://your-vercel-url.vercel.app';
   ```

## 📁 Files Overview

- `vercel.json` - Vercel configuration
- `vercel_api.py` - Serverless-optimized API
- `requirements-vercel.txt` - Python dependencies for Vercel
- `deploy_vercel.sh` - Deployment script

## 🔧 Configuration Details

### vercel.json
- Configures Python runtime
- Sets up routing
- Optimizes for serverless deployment
- Limits function execution time to 30 seconds

### vercel_api.py
- Simplified version of the main API
- Uses rule-based predictions for faster cold starts
- Reduced dependencies for smaller bundle size
- Optimized for serverless environment

## 🌐 API Endpoints

After deployment, your API will be available at:
- `GET /` - Health check
- `GET /health` - System status
- `POST /predict` - Health prediction
- `POST /batch-predict` - Batch predictions
- `GET /metrics` - Model performance
- `GET /alerts` - Recent alerts
- `GET /docs` - Interactive API documentation

## 🧪 Testing the Deployed API

1. **Health Check**:
   ```bash
   curl https://your-vercel-url.vercel.app/
   ```

2. **Make a Prediction**:
   ```bash
   curl -X POST https://your-vercel-url.vercel.app/predict \
     -H "Content-Type: application/json" \
     -d '{"heart_rate": 120, "spo2": 97, "temperature": 37.2}'
   ```

3. **View API Documentation**:
   Visit: `https://your-vercel-url.vercel.app/docs`

## 🔄 Updating the Deployment

To update your deployment:
```bash
vercel --prod
```

## 🐛 Troubleshooting

### Common Issues:

1. **Cold Start Delays**: First request may be slower due to serverless cold starts
2. **Memory Limits**: Large ML models may hit Vercel's memory limits
3. **Timeout Issues**: Complex predictions may exceed 30-second function limit

### Solutions:

1. **Use Rule-Based Fallbacks**: The Vercel API uses simplified rule-based predictions
2. **Optimize Bundle Size**: Minimal dependencies in `requirements-vercel.txt`
3. **Cache Responses**: Consider adding caching for frequently requested predictions

## 📊 Performance Considerations

- **Cold Start**: ~2-5 seconds for first request
- **Warm Response**: ~100-500ms for subsequent requests
- **Concurrent Requests**: Vercel handles scaling automatically
- **Memory Usage**: ~128MB per function instance

## 🔒 Security

- CORS enabled for frontend integration
- Input validation using Pydantic models
- No sensitive data stored in serverless functions
- Environment variables for configuration

## 💰 Cost Estimation

Vercel's free tier includes:
- 100GB bandwidth per month
- 100 serverless function invocations per day
- Unlimited static deployments

For production use, consider Vercel Pro for higher limits.

## 🆘 Support

If you encounter issues:
1. Check Vercel function logs: `vercel logs`
2. Review the deployment: `vercel inspect`
3. Test locally: `vercel dev`

## 🔗 Related Files

- Main API: `api.py`
- Frontend: `frontend/`
- ML Models: `main.py`
- Configuration: `config.json`
