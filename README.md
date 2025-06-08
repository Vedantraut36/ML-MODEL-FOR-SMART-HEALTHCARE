# ML Model for Smart Healthcare Monitoring

A comprehensive machine learning-based healthcare monitoring system with real-time vital signs analysis, predictive modeling, and serverless deployment capabilities.

## 🏥 Project Overview

This project implements an intelligent healthcare monitoring system that uses machine learning to analyze vital signs and predict health conditions in real-time. The system features a FastAPI backend, React frontend, and is optimized for serverless deployment on Vercel.

## ✨ Key Features

- **Real-time Health Monitoring**: Continuous monitoring of vital signs (heart rate, SpO2, temperature)
- **ML-Powered Predictions**: Multiple machine learning models for health status prediction
- **Risk Assessment**: Automated risk scoring and categorization
- **Alert System**: Real-time alerts for critical health conditions
- **Serverless Deployment**: Optimized for Vercel serverless platform
- **Modern Web Interface**: React-based dashboard with real-time visualizations
- **RESTful API**: Comprehensive API for healthcare data integration

## 🏗️ System Architecture

### Backend (FastAPI)
- **Main API** (`api.py`): Full-featured ML healthcare monitoring system
- **Vercel API** (`vercel_api.py`): Serverless-optimized version for cloud deployment
- **Monitoring Utils** (`monitoring_utils.py`): Real-time monitoring and performance analysis
- **Core System** (`main.py`): Healthcare monitoring system core logic

### Frontend (React)
- Modern, responsive web interface
- Real-time vital signs dashboard
- Interactive charts and visualizations
- Patient management interface

### Machine Learning Models
- **Binary Classification**: Normal vs Abnormal health status
- **Multi-class Classification**: Specific condition detection
- **Risk Prediction**: Risk level assessment
- **Real-time Analytics**: Performance monitoring and optimization

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/Vedantraut36/ML-MODEL-FOR-SMART-HEALTHCARE.git
cd ML-MODEL-FOR-SMART-HEALTHCARE

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
python api.py
# or
./start_backend.sh
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the React development server
npm start
```

### Vercel Deployment
```bash
# Install Vercel CLI (if not already installed)
npm install -g vercel

# Deploy to Vercel
./deploy_vercel.sh
# or
vercel --prod
```

## 📊 API Endpoints

### Health Check
- `GET /` - Root endpoint health check
- `GET /health` - Detailed system status

### Predictions
- `POST /predict` - Single patient health prediction
- `POST /batch-predict` - Batch predictions for multiple patients

### Analytics
- `GET /metrics` - Model performance metrics
- `GET /alerts` - Recent health alerts

### Development
- `GET /test/generate-sample-data` - Generate sample data for testing

## 🔧 Configuration

### Vercel Configuration (`vercel.json`)
```json
{
  "version": 2,
  "builds": [
    {
      "src": "vercel_api.py",
      "use": "@vercel/python",
      "config": {
        "maxLambdaSize": "50mb"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "vercel_api.py"
    }
  ],
  "functions": {
    "vercel_api.py": {
      "maxDuration": 30
    }
  }
}
```

## 📈 Model Performance

### Current Metrics
- **Binary Classification**: 98.6% accuracy
- **Multi-class Classification**: 75.6% accuracy
- **Risk Prediction**: 82.1% accuracy

### Health Conditions Detected
- Normal health status
- Tachycardia (elevated heart rate)
- Bradycardia (low heart rate)
- Hypoxia (low blood oxygen)
- Fever (elevated temperature)

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Scikit-learn**: Machine learning library
- **Pandas & NumPy**: Data manipulation and analysis
- **Uvicorn**: ASGI server for FastAPI

### Frontend
- **React**: JavaScript library for building user interfaces
- **Material-UI**: React UI framework
- **Recharts**: Charting library for React
- **Axios**: HTTP client for API communication

### Deployment
- **Vercel**: Serverless deployment platform
- **GitHub**: Version control and CI/CD

## 📁 Project Structure

```
├── api.py                      # Main FastAPI application
├── vercel_api.py              # Vercel-optimized API
├── main.py                    # Core healthcare monitoring system
├── monitoring_utils.py        # Monitoring and analytics utilities
├── requirements.txt           # Python dependencies
├── requirements-vercel.txt    # Vercel-specific dependencies
├── vercel.json               # Vercel deployment configuration
├── deploy_vercel.sh          # Deployment script
├── VERCEL_DEPLOYMENT.md      # Deployment documentation
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   └── index.js         # Entry point
│   └── package.json         # Node.js dependencies
└── saved_models/            # Trained ML models
```

## 🚨 Health Monitoring Features

### Vital Signs Monitoring
- **Heart Rate**: 30-200 BPM range with tachycardia/bradycardia detection
- **SpO2**: 70-100% range with hypoxia detection
- **Temperature**: 35-42°C range with fever detection
- **HRV**: Heart rate variability analysis
- **Temperature Trends**: Monitoring temperature changes over time

### Risk Assessment
- **Low Risk**: Normal vital signs, continue regular activities
- **Medium Risk**: Minor deviations, monitor closely
- **High Risk**: Significant abnormalities, contact healthcare provider
- **Critical Risk**: Severe abnormalities, immediate medical attention required

### Alert System
- Real-time notifications for critical conditions
- Risk score calculations based on multiple vital signs
- Automated recommendations for appropriate actions

## 📚 Documentation

- [Vercel Deployment Guide](VERCEL_DEPLOYMENT.md)
- [API Documentation](http://localhost:8000/docs) (when running locally)
- [Project Charter Report](ML_EHealthcare_Project_Charter_Report.pdf)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Vedant Raut** - *Initial work* - [Vedantraut36](https://github.com/Vedantraut36)

## 🙏 Acknowledgments

- Healthcare professionals for domain expertise
- Open source machine learning community
- FastAPI and React development teams

## 📞 Support

For questions, issues, or contributions, please:
1. Check the [Issues](https://github.com/Vedantraut36/ML-MODEL-FOR-SMART-HEALTHCARE/issues) page
2. Create a new issue if your question isn't already addressed
3. Contact the maintainers for urgent matters

---

**⚠️ Disclaimer**: This system is for educational and research purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.
