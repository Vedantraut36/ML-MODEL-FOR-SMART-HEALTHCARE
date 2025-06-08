#!/usr/bin/env python3
"""
FastAPI Backend for ML Healthcare Monitoring System
Provides REST API endpoints for React frontend integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import uvicorn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import logging
from main import HealthcareMonitoringSystem, HealthcareMLModels, HealthcareDataGenerator
from monitoring_utils import RealTimeMonitor, PerformanceAnalyzer, AlertSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Monitoring API",
    description="ML-Based Real-Time E-Healthcare Monitoring System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],  # React default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the healthcare system
healthcare_system = None
alert_system = AlertSystem()
performance_analyzer = None

# Pydantic models for request/response validation
class VitalSigns(BaseModel):
    heart_rate: float
    spo2: float
    temperature: float
    hrv: Optional[float] = 50.0
    temp_trend: Optional[float] = 0.0
    patient_id: Optional[str] = "default_patient"
    timestamp: Optional[str] = None
    
    @validator('heart_rate')
    def validate_heart_rate(cls, v):
        if not 30 <= v <= 200:
            raise ValueError('Heart rate must be between 30 and 200 bpm')
        return v
    
    @validator('spo2')
    def validate_spo2(cls, v):
        if not 70 <= v <= 100:
            raise ValueError('SpO2 must be between 70 and 100%')
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 35.0 <= v <= 42.0:
            raise ValueError('Temperature must be between 35.0 and 42.0°C')
        return v

class PredictionResponse(BaseModel):
    binary_prediction: str
    confidence: float
    condition: str
    risk_category: str
    risk_score: float
    recommended_action: str
    vital_signs: Dict[str, float]
    timestamp: str
    patient_id: str

class ModelMetrics(BaseModel):
    binary_accuracy: float
    multiclass_accuracy: float
    risk_accuracy: float
    target_performance: Dict[str, float]
    performance_status: Dict[str, str]

class SystemStatus(BaseModel):
    is_initialized: bool
    is_trained: bool
    model_timestamp: Optional[str]
    total_predictions: int
    total_alerts: int
    uptime: str

class AlertResponse(BaseModel):
    alert_id: str
    timestamp: str
    patient_id: str
    alert_level: str
    risk_score: float
    condition: str
    vital_signs: Dict[str, float]
    recommended_action: str

# Global state tracking
app_state = {
    "total_predictions": 0,
    "total_alerts": 0,
    "start_time": datetime.now(),
    "is_initialized": False,
    "model_timestamp": None
}

@app.on_event("startup")
async def startup_event():
    """Initialize the healthcare monitoring system on startup"""
    global healthcare_system, performance_analyzer
    
    logger.info("Starting Healthcare Monitoring API...")
    
    try:
        # Initialize the system
        healthcare_system = HealthcareMonitoringSystem()
        
        # Try to load existing models
        if healthcare_system.ml_models.load_models():
            healthcare_system.is_trained = True
            app_state["is_initialized"] = True
            logger.info("Loaded existing trained models")
        else:
            logger.info("No existing models found. System will need training.")
        
        performance_analyzer = PerformanceAnalyzer(healthcare_system)
        logger.info("Healthcare Monitoring API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")

@app.get("/", tags=["Health Check"])
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Healthcare Monitoring API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "healthy"
    }

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Detailed health check endpoint"""
    uptime = datetime.now() - app_state["start_time"]
    
    return SystemStatus(
        is_initialized=app_state["is_initialized"],
        is_trained=healthcare_system.is_trained if healthcare_system else False,
        model_timestamp=app_state["model_timestamp"],
        total_predictions=app_state["total_predictions"],
        total_alerts=app_state["total_alerts"],
        uptime=str(uptime).split('.')[0]  # Remove microseconds
    )

@app.post("/initialize", tags=["System Management"])
async def initialize_system(background_tasks: BackgroundTasks):
    """Initialize and train the healthcare monitoring system"""
    global healthcare_system, performance_analyzer
    
    if not healthcare_system:
        raise HTTPException(status_code=500, detail="System not available")
    
    try:
        # Generate data and train models in background
        background_tasks.add_task(train_models_background)
        
        return {
            "message": "System initialization started",
            "status": "training_in_progress",
            "estimated_time": "2-3 minutes"
        }
    
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

async def train_models_background():
    """Background task to train models"""
    global healthcare_system, app_state
    
    try:
        logger.info("Starting model training...")
        
        # Generate data and train models
        healthcare_system.generate_data()
        healthcare_system.train_models()
        
        app_state["is_initialized"] = True
        app_state["model_timestamp"] = datetime.now().isoformat()
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Background training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_health_status(vital_signs: VitalSigns):
    """Make health status prediction based on vital signs"""
    
    if not healthcare_system or not healthcare_system.is_trained:
        raise HTTPException(
            status_code=400, 
            detail="System not initialized or trained. Please call /initialize first."
        )
    
    try:
        # Make prediction
        result = healthcare_system.predict_health_status(
            vital_signs.heart_rate,
            vital_signs.spo2,
            vital_signs.temperature,
            vital_signs.hrv,
            vital_signs.temp_trend
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Update global counters
        app_state["total_predictions"] += 1
        
        # Check if alert should be sent
        if result['risk_score'] > 50:  # Alert threshold
            alert = alert_system.send_alert(vital_signs.dict(), result)
            app_state["total_alerts"] += 1
        
        # Prepare response
        response = PredictionResponse(
            binary_prediction=result['binary_prediction'],
            confidence=result['confidence'],
            condition=result['condition'],
            risk_category=result['risk_category'],
            risk_score=result['risk_score'],
            recommended_action=result['recommended_action'],
            vital_signs=result['vital_signs'],
            timestamp=vital_signs.timestamp or datetime.now().isoformat(),
            patient_id=vital_signs.patient_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics", response_model=ModelMetrics, tags=["Analytics"])
async def get_model_metrics():
    """Get model performance metrics"""
    
    if not healthcare_system or not healthcare_system.is_trained:
        raise HTTPException(
            status_code=400,
            detail="System not trained. Please call /initialize first."
        )
    
    try:
        metrics = healthcare_system.ml_models.metrics
        
        target_performance = {
            "binary_classification": 0.94,
            "multiclass_classification": 0.85,
            "risk_prediction": 0.80
        }
        
        performance_status = {
            "binary_classification": "✅ Target met" if metrics.get('binary_accuracy', 0) >= 0.94 else "⚠️ Below target",
            "multiclass_classification": "✅ Target met" if metrics.get('multiclass_accuracy', 0) >= 0.85 else "⚠️ Below target",
            "risk_prediction": "✅ Target met" if metrics.get('risk_accuracy', 0) >= 0.80 else "⚠️ Below target"
        }
        
        return ModelMetrics(
            binary_accuracy=metrics.get('binary_accuracy', 0),
            multiclass_accuracy=metrics.get('multiclass_accuracy', 0),
            risk_accuracy=metrics.get('risk_accuracy', 0),
            target_performance=target_performance,
            performance_status=performance_status
        )
        
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/alerts", tags=["Alerts"])
async def get_alerts(limit: int = 50):
    """Get recent alerts"""
    
    try:
        alerts = alert_system.alert_log[-limit:] if alert_system.alert_log else []
        
        alert_responses = []
        for i, alert in enumerate(alerts):
            alert_responses.append(AlertResponse(
                alert_id=f"alert_{i}_{alert['timestamp']}",
                timestamp=alert['timestamp'],
                patient_id=alert['patient_id'],
                alert_level=alert['alert_level'],
                risk_score=alert['risk_score'],
                condition=alert['condition'],
                vital_signs=alert['vital_signs'],
                recommended_action=alert['recommended_action']
            ))
        
        return {
            "alerts": alert_responses,
            "total_alerts": len(alerts),
            "alert_summary": alert_system.get_alert_summary()
        }
        
    except Exception as e:
        logger.error(f"Alerts error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@app.get("/analytics/dashboard", tags=["Analytics"])
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    
    if not healthcare_system:
        raise HTTPException(status_code=400, detail="System not available")
    
    try:
        # Basic stats
        uptime = datetime.now() - app_state["start_time"]
        
        dashboard_data = {
            "system_status": {
                "is_initialized": app_state["is_initialized"],
                "is_trained": healthcare_system.is_trained,
                "uptime": str(uptime).split('.')[0],
                "total_predictions": app_state["total_predictions"],
                "total_alerts": app_state["total_alerts"]
            },
            "model_performance": {},
            "recent_activity": {
                "predictions_last_hour": app_state["total_predictions"],  # Simplified for demo
                "alerts_last_hour": app_state["total_alerts"]
            },
            "alert_summary": alert_system.get_alert_summary() if alert_system.alert_log else {}
        }
        
        # Add model metrics if available
        if healthcare_system.is_trained:
            metrics = healthcare_system.ml_models.metrics
            dashboard_data["model_performance"] = {
                "binary_accuracy": metrics.get('binary_accuracy', 0),
                "multiclass_accuracy": metrics.get('multiclass_accuracy', 0),
                "risk_accuracy": metrics.get('risk_accuracy', 0)
            }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

@app.post("/batch-predict", tags=["Predictions"])
async def batch_predict(vital_signs_list: List[VitalSigns]):
    """Make predictions for multiple patients"""
    
    if not healthcare_system or not healthcare_system.is_trained:
        raise HTTPException(
            status_code=400,
            detail="System not trained. Please call /initialize first."
        )
    
    if len(vital_signs_list) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 100 predictions per request."
        )
    
    try:
        predictions = []
        
        for vital_signs in vital_signs_list:
            result = healthcare_system.predict_health_status(
                vital_signs.heart_rate,
                vital_signs.spo2,
                vital_signs.temperature,
                vital_signs.hrv,
                vital_signs.temp_trend
            )
            
            if result:
                app_state["total_predictions"] += 1
                
                # Check for alerts
                if result['risk_score'] > 50:
                    alert_system.send_alert(vital_signs.dict(), result)
                    app_state["total_alerts"] += 1
                
                predictions.append(PredictionResponse(
                    binary_prediction=result['binary_prediction'],
                    confidence=result['confidence'],
                    condition=result['condition'],
                    risk_category=result['risk_category'],
                    risk_score=result['risk_score'],
                    recommended_action=result['recommended_action'],
                    vital_signs=result['vital_signs'],
                    timestamp=vital_signs.timestamp or datetime.now().isoformat(),
                    patient_id=vital_signs.patient_id
                ))
        
        return {
            "predictions": predictions,
            "total_processed": len(predictions),
            "processing_time": f"{len(predictions) * 0.1:.2f}s"  # Estimated
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/export/data", tags=["Data Export"])
async def export_data():
    """Export training data visualization"""
    
    image_path = "/home/ap3x/Projects/HeathModel/healthcare_data_analysis.png"
    
    if os.path.exists(image_path):
        return FileResponse(
            image_path,
            media_type="image/png",
            filename="healthcare_data_analysis.png"
        )
    else:
        raise HTTPException(status_code=404, detail="Data visualization not found")

@app.get("/export/alerts", tags=["Data Export"])
async def export_alerts():
    """Export alerts as JSON"""
    
    try:
        alerts_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_alerts": len(alert_system.alert_log),
            "alert_summary": alert_system.get_alert_summary(),
            "alerts": alert_system.alert_log
        }
        
        return alerts_data
        
    except Exception as e:
        logger.error(f"Alert export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export alerts: {str(e)}")

# Test endpoints for development
@app.post("/test/generate-sample-data", tags=["Development"])
async def generate_sample_data():
    """Generate sample vital signs data for testing"""
    
    sample_data = [
        VitalSigns(heart_rate=75, spo2=98, temperature=37.0, patient_id="patient_001"),
        VitalSigns(heart_rate=120, spo2=97, temperature=37.2, patient_id="patient_002"),
        VitalSigns(heart_rate=85, spo2=88, temperature=37.1, patient_id="patient_003"),
        VitalSigns(heart_rate=95, spo2=96, temperature=39.5, patient_id="patient_004"),
        VitalSigns(heart_rate=140, spo2=85, temperature=40.0, patient_id="patient_005")
    ]
    
    return {
        "sample_data": [data.dict() for data in sample_data],
        "note": "Use this data with the /predict or /batch-predict endpoints"
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
