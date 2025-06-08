#!/usr/bin/env python3
"""
Real-time Healthcare Monitoring Utilities
Supporting tools for the ML healthcare monitoring system
"""

import json
import time
import numpy as np
from datetime import datetime
import threading
import queue

class RealTimeMonitor:
    """Simulate real-time health monitoring"""
    
    def __init__(self, healthcare_system):
        self.system = healthcare_system
        self.is_monitoring = False
        self.data_queue = queue.Queue()
        self.alert_threshold = 75
        
    def generate_patient_stream(self, patient_type="normal", duration_seconds=60):
        """Generate simulated patient data stream"""
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds and self.is_monitoring:
            if patient_type == "normal":
                hr = np.random.normal(75, 5)
                spo2 = np.random.normal(98, 1)
                temp = np.random.normal(37, 0.3)
            elif patient_type == "deteriorating":
                # Simulate gradual deterioration
                elapsed = time.time() - start_time
                hr = 75 + (elapsed / duration_seconds) * 40  # Rising heart rate
                spo2 = 98 - (elapsed / duration_seconds) * 10  # Falling SpO2
                temp = 37 + (elapsed / duration_seconds) * 2   # Rising temperature
            elif patient_type == "critical":
                hr = np.random.normal(130, 10)
                spo2 = np.random.normal(85, 3)
                temp = np.random.normal(39.5, 0.5)
            
            # Add some noise and ensure realistic bounds
            hr = np.clip(hr + np.random.normal(0, 2), 30, 180)
            spo2 = np.clip(spo2 + np.random.normal(0, 0.5), 70, 100)
            temp = np.clip(temp + np.random.normal(0, 0.1), 35, 42)
            
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'heart_rate': hr,
                'spo2': spo2,
                'temperature': temp,
                'patient_type': patient_type
            }
            
            self.data_queue.put(data_point)
            time.sleep(1)  # 1 Hz sampling rate
    
    def start_monitoring(self, patient_type="normal", duration=60):
        """Start real-time monitoring"""
        self.is_monitoring = True
        print(f"🔴 Starting real-time monitoring for {patient_type} patient...")
        
        # Start data generation in separate thread
        generator_thread = threading.Thread(
            target=self.generate_patient_stream,
            args=(patient_type, duration)
        )
        generator_thread.start()
        
        alerts_sent = 0
        
        while self.is_monitoring and generator_thread.is_alive():
            try:
                # Get data point from queue (timeout after 2 seconds)
                data_point = self.data_queue.get(timeout=2)
                
                # Make prediction
                result = self.system.predict_health_status(
                    data_point['heart_rate'],
                    data_point['spo2'],
                    data_point['temperature']
                )
                
                # Display monitoring info
                timestamp = data_point['timestamp'].split('T')[1][:8]
                print(f"[{timestamp}] HR:{data_point['heart_rate']:.0f} SpO2:{data_point['spo2']:.1f} Temp:{data_point['temperature']:.1f}°C "
                      f"→ {result['binary_prediction']} (Risk: {result['risk_score']:.0f})")
                
                # Check for alerts
                if result['risk_score'] > self.alert_threshold:
                    alerts_sent += 1
                    print(f"🚨 ALERT #{alerts_sent}: {result['condition']} detected! "
                          f"Action: {result['recommended_action']}")
                
            except queue.Empty:
                print("No data received, stopping monitoring...")
                break
        
        self.is_monitoring = False
        generator_thread.join()
        print(f"✅ Monitoring session complete. Alerts sent: {alerts_sent}")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False

class PerformanceAnalyzer:
    """Analyze model performance and generate reports"""
    
    def __init__(self, healthcare_system):
        self.system = healthcare_system
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.system.is_trained:
            print("Models not trained. Cannot generate performance report.")
            return
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_performance": {
                "binary_classification": {
                    "accuracy": self.system.ml_models.metrics.get('binary_accuracy', 0),
                    "target": 0.94,
                    "status": "✅ Target met" if self.system.ml_models.metrics.get('binary_accuracy', 0) >= 0.94 else "⚠️ Below target"
                },
                "multiclass_classification": {
                    "accuracy": self.system.ml_models.metrics.get('multiclass_accuracy', 0),
                    "target": 0.85,
                    "status": "✅ Target met" if self.system.ml_models.metrics.get('multiclass_accuracy', 0) >= 0.85 else "⚠️ Below target"
                },
                "risk_prediction": {
                    "accuracy": self.system.ml_models.metrics.get('risk_accuracy', 0),
                    "target": 0.80,
                    "status": "✅ Target met" if self.system.ml_models.metrics.get('risk_accuracy', 0) >= 0.80 else "⚠️ Below target"
                }
            },
            "data_statistics": {
                "total_samples": len(self.system.data) if self.system.data is not None else 0,
                "normal_samples": len(self.system.data[self.system.data['binary_label'] == 'Normal']) if self.system.data is not None else 0,
                "abnormal_samples": len(self.system.data[self.system.data['binary_label'] == 'Abnormal']) if self.system.data is not None else 0
            }
        }
        
        return report
    
    def save_performance_report(self, filename="performance_report.json"):
        """Save performance report to file"""
        report = self.generate_performance_report()
        filepath = f"/home/ap3x/Projects/HeathModel/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Performance report saved to {filepath}")
        return filepath
    
    def print_performance_summary(self):
        """Print a formatted performance summary"""
        report = self.generate_performance_report()
        
        print("\n" + "="*60)
        print("HEALTHCARE MONITORING SYSTEM PERFORMANCE REPORT")
        print("="*60)
        print(f"Generated: {report['timestamp']}")
        print("\nModel Performance:")
        
        for model_name, metrics in report['model_performance'].items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Accuracy: {metrics['accuracy']:.1%}")
            print(f"  Target: {metrics['target']:.1%}")
            print(f"  Status: {metrics['status']}")
        
        print(f"\nData Statistics:")
        print(f"  Total Samples: {report['data_statistics']['total_samples']:,}")
        print(f"  Normal Samples: {report['data_statistics']['normal_samples']:,}")
        print(f"  Abnormal Samples: {report['data_statistics']['abnormal_samples']:,}")
        print("="*60)

class AlertSystem:
    """Healthcare alert management system"""
    
    def __init__(self):
        self.alert_log = []
        self.alert_levels = {
            'LOW': {'threshold': 25, 'color': '🟢'},
            'MEDIUM': {'threshold': 50, 'color': '🟡'},
            'HIGH': {'threshold': 75, 'color': '🟠'},
            'CRITICAL': {'threshold': 100, 'color': '🔴'}
        }
    
    def determine_alert_level(self, risk_score):
        """Determine alert level based on risk score"""
        if risk_score >= 75:
            return 'CRITICAL'
        elif risk_score >= 50:
            return 'HIGH'
        elif risk_score >= 25:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def send_alert(self, patient_data, prediction_result):
        """Send alert based on prediction result"""
        alert_level = self.determine_alert_level(prediction_result['risk_score'])
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': 'DEMO_PATIENT',
            'alert_level': alert_level,
            'risk_score': prediction_result['risk_score'],
            'condition': prediction_result['condition'],
            'vital_signs': prediction_result['vital_signs'],
            'recommended_action': prediction_result['recommended_action']
        }
        
        self.alert_log.append(alert)
        
        # Display alert
        color = self.alert_levels[alert_level]['color']
        print(f"{color} {alert_level} ALERT: {alert['condition']} "
              f"(Risk: {alert['risk_score']:.0f}) - {alert['recommended_action']}")
        
        return alert
    
    def get_alert_summary(self):
        """Get summary of all alerts"""
        if not self.alert_log:
            return "No alerts recorded"
        
        summary = {}
        for alert in self.alert_log:
            level = alert['alert_level']
            summary[level] = summary.get(level, 0) + 1
        
        return summary
    
    def save_alert_log(self, filename="alert_log.json"):
        """Save alert log to file"""
        filepath = f"/home/ap3x/Projects/HeathModel/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(self.alert_log, f, indent=2)
        
        print(f"📝 Alert log saved to {filepath}")
        return filepath
