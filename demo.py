#!/usr/bin/env python3
"""
Healthcare Monitoring System Demo
Comprehensive demonstration of the ML-based healthcare monitoring system
"""

from main import HealthcareMonitoringSystem
from monitoring_utils import RealTimeMonitor, PerformanceAnalyzer, AlertSystem
import time

def run_comprehensive_demo():
    """Run a comprehensive demonstration of the healthcare monitoring system"""
    
    print("🏥 ML-Based Real-Time E-Healthcare Monitoring System")
    print("🚀 COMPREHENSIVE DEMO")
    print("=" * 70)
    
    # Initialize the main system
    print("\n📊 Phase 1: Data Generation and Model Training")
    print("-" * 50)
    
    system = HealthcareMonitoringSystem()
    
    # Generate data and train models
    system.generate_data()
    system.train_models()
    system.visualize_data()
    
    # Initialize utilities
    monitor = RealTimeMonitor(system)
    analyzer = PerformanceAnalyzer(system)
    alert_system = AlertSystem()
    
    print("\n📈 Phase 2: Performance Analysis")
    print("-" * 50)
    
    # Generate and display performance report
    analyzer.print_performance_summary()
    analyzer.save_performance_report()
    
    print("\n🔬 Phase 3: Static Prediction Testing")
    print("-" * 50)
    
    # Test various scenarios
    test_scenarios = [
        {"name": "Healthy Individual", "hr": 72, "spo2": 98, "temp": 36.8},
        {"name": "Mild Tachycardia", "hr": 105, "spo2": 97, "temp": 37.1},
        {"name": "Hypoxia Patient", "hr": 88, "spo2": 89, "temp": 37.0},
        {"name": "Fever Case", "hr": 92, "spo2": 96, "temp": 38.8},
        {"name": "Critical Emergency", "hr": 135, "spo2": 82, "temp": 40.2}
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Testing: {scenario['name']}")
        result = system.predict_health_status(
            scenario['hr'], scenario['spo2'], scenario['temp']
        )
        
        # Send alert if needed
        if result['risk_score'] > 50:
            alert_system.send_alert(scenario, result)
        
        system.display_prediction(result)
    
    print(f"\n📝 Alert Summary: {alert_system.get_alert_summary()}")
    alert_system.save_alert_log()
    
    print("\n⏱️ Phase 4: Real-Time Monitoring Simulation")
    print("-" * 50)
    
    # Simulate different patient types
    patient_types = ["normal", "deteriorating", "critical"]
    
    for patient_type in patient_types:
        print(f"\n🩺 Simulating {patient_type} patient monitoring (30 seconds)...")
        monitor.start_monitoring(patient_type=patient_type, duration=30)
        time.sleep(1)  # Brief pause between simulations
    
    print("\n✅ Phase 5: System Summary")
    print("-" * 50)
    
    # Final system summary
    print("🎯 Key Features Demonstrated:")
    print("  ✅ Synthetic healthcare data generation")
    print("  ✅ Multi-model ML training (Binary, Multi-class, Risk prediction)")
    print("  ✅ Ensemble learning for improved accuracy")
    print("  ✅ Real-time monitoring simulation")
    print("  ✅ Automated alert system")
    print("  ✅ Performance analysis and reporting")
    print("  ✅ Data visualization")
    
    print("\n📊 Files Generated:")
    print("  📈 healthcare_data_analysis.png - Data visualization")
    print("  📊 performance_report.json - Model performance metrics")
    print("  📝 alert_log.json - Alert history")
    
    print(f"\n🏆 Model Performance Summary:")
    metrics = system.ml_models.metrics
    print(f"  Binary Classification: {metrics.get('binary_accuracy', 0):.1%}")
    print(f"  Multi-class Classification: {metrics.get('multiclass_accuracy', 0):.1%}")
    print(f"  Risk Prediction: {metrics.get('risk_accuracy', 0):.1%}")
    
    print("\n🎉 Healthcare Monitoring System Demo Complete!")
    print("=" * 70)

def run_quick_demo():
    """Run a quick demonstration focusing on core functionality"""
    
    print("🏥 Quick Healthcare Monitoring Demo")
    print("=" * 50)
    
    # Quick system initialization
    system = HealthcareMonitoringSystem()
    system.generate_data()
    system.train_models()
    
    # Quick performance check
    analyzer = PerformanceAnalyzer(system)
    analyzer.print_performance_summary()
    
    # Test one critical case
    print("\n🚨 Testing Critical Emergency Case:")
    result = system.predict_health_status(140, 85, 40.0)
    system.display_prediction(result)
    
    print("\n✅ Quick demo complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_demo()
    else:
        run_comprehensive_demo()
