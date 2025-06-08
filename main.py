#!/usr/bin/env python3
"""
ML-Based Real-Time E-Healthcare Monitoring System
Project Charter Implementation

This system implements multiple ML models for healthcare monitoring:
- Binary classification (Normal/Abnormal)
- Multi-class condition detection
- Risk scoring
- Time-series forecasting
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HealthcareDataGenerator:
    """Generate synthetic healthcare monitoring data"""
    
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        np.random.seed(42)
    
    def generate_normal_data(self, n_normal):
        """Generate normal health parameter data"""
        # Normal ranges
        heart_rate = np.random.normal(75, 10, n_normal)  # 65-85 normal
        spo2 = np.random.normal(98, 1.5, n_normal)       # 95-100 normal
        temperature = np.random.normal(37, 0.5, n_normal) # 36.5-37.5 normal
        
        # Ensure values are within realistic bounds
        heart_rate = np.clip(heart_rate, 60, 100)
        spo2 = np.clip(spo2, 95, 100)
        temperature = np.clip(temperature, 36.0, 38.0)
        
        return heart_rate, spo2, temperature
    
    def generate_abnormal_data(self, n_abnormal):
        """Generate abnormal health parameter data with specific conditions"""
        conditions = []
        heart_rates = []
        spo2_values = []
        temperatures = []
        
        # Split abnormal data into different conditions
        n_per_condition = n_abnormal // 4
        
        # 1. Tachycardia (high heart rate)
        hr_tachy = np.random.normal(110, 15, n_per_condition)
        spo2_tachy = np.random.normal(97, 2, n_per_condition)
        temp_tachy = np.random.normal(37.5, 0.8, n_per_condition)
        conditions.extend(['Tachycardia'] * n_per_condition)
        
        # 2. Hypoxia (low SpO2)
        hr_hypoxia = np.random.normal(85, 12, n_per_condition)
        spo2_hypoxia = np.random.normal(88, 3, n_per_condition)
        temp_hypoxia = np.random.normal(37, 0.6, n_per_condition)
        conditions.extend(['Hypoxia'] * n_per_condition)
        
        # 3. Fever
        hr_fever = np.random.normal(95, 10, n_per_condition)
        spo2_fever = np.random.normal(96, 2, n_per_condition)
        temp_fever = np.random.normal(39.5, 1, n_per_condition)
        conditions.extend(['Fever'] * n_per_condition)
        
        # 4. Bradycardia (low heart rate)
        hr_brady = np.random.normal(45, 8, n_abnormal - 3*n_per_condition)
        spo2_brady = np.random.normal(96, 2, n_abnormal - 3*n_per_condition)
        temp_brady = np.random.normal(36.8, 0.5, n_abnormal - 3*n_per_condition)
        conditions.extend(['Bradycardia'] * (n_abnormal - 3*n_per_condition))
        
        # Combine all abnormal data
        heart_rates = np.concatenate([hr_tachy, hr_hypoxia, hr_fever, hr_brady])
        spo2_values = np.concatenate([spo2_tachy, spo2_hypoxia, spo2_fever, spo2_brady])
        temperatures = np.concatenate([temp_tachy, temp_hypoxia, temp_fever, temp_brady])
        
        # Ensure realistic bounds
        heart_rates = np.clip(heart_rates, 30, 180)
        spo2_values = np.clip(spo2_values, 70, 100)
        temperatures = np.clip(temperatures, 35.0, 42.0)
        
        return heart_rates, spo2_values, temperatures, conditions
    
    def generate_dataset(self):
        """Generate complete dataset with features"""
        n_normal = int(self.n_samples * 0.7)  # 70% normal
        n_abnormal = self.n_samples - n_normal
        
        # Generate normal data
        hr_normal, spo2_normal, temp_normal = self.generate_normal_data(n_normal)
        
        # Generate abnormal data
        hr_abnormal, spo2_abnormal, temp_abnormal, conditions = self.generate_abnormal_data(n_abnormal)
        
        # Combine datasets
        heart_rate = np.concatenate([hr_normal, hr_abnormal])
        spo2 = np.concatenate([spo2_normal, spo2_abnormal])
        temperature = np.concatenate([temp_normal, temp_abnormal])
        
        # Create labels
        binary_labels = ['Normal'] * n_normal + ['Abnormal'] * n_abnormal
        condition_labels = ['Normal'] * n_normal + conditions
        
        # Create derived features
        hrv = np.random.normal(50, 15, self.n_samples)  # Heart Rate Variability
        temp_trend = np.random.normal(0, 0.2, self.n_samples)  # Temperature trend
        
        # Calculate risk scores (0-100)
        risk_scores = self.calculate_risk_scores(heart_rate, spo2, temperature)
        
        # Create DataFrame
        df = pd.DataFrame({
            'heart_rate': heart_rate,
            'spo2': spo2,
            'temperature': temperature,
            'hrv': hrv,
            'temp_trend': temp_trend,
            'binary_label': binary_labels,
            'condition': condition_labels,
            'risk_score': risk_scores
        })
        
        return df
    
    def calculate_risk_scores(self, heart_rate, spo2, temperature):
        """Calculate risk scores based on vital signs"""
        risk_scores = np.zeros(len(heart_rate))
        
        for i in range(len(heart_rate)):
            score = 0
            
            # Heart rate scoring
            if heart_rate[i] < 50 or heart_rate[i] > 120:
                score += 40
            elif heart_rate[i] < 60 or heart_rate[i] > 100:
                score += 20
            
            # SpO2 scoring
            if spo2[i] < 90:
                score += 35
            elif spo2[i] < 95:
                score += 15
            
            # Temperature scoring
            if temperature[i] > 39 or temperature[i] < 36:
                score += 25
            elif temperature[i] > 38 or temperature[i] < 36.5:
                score += 10
            
            risk_scores[i] = min(score, 100)
        
        return risk_scores

class HealthcareMLModels:
    """ML Models for healthcare monitoring"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.metrics = {}
        self.model_save_path = "/home/ap3x/Projects/HeathModel/saved_models/"
        
        # Create save directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def prepare_data(self, df):
        """Prepare data for ML models"""
        # Features for ML
        feature_cols = ['heart_rate', 'spo2', 'temperature', 'hrv', 'temp_trend']
        X = df[feature_cols].values
        
        # Binary classification target
        y_binary = (df['binary_label'] == 'Abnormal').astype(int)
        
        # Multi-class classification target
        y_multiclass = self.label_encoder.fit_transform(df['condition'])
        
        # Risk score regression target
        y_risk = df['risk_score'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_binary, y_multiclass, y_risk, feature_cols
    
    def train_binary_classifier(self, X_train, y_train, X_test, y_test):
        """Train ensemble classifier for binary classification"""
        print("Training Enhanced Binary Classifier (Ensemble)...")
        
        # Train multiple models
        models = {
            'logistic': LogisticRegression(random_state=42, C=0.1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42, C=1.0, gamma='scale')
        }
        
        predictions = {}
        best_accuracy = 0
        best_model = None
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            predictions[name] = y_pred
            
            print(f"{name.replace('_', ' ').title()} Accuracy: {accuracy:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        
        # Ensemble prediction (majority voting)
        ensemble_pred = []
        for i in range(len(y_test)):
            votes = [predictions[name][i] for name in models.keys()]
            ensemble_pred.append(1 if sum(votes) >= 2 else 0)
        
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        print(f"Ensemble Accuracy: {ensemble_accuracy:.3f}")
        
        # Use ensemble if better, otherwise use best individual model
        if ensemble_accuracy > best_accuracy:
            self.models['binary_classifier'] = models  # Store all models for ensemble
            self.models['ensemble_mode'] = True
            final_accuracy = ensemble_accuracy
            final_pred = ensemble_pred
        else:
            self.models['binary_classifier'] = best_model
            self.models['ensemble_mode'] = False
            final_accuracy = best_accuracy
            final_pred = best_model.predict(X_test)
        
        self.metrics['binary_accuracy'] = final_accuracy
        
        print(f"\nFinal Binary Classification Accuracy: {final_accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, final_pred, target_names=['Normal', 'Abnormal']))
        
        return self.models['binary_classifier'], final_pred
    
    def train_multiclass_classifier(self, X_train, y_train, X_test, y_test):
        """Train Random Forest for multi-class condition detection with hyperparameter tuning"""
        print("\nTraining Multi-class Classifier (Tuned Random Forest)...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['multiclass_classifier'] = model
        self.metrics['multiclass_accuracy'] = accuracy
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Multi-class Classification Accuracy: {accuracy:.3f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        return model, y_pred, feature_importance
    
    def train_risk_predictor(self, X_train, y_train, X_test, y_test):
        """Train neural network for risk score prediction"""
        print("\nTraining Risk Score Predictor (Neural Network)...")
        
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        
        # Convert continuous risk scores to risk categories for classification
        risk_categories_train = self.categorize_risk(y_train)
        risk_categories_test = self.categorize_risk(y_test)
        
        model.fit(X_train, risk_categories_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(risk_categories_test, y_pred)
        
        self.models['risk_predictor'] = model
        self.metrics['risk_accuracy'] = accuracy
        
        print(f"Risk Prediction Accuracy: {accuracy:.3f}")
        
        return model, y_pred
    
    def categorize_risk(self, risk_scores):
        """Convert continuous risk scores to categories"""
        categories = []
        for score in risk_scores:
            if score < 25:
                categories.append('Low')
            elif score < 50:
                categories.append('Medium')
            elif score < 75:
                categories.append('High')
            else:
                categories.append('Critical')
        return categories
    
    def create_lstm_model(self, sequence_length=10, n_features=5):
        """Create LSTM model for time-series forecasting"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')  # Binary output
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def save_models(self):
        """Save all trained models and preprocessing objects"""
        if not self.models:
            print("No models to save. Please train models first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual models
        for model_name, model in self.models.items():
            if model_name == 'ensemble_mode':
                continue
                
            model_filename = f"{model_name}_{timestamp}.pkl"
            model_path = os.path.join(self.model_save_path, model_filename)
            
            if model_name == 'binary_classifier' and isinstance(model, dict):
                # Handle ensemble models
                joblib.dump(model, model_path)
            else:
                joblib.dump(model, model_path)
            
            print(f"Saved {model_name} to {model_path}")
        
        # Save preprocessing objects
        scaler_path = os.path.join(self.model_save_path, f"scaler_{timestamp}.pkl")
        encoder_path = os.path.join(self.model_save_path, f"label_encoder_{timestamp}.pkl")
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save metrics and metadata
        metadata = {
            "timestamp": timestamp,
            "metrics": self.metrics,
            "ensemble_mode": self.models.get('ensemble_mode', False),
            "model_files": {
                "scaler": f"scaler_{timestamp}.pkl",
                "label_encoder": f"label_encoder_{timestamp}.pkl"
            }
        }
        
        for model_name in self.models.keys():
            if model_name != 'ensemble_mode':
                metadata["model_files"][model_name] = f"{model_name}_{timestamp}.pkl"
        
        metadata_path = os.path.join(self.model_save_path, f"model_metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model metadata saved to {metadata_path}")
        return timestamp
    
    def load_models(self, timestamp=None):
        """Load trained models and preprocessing objects"""
        if timestamp is None:
            # Find the latest timestamp
            metadata_files = [f for f in os.listdir(self.model_save_path) if f.startswith('model_metadata_')]
            if not metadata_files:
                print("No saved models found.")
                return False
            
            latest_file = sorted(metadata_files)[-1]
            timestamp = latest_file.replace('model_metadata_', '').replace('.json', '')
        
        metadata_path = os.path.join(self.model_save_path, f"model_metadata_{timestamp}.json")
        
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found: {metadata_path}")
            return False
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load preprocessing objects
        scaler_path = os.path.join(self.model_save_path, metadata["model_files"]["scaler"])
        encoder_path = os.path.join(self.model_save_path, metadata["model_files"]["label_encoder"])
        
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        
        # Load models
        for model_name, filename in metadata["model_files"].items():
            if model_name in ["scaler", "label_encoder"]:
                continue
                
            model_path = os.path.join(self.model_save_path, filename)
            self.models[model_name] = joblib.load(model_path)
        
        self.models['ensemble_mode'] = metadata.get('ensemble_mode', False)
        self.metrics = metadata.get('metrics', {})
        
        print(f"Successfully loaded models from timestamp: {timestamp}")
        return True

class HealthcareMonitoringSystem:
    """Main healthcare monitoring system"""
    
    def __init__(self):
        self.data_generator = HealthcareDataGenerator(n_samples=10000)
        self.ml_models = HealthcareMLModels()
        self.data = None
        self.is_trained = False
    
    def generate_data(self):
        """Generate synthetic healthcare data"""
        print("Generating synthetic healthcare monitoring data...")
        self.data = self.data_generator.generate_dataset()
        print(f"Generated {len(self.data)} samples")
        print(f"Data distribution:")
        print(self.data['binary_label'].value_counts())
        print(f"\nCondition distribution:")
        print(self.data['condition'].value_counts())
        
    def visualize_data(self):
        """Create visualizations of the healthcare data"""
        if self.data is None:
            print("No data available. Please generate data first.")
            return
        
        plt.figure(figsize=(15, 12))
        
        # 1. Distribution of vital signs by condition
        plt.subplot(2, 3, 1)
        sns.boxplot(data=self.data, x='binary_label', y='heart_rate')
        plt.title('Heart Rate Distribution')
        
        plt.subplot(2, 3, 2)
        sns.boxplot(data=self.data, x='binary_label', y='spo2')
        plt.title('SpO2 Distribution')
        
        plt.subplot(2, 3, 3)
        sns.boxplot(data=self.data, x='binary_label', y='temperature')
        plt.title('Temperature Distribution')
        
        # 2. Risk score distribution
        plt.subplot(2, 3, 4)
        plt.hist(self.data['risk_score'], bins=30, alpha=0.7, color='red')
        plt.title('Risk Score Distribution')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')
        
        # 3. Correlation heatmap
        plt.subplot(2, 3, 5)
        correlation_data = self.data[['heart_rate', 'spo2', 'temperature', 'hrv', 'risk_score']]
        sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation')
        
        # 4. Condition distribution
        plt.subplot(2, 3, 6)
        condition_counts = self.data['condition'].value_counts()
        plt.pie(condition_counts.values, labels=condition_counts.index, autopct='%1.1f%%')
        plt.title('Health Condition Distribution')
        
        plt.tight_layout()
        plt.savefig('/home/ap3x/Projects/HeathModel/healthcare_data_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Data visualization saved as 'healthcare_data_analysis.png'")
    
    def train_models(self):
        """Train all ML models"""
        if self.data is None:
            print("No data available. Please generate data first.")
            return
        
        print("Preparing data for ML models...")
        X, y_binary, y_multiclass, y_risk, feature_cols = self.ml_models.prepare_data(self.data)
        
        # Split data
        X_train, X_test, y_binary_train, y_binary_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        _, _, y_multi_train, y_multi_test = train_test_split(
            X, y_multiclass, test_size=0.2, random_state=42, stratify=y_multiclass
        )
        
        _, _, y_risk_train, y_risk_test = train_test_split(
            X, y_risk, test_size=0.2, random_state=42
        )
        
        print("="*60)
        
        # Train models
        binary_model, binary_pred = self.ml_models.train_binary_classifier(
            X_train, y_binary_train, X_test, y_binary_test
        )
        
        multi_model, multi_pred, feature_importance = self.ml_models.train_multiclass_classifier(
            X_train, y_multi_train, X_test, y_multi_test
        )
        
        risk_model, risk_pred = self.ml_models.train_risk_predictor(
            X_train, y_risk_train, X_test, y_risk_test
        )
        
        self.is_trained = True
        
        # Display feature importance
        print("\nFeature Importance (Random Forest):")
        for feature, importance in zip(feature_cols, feature_importance):
            print(f"{feature}: {importance:.3f}")
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Binary Classification Accuracy: {self.ml_models.metrics['binary_accuracy']:.1%}")
        print(f"Multi-class Classification Accuracy: {self.ml_models.metrics['multiclass_accuracy']:.1%}")
        print(f"Risk Prediction Accuracy: {self.ml_models.metrics['risk_accuracy']:.1%}")
        
        # Compare with target performance from project charter
        print("\nTarget vs Actual Performance:")
        target_accuracy = 0.94  # From project charter
        actual_accuracy = self.ml_models.metrics['binary_accuracy']
        print(f"Heart Rate Anomaly Detection: Target 94%, Actual {actual_accuracy:.1%}")
        
        if actual_accuracy >= target_accuracy:
            print("✅ Target performance achieved!")
        else:
            print("⚠️ Performance below target - consider model tuning")
        
        # Save the trained models
        print("\n💾 Saving trained models...")
        timestamp = self.ml_models.save_models()
        print(f"✅ Models saved with timestamp: {timestamp}")
    
    def predict_health_status(self, heart_rate, spo2, temperature, hrv=50, temp_trend=0):
        """Make predictions for new health data"""
        if not self.is_trained:
            print("Models not trained yet. Please train models first.")
            return None
        
        # Prepare input data
        input_data = np.array([[heart_rate, spo2, temperature, hrv, temp_trend]])
        input_scaled = self.ml_models.scaler.transform(input_data)
        
        # Make binary prediction (handle ensemble mode)
        if self.ml_models.models.get('ensemble_mode', False):
            # Ensemble prediction
            votes = []
            probs = []
            for model in self.ml_models.models['binary_classifier'].values():
                vote = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0]
                votes.append(vote)
                probs.append(prob)
            
            binary_pred = 1 if sum(votes) >= 2 else 0
            # Average probabilities
            avg_prob = np.mean(probs, axis=0)
            binary_prob = avg_prob
        else:
            # Single model prediction
            binary_pred = self.ml_models.models['binary_classifier'].predict(input_scaled)[0]
            binary_prob = self.ml_models.models['binary_classifier'].predict_proba(input_scaled)[0]
        
        multiclass_pred = self.ml_models.models['multiclass_classifier'].predict(input_scaled)[0]
        condition = self.ml_models.label_encoder.inverse_transform([multiclass_pred])[0]
        
        risk_pred = self.ml_models.models['risk_predictor'].predict(input_scaled)[0]
        
        # Calculate risk score
        risk_score = self.data_generator.calculate_risk_scores(
            np.array([heart_rate]), np.array([spo2]), np.array([temperature])
        )[0]
        
        # Determine action
        if risk_score > 75:
            action = "Send Alert"
        elif risk_score > 50:
            action = "Suggest Rest"
        else:
            action = "No Action"
        
        result = {
            'binary_prediction': 'Abnormal' if binary_pred == 1 else 'Normal',
            'confidence': max(binary_prob),
            'condition': condition,
            'risk_category': risk_pred,
            'risk_score': risk_score,
            'recommended_action': action,
            'vital_signs': {
                'heart_rate': heart_rate,
                'spo2': spo2,
                'temperature': temperature,
                'hrv': hrv,
                'temp_trend': temp_trend
            }
        }
        
        return result
    
    def display_prediction(self, result):
        """Display prediction results in a formatted way"""
        if result is None:
            return
        
        print("\n" + "="*50)
        print("HEALTH MONITORING PREDICTION")
        print("="*50)
        print(f"Status: {result['binary_prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Detected Condition: {result['condition']}")
        print(f"Risk Category: {result['risk_category']}")
        print(f"Risk Score: {result['risk_score']:.0f}/100")
        print(f"Recommended Action: {result['recommended_action']}")
        
        print("\nVital Signs:")
        print(f"  Heart Rate: {result['vital_signs']['heart_rate']:.0f} bpm")
        print(f"  SpO2: {result['vital_signs']['spo2']:.1f}%")
        print(f"  Temperature: {result['vital_signs']['temperature']:.1f}°C")
        print("="*50)
    
    def run_demo(self):
        """Run a complete demo of the healthcare monitoring system"""
        print("🏥 ML-Based Real-Time E-Healthcare Monitoring System")
        print("=" * 70)
        
        # Generate and visualize data
        self.generate_data()
        
        # Train models
        self.train_models()
        
        # Visualize data
        self.visualize_data()
        
        # Demo predictions
        print("\n🔬 TESTING PREDICTIONS")
        print("=" * 40)
        
        test_cases = [
            {"name": "Normal Patient", "hr": 75, "spo2": 98, "temp": 37.0},
            {"name": "Tachycardia Patient", "hr": 120, "spo2": 97, "temp": 37.2},
            {"name": "Hypoxia Patient", "hr": 85, "spo2": 88, "temp": 37.1},
            {"name": "Fever Patient", "hr": 95, "spo2": 96, "temp": 39.5},
            {"name": "Critical Patient", "hr": 140, "spo2": 85, "temp": 40.0}
        ]
        
        for test_case in test_cases:
            print(f"\n🧑‍⚕️ Testing: {test_case['name']}")
            result = self.predict_health_status(
                test_case['hr'], test_case['spo2'], test_case['temp']
            )
            self.display_prediction(result)
        
        print("\n✅ Healthcare Monitoring System Demo Complete!")
        print(f"📊 Data visualization saved as 'healthcare_data_analysis.png'")

def main():
    """Main function to run the healthcare monitoring system"""
    # Create and run the healthcare monitoring system
    system = HealthcareMonitoringSystem()
    system.run_demo()

if __name__ == "__main__":
    main()