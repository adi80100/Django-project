#!/usr/bin/env python3
"""
College Predictor ML Training Pipeline

This script trains a machine learning model to predict college admission probabilities
based on student percentile, category, branch preference, and exam type.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json
from typing import Dict, Any, List, Tuple

class CollegePredictorModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'admission_probability'
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess the college data"""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Create admission probability based on cutoffs and student percentile
        # This simulates real admission chances
        processed_data = []
        
        categories = ['General', 'OBC', 'SC', 'ST', 'EWS']
        
        for _, row in df.iterrows():
            college_data = row.to_dict()
            
            # Generate training samples for each category
            for category in categories:
                cutoff_col = f"{category.lower()}_cutoff"
                cutoff = row[cutoff_col]
                
                # Generate positive samples (above cutoff)
                for _ in range(15):  # 15 positive samples per college-category combo
                    percentile = np.random.uniform(cutoff + 0.1, 100)
                    sample = {
                        'student_percentile': percentile,
                        'category': category,
                        'preferred_branch': row['branch'],
                        'exam_type': row['exam_type'],
                        'college_name': row['college_name'],
                        'branch': row['branch'],
                        'state': row['state'],
                        'tier': row['tier'],
                        'fees_per_year': row['fees_per_year'],
                        'placement_percentage': row['placement_percentage'],
                        'admission_probability': 1 if percentile >= cutoff else 0
                    }
                    processed_data.append(sample)
                
                # Generate negative samples (below cutoff)
                for _ in range(5):  # 5 negative samples per college-category combo
                    percentile = np.random.uniform(max(0, cutoff - 10), cutoff - 0.1)
                    sample = {
                        'student_percentile': percentile,
                        'category': category,
                        'preferred_branch': row['branch'],
                        'exam_type': row['exam_type'],
                        'college_name': row['college_name'],
                        'branch': row['branch'],
                        'state': row['state'],
                        'tier': row['tier'],
                        'fees_per_year': row['fees_per_year'],
                        'placement_percentage': row['placement_percentage'],
                        'admission_probability': 0
                    }
                    processed_data.append(sample)
        
        return pd.DataFrame(processed_data)
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for training"""
        print("Preprocessing features...")
        
        # Select feature columns
        feature_cols = [
            'student_percentile', 'category', 'preferred_branch', 
            'exam_type', 'tier', 'fees_per_year', 'placement_percentage'
        ]
        
        # Create a copy for processing
        processed_df = df[feature_cols + [self.target_column]].copy()
        
        # Encode categorical variables
        categorical_cols = ['category', 'preferred_branch', 'exam_type']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col])
            else:
                processed_df[col] = self.label_encoders[col].transform(processed_df[col])
        
        self.feature_columns = feature_cols
        return processed_df
    
    def train(self, data_path: str) -> Dict[str, Any]:
        """Train the college prediction model"""
        # Load and preprocess data
        df = self.load_data(data_path)
        processed_df = self.preprocess_features(df)
        
        # Prepare features and target
        X = processed_df[self.feature_columns]
        y = processed_df[self.target_column]
        
        print(f"Training on {len(X)} samples with {len(self.feature_columns)} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_cols = ['student_percentile', 'tier', 'fees_per_year', 'placement_percentage']
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        # Train model
        print("Training Random Forest model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        print("\nFeature Importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_colleges(self, student_percentile: float, category: str, 
                        preferred_branch: str, exam_type: str) -> List[Dict[str, Any]]:
        """Predict top colleges for a student"""
        # Load original college data
        colleges_df = pd.read_csv('../data/colleges_data.csv')
        
        # Filter colleges by exam type and branch
        filtered_colleges = colleges_df[
            (colleges_df['exam_type'] == exam_type) & 
            (colleges_df['branch'] == preferred_branch)
        ].copy()
        
        predictions = []
        
        for _, college in filtered_colleges.iterrows():
            # Prepare features for prediction
            features = pd.DataFrame({
                'student_percentile': [student_percentile],
                'category': [category],
                'preferred_branch': [preferred_branch],
                'exam_type': [exam_type],
                'tier': [college['tier']],
                'fees_per_year': [college['fees_per_year']],
                'placement_percentage': [college['placement_percentage']]
            })
            
            # Encode categorical variables
            for col in ['category', 'preferred_branch', 'exam_type']:
                if col in self.label_encoders:
                    features[col] = self.label_encoders[col].transform(features[col])
            
            # Scale numerical features
            numerical_cols = ['student_percentile', 'tier', 'fees_per_year', 'placement_percentage']
            features[numerical_cols] = self.scaler.transform(features[numerical_cols])
            
            # Get prediction probability
            admission_prob = self.model.predict_proba(features)[0][1]  # Probability of admission
            
            predictions.append({
                'college_name': college['college_name'],
                'branch': college['branch'],
                'state': college['state'],
                'tier': int(college['tier']),
                'fees_per_year': int(college['fees_per_year']),
                'placement_percentage': int(college['placement_percentage']),
                'admission_probability': float(admission_prob),
                'cutoff_percentile': float(college[f"{category.lower()}_cutoff"])
            })
        
        # Sort by admission probability (descending)
        predictions.sort(key=lambda x: x['admission_probability'], reverse=True)
        
        return predictions
    
    def save_model(self, model_dir: str):
        """Save the trained model and preprocessors"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, os.path.join(model_dir, 'college_predictor_model.pkl'))
        
        # Save preprocessors
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        # Save feature columns
        with open(os.path.join(model_dir, 'feature_columns.json'), 'w') as f:
            json.dump(self.feature_columns, f)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir: str):
        """Load a trained model and preprocessors"""
        self.model = joblib.load(os.path.join(model_dir, 'college_predictor_model.pkl'))
        self.label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        with open(os.path.join(model_dir, 'feature_columns.json'), 'r') as f:
            self.feature_columns = json.load(f)
        
        print(f"Model loaded from {model_dir}")

def main():
    """Main training function"""
    print("=" * 50)
    print("College Predictor ML Training Pipeline")
    print("=" * 50)
    
    # Initialize model
    predictor = CollegePredictorModel()
    
    # Train model
    data_path = '../data/colleges_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    training_results = predictor.train(data_path)
    
    # Save model
    model_dir = '../models'
    predictor.save_model(model_dir)
    
    # Test prediction
    print("\n" + "=" * 50)
    print("Testing Model Prediction")
    print("=" * 50)
    
    test_predictions = predictor.predict_colleges(
        student_percentile=95.0,
        category='General',
        preferred_branch='Computer Science',
        exam_type='CET'
    )
    
    print(f"\nTop 5 college predictions for 95th percentile General category student:")
    for i, pred in enumerate(test_predictions[:5]):
        print(f"{i+1}. {pred['college_name']} - {pred['admission_probability']:.3f} probability")
    
    print(f"\nTraining completed successfully!")
    print(f"Model accuracy: {training_results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
