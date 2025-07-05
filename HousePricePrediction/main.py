#!/usr/bin/env python3
"""
House Price Prediction - Main Script
This script handles the complete ML pipeline for house price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings
import os
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        """Initialize the HousePricePredictor"""
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.best_model = None
        self.feature_columns = None
        
    def load_data(self, train_path='train.csv', test_path='test.csv', 
                  sample_submission_path='sample_submission.csv'):
        """Load the training, test, and sample submission data"""
        print("Loading data...")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct full paths
        train_path = os.path.join(script_dir, train_path)
        test_path = os.path.join(script_dir, test_path)
        sample_submission_path = os.path.join(script_dir, sample_submission_path)
        
        try:
            self.train_data = pd.read_csv(train_path)
            self.test_data = pd.read_csv(test_path)
            self.sample_submission = pd.read_csv(sample_submission_path)
            
            print(f"✅ Training data loaded: {self.train_data.shape}")
            print(f"✅ Test data loaded: {self.test_data.shape}")
            print(f"✅ Sample submission loaded: {self.sample_submission.shape}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("Please ensure all CSV files are in the HousePricePrediction directory:")
            print("- train.csv")
            print("- test.csv") 
            print("- sample_submission.csv")
            return False
    
    def explore_data(self):
        """Perform initial data exploration"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        print("\nTraining Data Info:")
        print(f"Shape: {self.train_data.shape}")
        print(f"Columns: {list(self.train_data.columns)}")
        
        # Check for missing values
        print("\nMissing Values in Training Data:")
        missing_train = self.train_data.isnull().sum()
        missing_train = missing_train[missing_train > 0]
        if len(missing_train) > 0:
            print(missing_train)
        else:
            print("No missing values found!")
        
        # Target variable analysis
        if 'SalePrice' in self.train_data.columns:
            print(f"\nTarget Variable (SalePrice) Statistics:")
            print(f"Mean: ${self.train_data['SalePrice'].mean():,.2f}")
            print(f"Median: ${self.train_data['SalePrice'].median():,.2f}")
            print(f"Std: ${self.train_data['SalePrice'].std():,.2f}")
            print(f"Min: ${self.train_data['SalePrice'].min():,.2f}")
            print(f"Max: ${self.train_data['SalePrice'].max():,.2f}")
            
            # Plot target distribution
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.hist(self.train_data['SalePrice'], bins=50, alpha=0.7, color='skyblue')
            plt.title('Sale Price Distribution')
            plt.xlabel('Sale Price ($)')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            plt.hist(np.log1p(self.train_data['SalePrice']), bins=50, alpha=0.7, color='lightgreen')
            plt.title('Log Sale Price Distribution')
            plt.xlabel('Log Sale Price')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('target_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Target distribution plot saved as 'target_distribution.png'")
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Combine train and test for consistent preprocessing
        train_id = self.train_data['Id'].copy()
        test_id = self.test_data['Id'].copy()
        
        # Remove ID columns
        train_data = self.train_data.drop('Id', axis=1)
        test_data = self.test_data.drop('Id', axis=1)
        
        # Separate target variable
        if 'SalePrice' in train_data.columns:
            self.target = train_data['SalePrice']
            train_data = train_data.drop('SalePrice', axis=1)
        else:
            self.target = None
        
        # Combine for preprocessing
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
        
        # Handle missing values
        print("Handling missing values...")
        
        # Numeric columns
        numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            combined_data[numeric_columns] = self.imputer.fit_transform(combined_data[numeric_columns])
        
        # Categorical columns
        categorical_columns = combined_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            combined_data[col] = combined_data[col].fillna('Unknown')
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            combined_data[col] = self.label_encoders[col].fit_transform(combined_data[col])
        
        # Split back into train and test
        train_processed = combined_data.iloc[:len(train_data)]
        test_processed = combined_data.iloc[len(train_data):]
        
        # Store processed data
        self.X_train = train_processed
        self.X_test = test_processed
        self.train_id = train_id
        self.test_id = test_id
        
        print(f"✅ Preprocessing completed!")
        print(f"   Training features: {self.X_train.shape}")
        print(f"   Test features: {self.X_test.shape}")
        
        return True
    
    def train_models(self):
        """Train multiple models and select the best one"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        if self.target is None:
            print("❌ No target variable found. Cannot train models.")
            return False
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # Define models
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        # Train and evaluate models
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, self.target, 
                                      cv=5, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-cv_scores)
            
            # Train on full dataset
            model.fit(X_train_scaled, self.target)
            
            # Store results
            model_scores[name] = {
                'model': model,
                'cv_rmse_mean': rmse_scores.mean(),
                'cv_rmse_std': rmse_scores.std()
            }
            
            print(f"   CV RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std() * 2:.2f})")
        
        # Select best model
        best_model_name = min(model_scores.keys(), 
                            key=lambda x: model_scores[x]['cv_rmse_mean'])
        self.best_model = model_scores[best_model_name]['model']
        
        print(f"\n✅ Best model: {best_model_name}")
        print(f"   CV RMSE: {model_scores[best_model_name]['cv_rmse_mean']:.2f}")
        
        # Feature importance (for tree-based models)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Feature importance plot saved as 'feature_importance.png'")
        
        return True
    
    def make_predictions(self):
        """Make predictions on test data"""
        print("\n" + "="*50)
        print("MAKING PREDICTIONS")
        print("="*50)
        
        if self.best_model is None:
            print("❌ No trained model available. Please train models first.")
            return False
        
        # Scale test features
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Make predictions
        predictions = self.best_model.predict(X_test_scaled)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'Id': self.test_id,
            'SalePrice': predictions
        })
        
        # Save submission
        submission.to_csv('submission.csv', index=False)
        
        print(f"✅ Predictions completed!")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Submission saved as 'submission.csv'")
        
        # Show prediction statistics
        print(f"\nPrediction Statistics:")
        print(f"   Mean predicted price: ${predictions.mean():,.2f}")
        print(f"   Median predicted price: ${np.median(predictions):,.2f}")
        print(f"   Min predicted price: ${predictions.min():,.2f}")
        print(f"   Max predicted price: ${predictions.max():,.2f}")
        
        return True
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("🏠 HOUSE PRICE PREDICTION PIPELINE")
        print("="*50)
        
        # Load data
        if not self.load_data():
            return False
        
        # Explore data
        self.explore_data()
        
        # Preprocess data
        if not self.preprocess_data():
            return False
        
        # Train models
        if not self.train_models():
            return False
        
        # Make predictions
        if not self.make_predictions():
            return False
        
        print("\n" + "="*50)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Files created:")
        print("- submission.csv (predictions)")
        print("- target_distribution.png (target analysis)")
        print("- feature_importance.png (feature analysis)")
        
        return True

def main():
    """Main function to run the house price prediction pipeline"""
    predictor = HousePricePredictor()
    success = predictor.run_complete_pipeline()
    
    if success:
        print("\n🚀 Ready to submit your predictions!")
        print("Check 'submission.csv' for your results.")
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
