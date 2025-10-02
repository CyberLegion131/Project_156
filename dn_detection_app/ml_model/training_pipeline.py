import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DNTrainingPipeline:
    """
    Diabetic Nephropathy Training Pipeline
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = [
            'age', 'gender', 'glucose', 'hba1c', 'creatinine', 
            'urea', 'systolic_bp', 'diastolic_bp'
        ]
    
    def load_data(self, filepath=None):
        """
        Load and prepare the dataset
        """
        if filepath and filepath.endswith('.xlsx'):
            try:
                # Try to load from Excel file
                data = pd.read_excel(filepath)
                print(f"Data loaded from {filepath}")
                print(f"Shape: {data.shape}")
                return data
            except Exception as e:
                print(f"Error loading Excel file: {e}")
                print("Creating synthetic dataset for training...")
        
        # Create synthetic dataset for training
        return self.create_synthetic_dataset()
    
    def create_synthetic_dataset(self, n_samples=2000):
        """
        Create synthetic diabetic nephropathy dataset
        """
        np.random.seed(42)
        
        # Generate base features
        age = np.random.normal(65, 15, n_samples)
        age = np.clip(age, 30, 90)
        
        gender = np.random.choice([0, 1], n_samples)
        
        # Generate correlated medical features
        # Glucose levels (higher in diabetics)
        glucose = np.random.normal(150, 50, n_samples)
        glucose = np.clip(glucose, 80, 400)
        
        # HbA1c (glycated hemoglobin)
        hba1c = glucose / 30 + np.random.normal(0, 1, n_samples)
        hba1c = np.clip(hba1c, 5, 15)
        
        # Creatinine (kidney function marker)
        base_creatinine = 0.8 + (age - 30) * 0.01 + gender * 0.2
        creatinine = base_creatinine + np.random.normal(0, 0.3, n_samples)
        creatinine = np.clip(creatinine, 0.5, 8.0)
        
        # Urea (related to kidney function)
        urea = creatinine * 25 + np.random.normal(0, 10, n_samples)
        urea = np.clip(urea, 15, 150)
        
        # Blood pressure (related to diabetes complications)
        systolic_bp = 120 + (glucose - 100) * 0.2 + np.random.normal(0, 15, n_samples)
        systolic_bp = np.clip(systolic_bp, 90, 200)
        
        diastolic_bp = systolic_bp * 0.6 + np.random.normal(0, 10, n_samples)
        diastolic_bp = np.clip(diastolic_bp, 60, 120)
        
        # Create risk levels based on medical thresholds
        risk_scores = []
        for i in range(n_samples):
            score = 0
            
            # Age factor
            if age[i] > 65:
                score += 1
            elif age[i] > 55:
                score += 0.5
                
            # Glucose factor
            if glucose[i] > 200:
                score += 2
            elif glucose[i] > 140:
                score += 1
                
            # HbA1c factor
            if hba1c[i] > 9:
                score += 2
            elif hba1c[i] > 7:
                score += 1
                
            # Creatinine factor (most important for kidney disease)
            if creatinine[i] > 2.0:
                score += 3
            elif creatinine[i] > 1.5:
                score += 2
            elif creatinine[i] > 1.2:
                score += 1
                
            # Blood pressure factor
            if systolic_bp[i] > 160:
                score += 1
            elif systolic_bp[i] > 140:
                score += 0.5
                
            risk_scores.append(score)
        
        # Convert scores to risk levels
        risk_levels = []
        for score in risk_scores:
            if score <= 2:
                risk_levels.append('Low')
            elif score <= 5:
                risk_levels.append('Medium')
            else:
                risk_levels.append('High')
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'glucose': glucose,
            'hba1c': hba1c,
            'creatinine': creatinine,
            'urea': urea,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'risk_level': risk_levels
        })
        
        print(f"Synthetic dataset created with {n_samples} samples")
        print(f"Risk level distribution:")
        print(data['risk_level'].value_counts())
        
        return data
    
    def explore_data(self, data):
        """
        Perform exploratory data analysis
        """
        print("\\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {data.shape}")
        print(f"\\nMissing values:")
        print(data.isnull().sum())
        
        print(f"\\nBasic statistics:")
        print(data.describe())
        
        # Risk level distribution
        print(f"\\nRisk level distribution:")
        print(data['risk_level'].value_counts(normalize=True))
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Diabetic Nephropathy Dataset - Feature Distributions')
        
        # Plot distributions
        for i, feature in enumerate(['age', 'glucose', 'hba1c', 'creatinine', 'urea', 'systolic_bp']):
            row, col = i // 3, i % 3
            axes[row, col].hist(data[feature], bins=30, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{feature.title()} Distribution')
            axes[row, col].set_xlabel(feature.title())
            axes[row, col].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_data = data.copy()
        correlation_data['risk_level_encoded'] = self.label_encoder.fit_transform(data['risk_level'])
        
        corr_matrix = correlation_data.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the data for training
        """
        print("\\n=== DATA PREPROCESSING ===")
        
        # Separate features and target
        X = data[self.feature_names].copy()
        y = data['risk_level'].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Features: {self.feature_names}")
        print(f"Target classes: {self.label_encoder.classes_}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train and compare different models
        """
        print("\\n=== MODEL TRAINING ===")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"Train Accuracy: {train_score:.4f}")
            print(f"Test Accuracy: {test_score:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        best_model = results[best_model_name]['model']
        
        print(f"\\nBest model: {best_model_name}")
        print(f"Test accuracy: {results[best_model_name]['test_accuracy']:.4f}")
        
        self.model = best_model
        return best_model, results
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Detailed model evaluation
        """
        print("\\n=== MODEL EVALUATION ===")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance_df, x='importance', y='feature')
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            print("\\nFeature Importance:")
            print(feature_importance_df)
        
        return y_pred, y_pred_proba
    
    def save_model(self, model_path='models/', scaler_path='models/'):
        """
        Save the trained model and scaler
        """
        import os
        os.makedirs(model_path, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(scaler_path, 'scaler.pkl'))
        print(f"Scaler saved to {scaler_path}/scaler.pkl")
        
        # Save label encoder
        joblib.dump(self.label_encoder, os.path.join(model_path, 'label_encoder.pkl'))
        print(f"Label encoder saved to {model_path}/label_encoder.pkl")
        
        # Save sklearn model
        if self.model:
            joblib.dump(self.model, os.path.join(model_path, 'best_model.pkl'))
            print(f"Model saved to {model_path}/best_model.pkl")
        
        # Convert to ONNX if possible
        try:
            self.convert_to_onnx(os.path.join(model_path, 'best_model.onnx'))
        except Exception as e:
            print(f"ONNX conversion failed: {e}")
    
    def convert_to_onnx(self, onnx_path):
        """
        Convert sklearn model to ONNX format
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Define input type
            initial_type = [('float_input', FloatTensorType([None, len(self.feature_names)]))]
            
            # Convert model
            onnx_model = convert_sklearn(self.model, initial_types=initial_type)
            
            # Save ONNX model
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"ONNX model saved to {onnx_path}")
            
        except ImportError:
            print("skl2onnx not available. Install with: pip install skl2onnx")
        except Exception as e:
            print(f"Error converting to ONNX: {e}")

def main():
    """
    Main training pipeline
    """
    pipeline = DNTrainingPipeline()
    
    # Load data
    # data = pipeline.load_data('../../Diabetic_Nephropathy_v1.xlsx')  # Uncomment if Excel file exists
    data = pipeline.load_data()  # Use synthetic data
    
    # Explore data
    data = pipeline.explore_data(data)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)
    
    # Train models
    best_model, results = pipeline.train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate best model
    y_pred, y_pred_proba = pipeline.evaluate_model(best_model, X_test, y_test)
    
    # Save model
    pipeline.save_model()
    
    print("\\n=== TRAINING COMPLETED ===")
    print("Model artifacts saved to 'models/' directory")
    print("Ready for deployment!")

if __name__ == "__main__":
    main()