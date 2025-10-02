import numpy as np
import onnxruntime as ort
import joblib
import pandas as pd
from pathlib import Path
import logging

class DNInferencePipeline:
    """
    Diabetic Nephropathy Inference Pipeline
    Loads ONNX model and makes predictions
    """
    
    def __init__(self, model_dir='models/'):
        self.model_dir = Path(model_dir)
        self.session = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = [
            'age', 'gender', 'glucose', 'hba1c', 'creatinine', 
            'urea', 'systolic_bp', 'diastolic_bp'
        ]
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """
        Load the ONNX model and preprocessing components
        """
        try:
            # Load ONNX model
            onnx_path = self.model_dir / 'best_model.onnx'
            if onnx_path.exists():
                self.session = ort.InferenceSession(str(onnx_path))
                self.logger.info(f"ONNX model loaded from {onnx_path}")
            else:
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
            
            # Load scaler
            scaler_path = self.model_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info("Scaler loaded successfully")
            else:
                self.logger.warning("Scaler not found, using identity scaling")
            
            # Load label encoder
            encoder_path = self.model_dir / 'label_encoder.pkl'
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                self.logger.info("Label encoder loaded successfully")
            else:
                # Default label encoder
                from sklearn.preprocessing import LabelEncoder
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(['Low', 'Medium', 'High'])
                self.logger.warning("Using default label encoder")
                
        except Exception as e:
            self.logger.error(f"Error loading model components: {e}")
            raise
    
    def preprocess_input(self, patient_data):
        """
        Preprocess patient data for model input
        
        Args:
            patient_data: Dictionary or DataFrame with patient features
            
        Returns:
            Preprocessed numpy array ready for model inference
        """
        try:
            # Convert to DataFrame if dictionary
            if isinstance(patient_data, dict):
                df = pd.DataFrame([patient_data])
            else:
                df = patient_data.copy()
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = self.get_default_value(feature)
                    self.logger.warning(f"Missing feature {feature}, using default value")
            
            # Select and order features
            X = df[self.feature_names].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)
            
            # Apply scaling if available
            if self.scaler:
                X = self.scaler.transform(X)
            
            return X.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error preprocessing input: {e}")
            raise
    
    def get_default_value(self, feature):
        """
        Get default value for missing features
        """
        defaults = {
            'age': 50.0,
            'gender': 0.0,  # Female
            'glucose': 120.0,
            'hba1c': 6.0,
            'creatinine': 1.0,
            'urea': 30.0,
            'systolic_bp': 120.0,
            'diastolic_bp': 80.0
        }
        return defaults.get(feature, 0.0)
    
    def predict(self, patient_data):
        """
        Make prediction for patient data
        
        Args:
            patient_data: Dictionary or DataFrame with patient features
            
        Returns:
            Tuple of (risk_score, risk_level, confidence, probabilities)
        """
        try:
            if self.session is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            # Preprocess input
            X = self.preprocess_input(patient_data)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: X})
            
            # Extract predictions and probabilities
            if len(outputs) >= 2:
                predictions = outputs[0]
                probabilities = outputs[1]
            else:
                # If only one output, assume it's probabilities
                probabilities = outputs[0]
                predictions = np.argmax(probabilities, axis=1)
            
            # Process results for single prediction
            if len(predictions) == 1:
                pred_class = predictions[0]
                prob_scores = probabilities[0]
                
                # Get risk level
                risk_level = self.label_encoder.inverse_transform([pred_class])[0]
                
                # Calculate risk score (0-100)
                risk_score = float(np.max(prob_scores) * 100)
                
                # Calculate confidence
                confidence = float(np.max(prob_scores))
                
                return risk_score, risk_level, confidence, prob_scores.tolist()
            else:
                # Batch prediction
                results = []
                for i in range(len(predictions)):
                    pred_class = predictions[i]
                    prob_scores = probabilities[i]
                    
                    risk_level = self.label_encoder.inverse_transform([pred_class])[0]
                    risk_score = float(np.max(prob_scores) * 100)
                    confidence = float(np.max(prob_scores))
                    
                    results.append((risk_score, risk_level, confidence, prob_scores.tolist()))
                
                return results
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_proba(self, patient_data):
        """
        Get prediction probabilities for all classes
        
        Args:
            patient_data: Dictionary or DataFrame with patient features
            
        Returns:
            Dictionary with class probabilities
        """
        try:
            _, _, _, probabilities = self.predict(patient_data)
            
            prob_dict = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                prob_dict[class_name] = probabilities[i]
            
            return prob_dict
            
        except Exception as e:
            self.logger.error(f"Probability prediction failed: {e}")
            raise
    
    def get_model_info(self):
        """
        Get information about the loaded model
        """
        if self.session is None:
            return {"status": "Model not loaded"}
        
        try:
            # Get input info
            inputs = []
            for inp in self.session.get_inputs():
                inputs.append({
                    "name": inp.name,
                    "type": str(inp.type),
                    "shape": inp.shape
                })
            
            # Get output info
            outputs = []
            for out in self.session.get_outputs():
                outputs.append({
                    "name": out.name,
                    "type": str(out.type),
                    "shape": out.shape
                })
            
            return {
                "status": "loaded",
                "model_type": "ONNX",
                "inputs": inputs,
                "outputs": outputs,
                "feature_names": self.feature_names,
                "classes": self.label_encoder.classes_.tolist()
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def validate_input(self, patient_data):
        """
        Validate patient data before prediction
        
        Args:
            patient_data: Dictionary with patient features
            
        Returns:
            Dictionary with validation results
        """
        errors = {}
        warnings = []
        
        # Check required fields
        required_fields = ['age', 'glucose', 'hba1c', 'creatinine']
        for field in required_fields:
            if field not in patient_data or patient_data[field] is None:
                errors[field] = f"{field} is required"
        
        # Validate ranges
        if 'age' in patient_data and patient_data['age'] is not None:
            age = float(patient_data['age'])
            if age < 0 or age > 120:
                errors['age'] = "Age must be between 0 and 120 years"
        
        if 'glucose' in patient_data and patient_data['glucose'] is not None:
            glucose = float(patient_data['glucose'])
            if glucose < 50 or glucose > 500:
                errors['glucose'] = "Glucose must be between 50 and 500 mg/dL"
        
        if 'hba1c' in patient_data and patient_data['hba1c'] is not None:
            hba1c = float(patient_data['hba1c'])
            if hba1c < 4.0 or hba1c > 15.0:
                errors['hba1c'] = "HbA1c must be between 4.0 and 15.0%"
        
        if 'creatinine' in patient_data and patient_data['creatinine'] is not None:
            creatinine = float(patient_data['creatinine'])
            if creatinine < 0.5 or creatinine > 10.0:
                errors['creatinine'] = "Creatinine must be between 0.5 and 10.0 mg/dL"
        
        # Check for missing optional fields
        optional_fields = ['urea', 'systolic_bp', 'diastolic_bp']
        missing_optional = [f for f in optional_fields if f not in patient_data or patient_data[f] is None]
        if missing_optional:
            warnings.append(f"Optional fields missing: {', '.join(missing_optional)}. Default values will be used.")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

def load_inference_pipeline(model_dir='models/'):
    """
    Factory function to create and load inference pipeline
    """
    pipeline = DNInferencePipeline(model_dir)
    pipeline.load_model()
    return pipeline

# Example usage
if __name__ == "__main__":
    # Example patient data
    patient_data = {
        'age': 65,
        'gender': 1,  # Male
        'glucose': 180,
        'hba1c': 8.5,
        'creatinine': 1.8,
        'urea': 45,
        'systolic_bp': 145,
        'diastolic_bp': 90
    }
    
    try:
        # Load pipeline
        pipeline = load_inference_pipeline()
        
        # Validate input
        validation = pipeline.validate_input(patient_data)
        print(f"Validation: {validation}")
        
        if validation['valid']:
            # Make prediction
            risk_score, risk_level, confidence, probabilities = pipeline.predict(patient_data)
            
            print(f"\\nPrediction Results:")
            print(f"Risk Score: {risk_score:.1f}%")
            print(f"Risk Level: {risk_level}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Class Probabilities: {probabilities}")
            
            # Get model info
            model_info = pipeline.get_model_info()
            print(f"\\nModel Info: {model_info}")
        
    except Exception as e:
        print(f"Error: {e}")