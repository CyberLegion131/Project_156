import numpy as np
import onnxruntime as ort
from typing import Tuple, Dict, Any, List
import logging
import os
from pathlib import Path
import joblib
import json

from app.utils.logger import setup_logger

logger = setup_logger()

class DNPredictor:
    """
    Diabetic Nethropathy predictor with automatic best model selection
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_config = None
        self.models_dir = Path(__file__).parent.parent.parent.parent / "ml_model" / "models"
        self.config_path = self.models_dir / "model_config.json"
        self.scaler_path = self.models_dir / "scaler.pkl"
        
        # All 21 clinical features from the dataset - EXACT ORDER from training notebook
        self.feature_names = [
            'sex',                      # 1. Sex
            'age',                      # 2. Age
            'diabetes_duration_y',      # 3. Diabetes duration (y)
            'diabetic_retinopathy_dr',  # 4. Diabetic retinopathy (DR)
            'smoking',                  # 5. Smoking
            'drinking',                 # 6. Drinking
            'height_cm',                # 7. Height(cm)
            'weight_kg',                # 8. Weight(kg)
            'bmi',                      # 9. BMI (kg/m2)
            'sbp_mmhg',                 # 10. SBP (mmHg)
            'dbp_mmhg',                 # 11. DBP (mmHg)
            'hba1c_percent',            # 12. HbA1c (%)
            'fbg_mmol_l',               # 13. FBG (mmol/L)
            'tg_mmol_l',                # 14. TG（mmoll）
            'c_peptide_ng_ml',          # 15. C-peptide (ng/ml）
            'tc_mmol_l',                # 16. TC（mmoll）
            'hdlc_mmol_l',              # 17. HDLC（mmoll）
            'ldlc_mmol_l',              # 18. LDLC（mmoll）
            'insulin',                  # 19. Insulin
            'metformin',                # 20. Metformin
            'lipid_lowering_drugs'      # 21. Lipid lowering drugs
        ]
    
    async def load_model(self):
        """
        🎯 Load the AUTOMATICALLY SELECTED BEST MODEL
        This reads model_config.json to determine which model to load
        """
        try:
            # Load model configuration (created by training notebook)
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.model_config = json.load(f)
                logger.info(f"Model config loaded: Best model is {self.model_config['best_model']['name']}")
                logger.info(f"Best model accuracy: {self.model_config['best_model']['accuracy_percentage']}")
            else:
                logger.warning("Model config not found. Using fallback model.")
                self._create_fallback_config()
            
            # Determine which model file to load
            best_model_file = self.model_config['best_model']['file_path']
            model_path = self.models_dir / best_model_file
            
            # Load the best model (scikit-learn pickle format)
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"BEST MODEL LOADED: {best_model_file}")
                logger.info(f"Model Type: {self.model_config['best_model']['model_type']}")
                logger.info(f"Accuracy: {self.model_config['best_model']['accuracy_percentage']}")
            else:
                logger.warning(f"Best model file not found: {model_path}")
                self._create_dummy_model()
            
            # Load scaler if exists
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Scaler loaded from {self.scaler_path}")
            else:
                logger.warning("Scaler not found. Using dummy scaler.")
                self._create_dummy_scaler()
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _create_fallback_config(self):
        """
        Create fallback configuration when model_config.json is not found
        """
        self.model_config = {
            "best_model": {
                "name": "Extra Trees (Ensemble)",
                "accuracy": 0.7792,
                "accuracy_percentage": "77.92%",
                "file_path": "extra_trees_model.pkl",
                "model_type": "ensemble",
                "interpretable": False
            }
        }
        logger.info("Created fallback model configuration")

    def _create_dummy_model(self):
        """
        Create a dummy scikit-learn model for development purposes
        """
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            from sklearn.datasets import make_classification
            
            # Create dummy data with 21 features (matching our clinical parameters)
            X, y = make_classification(
                n_samples=1000, 
                n_features=21,  # All clinical features
                n_classes=2,    # DN vs No DN
                random_state=42
            )
            
            # Train dummy model (Extra Trees as it's currently the best)
            self.model = ExtraTreesClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)
            
            # Save dummy model
            os.makedirs(self.models_dir, exist_ok=True)
            dummy_model_path = self.models_dir / "extra_trees_model.pkl"
            joblib.dump(self.model, dummy_model_path)
            logger.info(f"Created dummy model: {dummy_model_path}")
                
        except Exception as e:
            logger.error(f"Failed to create dummy model: {str(e)}")
            raise
    
    def _create_dummy_scaler(self):
        """
        Create a dummy scaler for development with all 21 clinical features
        """
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # Create dummy scaler with reasonable parameters for all 21 clinical features
        self.scaler = StandardScaler()
        
        # Dummy data with realistic medical values for all 21 features
        dummy_data = np.array([
            # [age, gender, glucose, hba1c, creatinine, urea, systolic_bp, diastolic_bp, 
            #  bmi, diabetes_duration, family_history, smoking, alcohol, hypertension,
            #  dyslipidemia, cardiac_disease, retinopathy, neuropathy, foot_ulcer, amputation, insulin_use]
            [65, 1, 150, 8.5, 1.2, 35, 140, 85, 28.5, 10, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],  # High risk
            [45, 0, 120, 6.5, 0.9, 25, 110, 70, 24.0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # Low risk
            [75, 1, 200, 10.5, 2.5, 55, 160, 95, 32.0, 15, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]  # Very high risk
        ])
        self.scaler.fit(dummy_data)
        
        # Save scaler
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Created dummy scaler for 21 clinical features")
    
    def enhanced_clinical_rules(self, patient_data: Dict[str, Any], original_risk_score: float) -> Tuple[float, bool, str, List[str]]:
        """
        🔧 Apply clinical decision rules to enhance DN prediction accuracy
        More conservative approach - only boost when strong evidence exists
        
        Args:
            patient_data: Original patient data dictionary (before preprocessing)
            original_risk_score: Risk score from the original model (0-100)
            
        Returns:
            Tuple of (enhanced_risk_score, clinical_override, override_reason, high_risk_factors)
        """
        high_risk_factors = []
        risk_score_boost = 0
        
        # Get key clinical parameters
        has_retinopathy = patient_data.get('diabetic_retinopathy_dr', 0) == 1
        duration = patient_data.get('diabetes_duration_y', 0)
        hba1c = patient_data.get('hba1c_percent', 0)
        sbp = patient_data.get('sbp_mmhg', 0)
        dbp = patient_data.get('dbp_mmhg', 0)
        age = patient_data.get('age', 0)
        takes_insulin = patient_data.get('insulin', 0) == 1
        
        # 🎯 CONSERVATIVE APPROACH: Only boost for strong evidence of DN risk
        
        # Rule 1: Diabetic retinopathy (strongest DN predictor) - MAJOR BOOST
        if has_retinopathy:
            high_risk_factors.append("Has diabetic retinopathy")
            risk_score_boost += 30  # Strong predictor
            
        # Rule 2: Very poor glycemic control - ONLY if severe
        if hba1c > 12:  # Only very high HbA1c (was >10)
            high_risk_factors.append(f"Very high HbA1c ({hba1c}%)")
            risk_score_boost += 25
        elif hba1c > 10 and has_retinopathy:  # Only if has retinopathy too
            high_risk_factors.append(f"High HbA1c with retinopathy ({hba1c}%)")
            risk_score_boost += 15
            
        # Rule 3: Very long diabetes duration - ONLY if very long
        if duration > 20:  # Only very long duration (was >15)
            high_risk_factors.append(f"Very long diabetes duration ({duration} years)")
            risk_score_boost += 20
        elif duration > 15 and has_retinopathy:  # Only if has retinopathy too
            high_risk_factors.append(f"Long duration with retinopathy ({duration} years)")
            risk_score_boost += 15
            
        # Rule 4: Severe hypertension - ONLY if very severe
        if sbp > 170 or dbp > 105:  # Only very severe hypertension (was >160/100)
            high_risk_factors.append(f"Very severe hypertension ({sbp}/{dbp})")
            risk_score_boost += 15
        elif (sbp > 160 or dbp > 100) and has_retinopathy:  # Only if has retinopathy too
            risk_score_boost += 10
            
        # Rule 5: Insulin use - ONLY in combination with other factors
        if takes_insulin and (has_retinopathy or duration > 15 or hba1c > 10):
            risk_score_boost += 10  # Only boost if has other risk factors
            
        # Rule 6: Advanced age - ONLY in combination with other factors  
        if age > 70 and (has_retinopathy or duration > 15):
            risk_score_boost += 10  # Only boost if has other risk factors
            
        # 🔒 SAFETY CHECK: Don't boost low-risk patients without strong evidence
        if original_risk_score < 20 and not has_retinopathy:
            # If original risk is very low and no retinopathy, limit boost
            risk_score_boost = min(risk_score_boost, 10)
            
        # Calculate enhanced risk score
        enhanced_risk_score = min(95, original_risk_score + risk_score_boost)
        
        # Clinical override conditions - MORE CONSERVATIVE
        clinical_override = False
        override_reason = ""
        
        # Override ONLY for very strong evidence
        if len(high_risk_factors) >= 2:
            clinical_override = True
            override_reason = f"Multiple severe risk factors: {', '.join(high_risk_factors)}"
            
        # Override for critical combinations with retinopathy
        if has_retinopathy and (duration > 20 or hba1c > 12 or sbp > 170):
            clinical_override = True
            override_reason = "Critical combination: Retinopathy + severe complications"
            
        # Override for extreme cases
        if hba1c > 14 or (has_retinopathy and duration > 25):
            clinical_override = True
            override_reason = "Extreme risk factors detected"
            
        return enhanced_risk_score, clinical_override, override_reason, high_risk_factors

    async def predict(self, input_data: np.ndarray, patient_data: Dict[str, Any] = None) -> Tuple[float, str, float]:
        """
        🎯 Enhanced prediction using AUTOMATICALLY SELECTED BEST MODEL + Clinical Rules
        
        Args:
            input_data: Preprocessed numpy array for model input
            patient_data: Original patient data dictionary (for clinical rules)
        
        Returns:
            Tuple of (risk_score, risk_level, confidence, model_accuracy, binary_prediction, clinical_details)
        """
        try:
            if self.model is None:
                raise ValueError("❌ Model not loaded")
            
            # Ensure input is correct shape
            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1)
            
            # Validate input features (should be 21 features for full clinical data)
            expected_features = len(self.feature_names)
            if input_data.shape[1] != expected_features:
                logger.warning(f"⚠️ Input has {input_data.shape[1]} features, expected {expected_features}")
                # Pad or truncate as needed for backward compatibility
                if input_data.shape[1] < expected_features:
                    # Pad with zeros for missing features
                    padding = np.zeros((input_data.shape[0], expected_features - input_data.shape[1]))
                    input_data = np.hstack([input_data, padding])
                else:
                    # Truncate to expected size
                    input_data = input_data[:, :expected_features]
            
            # Scale input data
            if self.scaler:
                input_data = self.scaler.transform(input_data)
            
            # 🎯 Step 1: Original model prediction
            if hasattr(self.model, 'predict_proba'):
                # Get probabilities for binary classification
                probabilities = self.model.predict_proba(input_data)[0]
                logger.info(f"Raw model probabilities: {probabilities}")
                
                # Get actual binary prediction from the model
                original_binary_prediction = int(self.model.predict(input_data)[0])  # 0 or 1
                
                # For binary classification (DN vs No DN)
                # Training data: class 0 = no DN (568 patients), class 1 = has DN (199 patients)
                if len(probabilities) == 2:
                    no_dn_probability = probabilities[0]  # Probability of NO DN
                    dn_probability = probabilities[1]     # Probability of HAS DN
                    
                    original_risk_score = float(dn_probability * 100)
                    original_confidence = float(np.max(probabilities))
                    
                    logger.info(f"Original Model - DN Probability: {dn_probability:.3f} ({original_risk_score:.1f}%)")
                    logger.info(f"Original Binary Prediction: {original_binary_prediction} ({'Has DN' if original_binary_prediction == 1 else 'No DN'})")
                else:
                    # Multi-class classification fallback
                    predicted_class = np.argmax(probabilities)
                    original_risk_score = float(np.max(probabilities) * 100)
                    original_confidence = float(np.max(probabilities))
                    original_binary_prediction = predicted_class
            else:
                # Fallback for models without predict_proba
                prediction = self.model.predict(input_data)[0]
                original_binary_prediction = int(prediction)
                original_risk_score = float(prediction * 100) if isinstance(prediction, (int, float)) else 50.0
                original_confidence = 0.75
            
            # 🔧 Step 2: Apply clinical enhancement if patient data is available
            clinical_details = {}
            if patient_data is not None:
                enhanced_risk_score, clinical_override, override_reason, risk_factors = \
                    self.enhanced_clinical_rules(patient_data, original_risk_score)
                
                # Step 3: Final decision logic
                if clinical_override:
                    final_prediction = 1
                    final_risk_score = max(enhanced_risk_score, 65)  # Minimum 65% for override
                    confidence_level = "HIGH"
                    explanation = f"Clinical Override: {override_reason}"
                elif enhanced_risk_score > 40:  # Lower threshold than original 50%
                    final_prediction = 1
                    final_risk_score = enhanced_risk_score
                    confidence_level = "MODERATE-HIGH"
                    explanation = f"Enhanced Model: Risk factors detected"
                else:
                    final_prediction = original_binary_prediction
                    final_risk_score = enhanced_risk_score
                    confidence_level = "LOW-MODERATE" if enhanced_risk_score > 25 else "LOW"
                    explanation = f"Standard Model: No major risk factors"
                
                # Determine risk level based on final score
                if final_risk_score >= 67:
                    risk_level = "High"
                elif final_risk_score >= 34:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                
                confidence = original_confidence  # Keep original model confidence
                
                clinical_details = {
                    'original_model_score': round(original_risk_score, 1),
                    'enhanced_score': round(final_risk_score, 1),
                    'clinical_override': clinical_override,
                    'risk_factors_detected': risk_factors,
                    'override_reason': override_reason if clinical_override else None,
                    'confidence_level': confidence_level,
                    'explanation': explanation
                }
                
                logger.info(f"🔧 ENHANCED PREDICTION:")
                logger.info(f"   Original: {original_risk_score:.1f}% → Enhanced: {final_risk_score:.1f}%")
                logger.info(f"   Clinical Override: {clinical_override}")
                logger.info(f"   Risk Factors: {risk_factors}")
                logger.info(f"   Final Decision: {risk_level} Risk ({final_risk_score:.1f}%)")
                
            else:
                # No patient data available - use original model prediction only
                final_risk_score = original_risk_score
                final_prediction = original_binary_prediction
                confidence = original_confidence
                
                # Determine risk level based on original DN probability
                if final_risk_score >= 67:
                    risk_level = "High"
                elif final_risk_score >= 34:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                
                clinical_details = {
                    'original_model_score': round(original_risk_score, 1),
                    'enhanced_score': round(final_risk_score, 1),
                    'clinical_override': False,
                    'risk_factors_detected': [],
                    'override_reason': None,
                    'confidence_level': 'STANDARD',
                    'explanation': 'Standard model prediction (no clinical enhancement)'
                }
                
                logger.info(f"Standard Model Prediction: {risk_level} ({final_risk_score:.1f}%)")
            
            # Log prediction details
            model_name = self.model_config['best_model']['name'] if self.model_config else "Unknown"
            model_accuracy = self.model_config['best_model']['accuracy_percentage'] if self.model_config else "Unknown"
            logger.info(f"Final Prediction by {model_name}: {risk_level} ({final_risk_score:.1f}%) - Confidence: {confidence:.3f}")
            
            return final_risk_score, risk_level, confidence, model_accuracy, final_prediction, clinical_details
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            # Return safe fallback values
            return 50.0, "Medium", 0.60, "Unknown", 0, {}
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        🎯 Get information about the AUTOMATICALLY SELECTED BEST MODEL
        """
        try:
            if self.model is None:
                return {"status": "❌ Model not loaded"}
            
            # Get basic model info
            model_info = {
                "status": "✅ loaded",
                "feature_names": self.feature_names,
                "num_features": len(self.feature_names),
                "model_class": str(type(self.model).__name__)
            }
            
            # Add best model configuration info
            if self.model_config:
                model_info.update({
                    "best_model_name": self.model_config['best_model']['name'],
                    "accuracy": self.model_config['best_model']['accuracy_percentage'],
                    "model_type": self.model_config['best_model']['model_type'],
                    "interpretable": self.model_config['best_model']['interpretable'],
                    "file_path": self.model_config['best_model']['file_path']
                })
                
                # Add all available models
                if 'all_models' in self.model_config:
                    model_info["all_available_models"] = [
                        {
                            "name": m['name'],
                            "accuracy": m['accuracy_percentage'],
                            "is_best": m['is_best']
                        } for m in self.model_config['all_models']
                    ]
            
            # Add scikit-learn specific info if available
            if hasattr(self.model, 'n_estimators'):
                model_info["n_estimators"] = self.model.n_estimators
            if hasattr(self.model, 'feature_importances_'):
                # Get top 5 most important features
                importances = self.model.feature_importances_
                top_features = sorted(
                    zip(self.feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                model_info["top_features"] = [
                    {"feature": name, "importance": float(importance)}
                    for name, importance in top_features
                ]
            
            return model_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get model info: {str(e)}")
            return {"status": "error", "message": str(e)}

# Global predictor instance
predictor_instance = None

async def get_predictor() -> DNPredictor:
    """
    Dependency to get the global predictor instance
    """
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = DNPredictor()
        await predictor_instance.load_model()
    return predictor_instance