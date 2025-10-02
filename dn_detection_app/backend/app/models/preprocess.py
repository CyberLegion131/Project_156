import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

from app.utils.logger import setup_logger

logger = setup_logger()

def preprocess_patient_data(patient_data: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess patient data for model input - ALL 21 CLINICAL PARAMETERS
    
    Args:
        patient_data: Dictionary containing patient information
        
    Returns:
        numpy array ready for model prediction (21 features)
    """
    try:
        # Define ALL 21 features in the EXACT order from the dataset
        feature_order = [
            'sex',                      # Sex
            'age',                      # Age  
            'diabetes_duration_y',      # Diabetes duration (y)
            'diabetic_retinopathy_dr',  # Diabetic retinopathy (DR)
            'smoking',                  # Smoking
            'drinking',                 # Drinking
            'height_cm',                # Height(cm)
            'weight_kg',                # Weight(kg)
            'bmi',                      # BMI (kg/m2)
            'sbp_mmhg',                 # SBP (mmHg) 
            'dbp_mmhg',                 # DBP (mmHg)
            'hba1c_percent',            # HbA1c (%)
            'fbg_mmol_l',               # FBG (mmol/L)
            'tg_mmol_l',                # TG（mmoll）
            'c_peptide_ng_ml',          # C-peptide (ng/ml）
            'tc_mmol_l',                # TC（mmoll）
            'hdlc_mmol_l',              # HDLC（mmoll）
            'ldlc_mmol_l',              # LDLC（mmoll）
            'insulin',                  # Insulin
            'metformin',                # Metformin
            'lipid_lowering_drugs'      # Lipid lowering drugs
        ]
        
        # Initialize processed data array for all 21 features
        processed_data = np.zeros(len(feature_order))
        
        # Process each feature
        for i, feature in enumerate(feature_order):
            if feature in patient_data and patient_data[feature] is not None:
                value = patient_data[feature]
                
                # Handle specific preprocessing for each feature - MATCH TRAINING ENCODING EXACTLY
                if feature == 'sex':
                    # Training: Male=0, Female=1
                    if isinstance(value, str):
                        processed_data[i] = 1.0 if value.lower() in ['female', 'f', '1'] else 0.0
                    else:
                        processed_data[i] = 1.0 if float(value) == 1 else 0.0
                        
                elif feature == 'diabetic_retinopathy_dr':
                    # Training: 1→0, 0→1 (REVERSED encoding!)
                    if isinstance(value, str):
                        original_val = 1.0 if value.lower() in ['yes', 'y', 'true', '1'] else 0.0
                    else:
                        original_val = float(value)
                    processed_data[i] = 0.0 if original_val == 1.0 else 1.0  # REVERSE
                    
                elif feature == 'smoking':
                    # Training: 1→0, 0→1 (REVERSED encoding!)
                    if isinstance(value, str):
                        original_val = 1.0 if value.lower() in ['yes', 'y', 'true', '1'] else 0.0
                    else:
                        original_val = float(value)
                    processed_data[i] = 0.0 if original_val == 1.0 else 1.0  # REVERSE
                    
                elif feature == 'drinking':
                    # Training: 0→0, 1→1 (NORMAL encoding)
                    if isinstance(value, str):
                        processed_data[i] = 1.0 if value.lower() in ['yes', 'y', 'true', '1'] else 0.0
                    else:
                        processed_data[i] = float(value)
                        
                elif feature == 'insulin':
                    # Training: 1→0, 0→1 (REVERSED encoding!)
                    if isinstance(value, str):
                        original_val = 1.0 if value.lower() in ['yes', 'y', 'true', '1'] else 0.0
                    else:
                        original_val = float(value)
                    processed_data[i] = 0.0 if original_val == 1.0 else 1.0  # REVERSE
                    
                elif feature == 'metformin':
                    # Training: 1→0, 0→1 (REVERSED encoding!)
                    if isinstance(value, str):
                        original_val = 1.0 if value.lower() in ['yes', 'y', 'true', '1'] else 0.0
                    else:
                        original_val = float(value)
                    processed_data[i] = 0.0 if original_val == 1.0 else 1.0  # REVERSE
                    
                elif feature == 'lipid_lowering_drugs':
                    # Training: 1→0, 0→1 (REVERSED encoding!)
                    if isinstance(value, str):
                        original_val = 1.0 if value.lower() in ['yes', 'y', 'true', '1'] else 0.0
                    else:
                        original_val = float(value)
                    processed_data[i] = 0.0 if original_val == 1.0 else 1.0  # REVERSE
                
                elif feature == 'age':
                    # Age in years
                    age = float(value)
                    processed_data[i] = max(0, min(120, age))
                
                elif feature == 'diabetes_duration_y':
                    # Duration of diabetes in years
                    duration = float(value)
                    processed_data[i] = max(0, min(50, duration))
                
                elif feature == 'height_cm':
                    # Height in cm
                    height = float(value)
                    processed_data[i] = max(100, min(250, height))
                
                elif feature == 'weight_kg':
                    # Weight in kg
                    weight = float(value)
                    processed_data[i] = max(30, min(200, weight))
                
                elif feature == 'bmi':
                    # Body Mass Index
                    bmi = float(value)
                    processed_data[i] = max(15.0, min(50.0, bmi))
                
                elif feature == 'sbp_mmhg':
                    # Systolic blood pressure
                    bp = float(value)
                    processed_data[i] = max(80, min(250, bp))
                
                elif feature == 'dbp_mmhg':
                    # Diastolic blood pressure
                    bp = float(value)
                    processed_data[i] = max(40, min(150, bp))
                
                elif feature == 'hba1c_percent':
                    # HbA1c percentage
                    hba1c = float(value)
                    processed_data[i] = max(4.0, min(15.0, hba1c))
                
                elif feature == 'fbg_mmol_l':
                    # Fasting blood glucose in mmol/L
                    fbg = float(value)
                    processed_data[i] = max(2.0, min(30.0, fbg))
                
                elif feature in ['tg_mmol_l', 'tc_mmol_l', 'hdlc_mmol_l', 'ldlc_mmol_l']:
                    # Lipid values in mmol/L
                    lipid = float(value)
                    if feature == 'tg_mmol_l':
                        processed_data[i] = max(0.3, min(20.0, lipid))
                    elif feature == 'tc_mmol_l':
                        processed_data[i] = max(2.0, min(15.0, lipid))
                    elif feature == 'hdlc_mmol_l':
                        processed_data[i] = max(0.5, min(3.0, lipid))
                    elif feature == 'ldlc_mmol_l':
                        processed_data[i] = max(1.0, min(10.0, lipid))
                
                elif feature == 'c_peptide_ng_ml':
                    # C-peptide in ng/ml
                    cpeptide = float(value)
                    processed_data[i] = max(0.1, min(10.0, cpeptide))
                
                else:
                    # Default: just convert to float
                    processed_data[i] = float(value)
            else:
                # Handle missing values with appropriate defaults
                processed_data[i] = get_default_value(feature)
                logger.warning(f"Missing feature {feature}, using default value: {processed_data[i]}")
        
        # Log detailed preprocessing for verification
        logger.info(f"PREPROCESSING VERIFICATION:")
        logger.info(f"   Input data keys: {list(patient_data.keys())}")
        for i, feature in enumerate(feature_order):
            original_val = patient_data.get(feature, "MISSING")
            processed_val = processed_data[i]
            logger.info(f"   {i+1:2d}. {feature:<25} | Input: {original_val:<8} -> Processed: {processed_val}")
        
        logger.info(f"Final preprocessed array (21 features): {processed_data}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise ValueError(f"Failed to preprocess patient data: {str(e)}")

def get_default_value(feature: str) -> float:
    """
    Get default value for missing features - MATCH TRAINING DATA MEDIANS
    """
    defaults = {
        # Demographics & Physical - match training medians
        'sex': 0.0,  # Male (most common in dataset)
        'age': 50.0,
        'height_cm': 168.0,  # Training median
        'weight_kg': 70.0,   
        'bmi': 24.8,  # Training median
        # Diabetes History
        'diabetes_duration_y': 9.0,  # Training median
        'diabetic_retinopathy_dr': 1.0,  # No retinopathy (encoded as 1)
        # Lifestyle - match training encoding
        'smoking': 1.0,  # Non-smoker (encoded as 1 due to reversal)
        'drinking': 0.0,  # No drinking (normal encoding)
        # Vital Signs
        'sbp_mmhg': 130.0,  # Typical diabetic BP
        'dbp_mmhg': 80.0,   
        # Laboratory Values - match training medians
        'hba1c_percent': 8.5,  # Training median
        'fbg_mmol_l': 7.83,     # Training median
        'tg_mmol_l': 1.58,      # Training median
        'c_peptide_ng_ml': 1.0, # Training median
        'tc_mmol_l': 4.35,      # Training median
        'hdlc_mmol_l': 1.02,    # Training median
        'ldlc_mmol_l': 2.56,    # Training median
        # Medications - match training encoding (reversed)
        'insulin': 1.0,         # No insulin (encoded as 1 due to reversal)
        'metformin': 1.0,       # No metformin (encoded as 1 due to reversal) 
        'lipid_lowering_drugs': 1.0  # No lipid drugs (encoded as 1 due to reversal)
    }
    return defaults.get(feature, 0.0)

def validate_patient_data(patient_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate patient data and return validation errors
    
    Returns:
        Dictionary of field_name: error_message for any validation errors
    """
    errors = {}
    
    # Required fields
    required_fields = ['age', 'glucose', 'hba1c', 'creatinine']
    for field in required_fields:
        if field not in patient_data or patient_data[field] is None:
            errors[field] = f"{field} is required"
    
    # Validate ranges
    if 'age' in patient_data:
        age = patient_data['age']
        if not isinstance(age, (int, float)) or age < 0 or age > 120:
            errors['age'] = "Age must be between 0 and 120"
    
    if 'glucose' in patient_data:
        glucose = patient_data['glucose']
        if not isinstance(glucose, (int, float)) or glucose < 50 or glucose > 500:
            errors['glucose'] = "Glucose must be between 50 and 500 mg/dL"
    
    if 'hba1c' in patient_data:
        hba1c = patient_data['hba1c']
        if not isinstance(hba1c, (int, float)) or hba1c < 4.0 or hba1c > 15.0:
            errors['hba1c'] = "HbA1c must be between 4.0 and 15.0%"
    
    if 'creatinine' in patient_data:
        creatinine = patient_data['creatinine']
        if not isinstance(creatinine, (int, float)) or creatinine < 0.5 or creatinine > 10.0:
            errors['creatinine'] = "Creatinine must be between 0.5 and 10.0 mg/dL"
    
    if 'systolic_bp' in patient_data:
        systolic = patient_data['systolic_bp']
        if not isinstance(systolic, (int, float)) or systolic < 80 or systolic > 250:
            errors['systolic_bp'] = "Systolic BP must be between 80 and 250 mmHg"
    
    if 'diastolic_bp' in patient_data:
        diastolic = patient_data['diastolic_bp']
        if not isinstance(diastolic, (int, float)) or diastolic < 40 or diastolic > 150:
            errors['diastolic_bp'] = "Diastolic BP must be between 40 and 150 mmHg"
    
    return errors

def calculate_derived_features(patient_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate derived features that might be useful for the model
    """
    derived = {}
    
    # BMI if height and weight are provided
    if 'height' in patient_data and 'weight' in patient_data:
        height_m = patient_data['height'] / 100  # Convert cm to m
        bmi = patient_data['weight'] / (height_m ** 2)
        derived['bmi'] = bmi
    
    # Pulse pressure
    if 'systolic_bp' in patient_data and 'diastolic_bp' in patient_data:
        derived['pulse_pressure'] = patient_data['systolic_bp'] - patient_data['diastolic_bp']
    
    # eGFR (estimated Glomerular Filtration Rate) using MDRD equation
    if all(k in patient_data for k in ['creatinine', 'age', 'gender']):
        creatinine = patient_data['creatinine']
        age = patient_data['age']
        gender = patient_data['gender']
        
        # MDRD equation
        egfr = 175 * (creatinine ** -1.154) * (age ** -0.203)
        if gender == 0:  # Female
            egfr *= 0.742
        
        derived['egfr'] = egfr
    
    return derived