from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

class PatientInput(BaseModel):
    """
    Complete input schema for patient data - EXACT 21 parameters from dataset
    """
    # Patient Identification
    patient_id: str = Field(..., description="Unique patient identifier")
    
    # Demographics & Physical
    sex: int = Field(..., ge=0, le=1, description="Sex: 0=Female, 1=Male")
    age: float = Field(..., ge=0, le=120, description="Age in years")
    height_cm: float = Field(..., ge=100, le=250, description="Height in cm")
    weight_kg: float = Field(..., ge=30, le=200, description="Weight in kg")
    bmi: float = Field(..., ge=15.0, le=50.0, description="BMI (kg/m²)")
    
    # Diabetes History
    diabetes_duration_y: float = Field(..., ge=0, le=50, description="Diabetes duration (years)")
    diabetic_retinopathy_dr: int = Field(..., ge=0, le=1, description="Diabetic retinopathy (DR): 0=No, 1=Yes")
    
    # Lifestyle
    smoking: int = Field(..., ge=0, le=1, description="Smoking: 0=No, 1=Yes")
    drinking: int = Field(..., ge=0, le=1, description="Drinking: 0=No, 1=Yes")
    
    # Vital Signs
    sbp_mmhg: float = Field(..., ge=80, le=250, description="SBP (mmHg)")
    dbp_mmhg: float = Field(..., ge=40, le=150, description="DBP (mmHg)")
    
    # Laboratory Values
    hba1c_percent: float = Field(..., ge=4.0, le=15.0, description="HbA1c (%)")
    fbg_mmol_l: float = Field(..., ge=2.0, le=30.0, description="FBG (mmol/L)")
    tg_mmol_l: float = Field(..., ge=0.3, le=20.0, description="TG (mmol/L)")
    c_peptide_ng_ml: float = Field(..., ge=0.1, le=10.0, description="C-peptide (ng/ml)")
    tc_mmol_l: float = Field(..., ge=2.0, le=15.0, description="TC (mmol/L)")
    hdlc_mmol_l: float = Field(..., ge=0.5, le=3.0, description="HDLC (mmol/L)")
    ldlc_mmol_l: float = Field(..., ge=1.0, le=10.0, description="LDLC (mmol/L)")
    
    # Medications
    insulin: int = Field(..., ge=0, le=1, description="Insulin: 0=No, 1=Yes")
    metformin: int = Field(..., ge=0, le=1, description="Metformin: 0=No, 1=Yes")
    lipid_lowering_drugs: int = Field(..., ge=0, le=1, description="Lipid lowering drugs: 0=No, 1=Yes")
    
    @validator('sex', 'diabetic_retinopathy_dr', 'smoking', 'drinking', 
              'insulin', 'metformin', 'lipid_lowering_drugs')
    def validate_binary_fields(cls, v):
        if v not in [0, 1]:
            raise ValueError('Binary fields must be 0 or 1')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P001",
                "sex": 1,
                "age": 65,
                "height_cm": 170,
                "weight_kg": 80,
                "bmi": 28.5,
                "diabetes_duration_y": 10,
                "diabetic_retinopathy_dr": 1,
                "smoking": 0,
                "drinking": 0,
                "sbp_mmhg": 145,
                "dbp_mmhg": 90,
                "hba1c_percent": 8.5,
                "fbg_mmol_l": 10.0,
                "tg_mmol_l": 2.5,
                "c_peptide_ng_ml": 1.8,
                "tc_mmol_l": 5.2,
                "hdlc_mmol_l": 1.1,
                "ldlc_mmol_l": 3.2,
                "insulin": 1,
                "metformin": 1,
                "lipid_lowering_drugs": 0
            }
        }

class PredictionResponse(BaseModel):
    """
    🔧 Enhanced response schema for DN prediction with clinical details
    """
    risk_score: float = Field(..., description="Risk score (0-100)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")
    model_accuracy: str = Field(..., description="ML model accuracy percentage")
    binary_prediction: int = Field(..., ge=0, le=1, description="Binary ML prediction: 0=No DN, 1=Has DN")
    recommendations: List[str] = Field(..., description="Clinical recommendations")
    clinical_details: Optional[dict] = Field(default=None, description="Enhanced clinical analysis details")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "risk_score": 77.5,
                "risk_level": "High",
                "confidence": 0.87,
                "model_accuracy": "78.57%",
                "binary_prediction": 1,
                "recommendations": [
                    "🚨 CLINICAL ALERT: Multiple severe risk factors: Has diabetic retinopathy, Very high HbA1c (14.1%)",
                    "⚠️ IMMEDIATE nephrology consultation recommended",
                    "🎯 URGENT: Intensive glycemic control - target HbA1c <7%",
                    "🔍 Ophthalmology follow-up for diabetic retinopathy management"
                ],
                "clinical_details": {
                    "original_model_score": 27.5,
                    "enhanced_score": 77.5,
                    "clinical_override": True,
                    "risk_factors_detected": ["Has diabetic retinopathy", "Very high HbA1c (14.1%)"],
                    "override_reason": "Multiple severe risk factors: Has diabetic retinopathy, Very high HbA1c (14.1%)",
                    "confidence_level": "HIGH",
                    "explanation": "Clinical Override: Multiple severe risk factors: Has diabetic retinopathy, Very high HbA1c (14.1%)"
                },
                "timestamp": "2024-01-01T10:00:00"
            }
        }

class PatientHistory(BaseModel):
    """
    Schema for patient history storage - EXACT 21 parameters from dataset
    """
    id: Optional[int] = None
    patient_id: str
    # Demographics & Physical
    sex: int
    age: float
    height_cm: float
    weight_kg: float
    bmi: float
    # Diabetes History
    diabetes_duration_y: float
    diabetic_retinopathy_dr: int
    # Lifestyle
    smoking: int
    drinking: int
    # Vital Signs
    sbp_mmhg: float
    dbp_mmhg: float
    # Laboratory Values
    hba1c_percent: float
    fbg_mmol_l: float
    tg_mmol_l: float
    c_peptide_ng_ml: float
    tc_mmol_l: float
    hdlc_mmol_l: float
    ldlc_mmol_l: float
    # Medications
    insulin: int
    metformin: int
    lipid_lowering_drugs: int
    # Prediction Results
    risk_score: float
    risk_level: str
    confidence: float
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    """
    Schema for login request
    """
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    
    class Config:
        schema_extra = {
            "example": {
                "username": "doctor1",
                "password": "secure_password"
            }
        }

class Token(BaseModel):
    """
    Schema for JWT token response
    """
    access_token: str
    token_type: str = "bearer"

class User(BaseModel):
    """
    User schema
    """
    id: Optional[int] = None
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    """
    Schema for creating new user
    """
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: Optional[str] = None
    full_name: Optional[str] = None