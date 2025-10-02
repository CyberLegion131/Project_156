from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Dict
from sqlalchemy.orm import Session
import logging

from app.models.inference import DNPredictor, get_predictor
from app.models.preprocess import preprocess_patient_data
from app.db.schemas import PatientInput, PredictionResponse, PatientHistory
from app.db.database import get_db
from app.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger()

@router.post("/predict", response_model=PredictionResponse)
async def predict_dn_risk(
    patient_data: PatientInput,
    predictor: DNPredictor = Depends(get_predictor),
    db: Session = Depends(get_db)
):
    """
    🔧 Enhanced prediction of diabetic nephropathy risk for a patient
    """
    try:
        logger.info(f"Received prediction request for patient")
        
        # Get original patient data for clinical rules
        original_patient_data = patient_data.dict()
        
        # Preprocess the input data
        processed_data = preprocess_patient_data(original_patient_data)
        
        # 🔧 Make enhanced prediction with clinical rules
        risk_score, risk_level, confidence, model_accuracy, binary_prediction, clinical_details = \
            await predictor.predict(processed_data, original_patient_data)
        
        # Create enhanced response
        response = PredictionResponse(
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            model_accuracy=model_accuracy,
            binary_prediction=binary_prediction,
            recommendations=get_recommendations(risk_level, clinical_details),
            clinical_details=clinical_details  # Add clinical enhancement details
        )
        
        # Save to database (optional)
        try:
            history_entry = PatientHistory(
                patient_id=patient_data.patient_id,
                # Demographics & Physical
                sex=patient_data.sex,
                age=patient_data.age,
                height_cm=patient_data.height_cm,
                weight_kg=patient_data.weight_kg,
                bmi=patient_data.bmi,
                # Diabetes History
                diabetes_duration_y=patient_data.diabetes_duration_y,
                diabetic_retinopathy_dr=patient_data.diabetic_retinopathy_dr,
                # Lifestyle
                smoking=patient_data.smoking,
                drinking=patient_data.drinking,
                # Vital Signs
                sbp_mmhg=patient_data.sbp_mmhg,
                dbp_mmhg=patient_data.dbp_mmhg,
                # Laboratory Values
                hba1c_percent=patient_data.hba1c_percent,
                fbg_mmol_l=patient_data.fbg_mmol_l,
                tg_mmol_l=patient_data.tg_mmol_l,
                c_peptide_ng_ml=patient_data.c_peptide_ng_ml,
                tc_mmol_l=patient_data.tc_mmol_l,
                hdlc_mmol_l=patient_data.hdlc_mmol_l,
                ldlc_mmol_l=patient_data.ldlc_mmol_l,
                # Medications
                insulin=patient_data.insulin,
                metformin=patient_data.metformin,
                lipid_lowering_drugs=patient_data.lipid_lowering_drugs,
                # Prediction Results
                risk_score=risk_score,
                risk_level=risk_level,
                confidence=confidence
            )
            db.add(history_entry)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to save prediction history: {str(e)}")
        
        logger.info(f"Prediction completed: risk_level={risk_level}, score={risk_score}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/history/{patient_id}", response_model=List[PredictionResponse])
async def get_patient_history(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Get prediction history for a patient
    """
    try:
        history = db.query(PatientHistory).filter(
            PatientHistory.patient_id == patient_id
        ).order_by(PatientHistory.created_at.desc()).limit(10).all()
        
        return [
            PredictionResponse(
                risk_score=h.risk_score,
                risk_level=h.risk_level,
                confidence=h.confidence,
                recommendations=get_recommendations(h.risk_level),
                timestamp=h.created_at
            ) for h in history
        ]
    except Exception as e:
        logger.error(f"Failed to fetch patient history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch patient history")

# Authentication removed - direct app access

@router.get("/model/info")
async def get_model_info(predictor: DNPredictor = Depends(get_predictor)):
    """
    Get information about the loaded model
    """
    try:
        info = await predictor.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

def get_recommendations(risk_level: str, clinical_details: Dict = None) -> List[str]:
    """
    🔧 Generate enhanced recommendations based on risk level and clinical details
    """
    base_recommendations = {
        "Low": [
            "Continue regular diabetes management",
            "Monitor blood glucose levels regularly", 
            "Maintain healthy diet and exercise routine",
            "Schedule annual kidney function tests"
        ],
        "Medium": [
            "Increase frequency of kidney function monitoring",
            "Consider ACE inhibitors or ARBs if not contraindicated",
            "Strict blood glucose control (HbA1c < 7%)",
            "Blood pressure control (<130/80 mmHg)",
            "Dietary protein restriction if needed"
        ],
        "High": [
            "⚠️ IMMEDIATE nephrology consultation recommended",
            "Intensive diabetes management required",
            "Strict blood pressure control essential",
            "Consider renal protective medications",
            "Regular monitoring of kidney function",
            "Prepare for potential renal replacement therapy"
        ]
    }
    
    recommendations = base_recommendations.get(risk_level, ["Consult with healthcare provider"])
    
    # 🔧 Add specific recommendations based on clinical risk factors
    if clinical_details and clinical_details.get('risk_factors_detected'):
        risk_factors = clinical_details['risk_factors_detected']
        
        # Add targeted recommendations based on detected risk factors
        if any('retinopathy' in factor.lower() for factor in risk_factors):
            recommendations.append("🔍 Ophthalmology follow-up for diabetic retinopathy management")
            
        if any('duration' in factor.lower() for factor in risk_factors):
            recommendations.append("📅 Enhanced diabetes monitoring due to long disease duration")
            
        if any('hba1c' in factor.lower() for factor in risk_factors):
            recommendations.append("🎯 URGENT: Intensive glycemic control - target HbA1c <7%")
            
        if any('hypertension' in factor.lower() for factor in risk_factors):
            recommendations.append("💊 Antihypertensive therapy optimization required")
            
        # Add clinical override explanation
        if clinical_details.get('clinical_override'):
            recommendations.insert(0, f"🚨 CLINICAL ALERT: {clinical_details.get('explanation', 'High-risk combination detected')}")
    
    return recommendations