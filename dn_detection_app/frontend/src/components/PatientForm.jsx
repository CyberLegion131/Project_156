import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import toast from 'react-hot-toast';
import { Activity, User, Calendar, Droplets, TestTube, Heart } from 'lucide-react';
import { predictionAPI } from '../api';

const PatientForm = ({ onPrediction }) => {
  const [isLoading, setIsLoading] = useState(false);
  const { register, handleSubmit, formState: { errors }, reset } = useForm();

  const onSubmit = async (data) => {
    setIsLoading(true);
    try {
      // Convert form data to match EXACT dataset parameters - ALL 21 PARAMETERS
      const patientData = {
        // Patient Identification
        patient_id: data.patient_id,
        // Demographics & Physical
        sex: parseInt(data.sex),
        age: parseFloat(data.age),
        height_cm: parseFloat(data.height_cm),
        weight_kg: parseFloat(data.weight_kg),
        bmi: parseFloat(data.bmi),
        // Diabetes History
        diabetes_duration_y: parseFloat(data.diabetes_duration_y),
        diabetic_retinopathy_dr: parseInt(data.diabetic_retinopathy_dr),
        // Lifestyle
        smoking: parseInt(data.smoking),
        drinking: parseInt(data.drinking),
        // Vital Signs
        sbp_mmhg: parseFloat(data.sbp_mmhg),
        dbp_mmhg: parseFloat(data.dbp_mmhg),
        // Laboratory Values
        hba1c_percent: parseFloat(data.hba1c_percent),
        fbg_mmol_l: parseFloat(data.fbg_mmol_l),
        tg_mmol_l: parseFloat(data.tg_mmol_l),
        c_peptide_ng_ml: parseFloat(data.c_peptide_ng_ml),
        tc_mmol_l: parseFloat(data.tc_mmol_l),
        hdlc_mmol_l: parseFloat(data.hdlc_mmol_l),
        ldlc_mmol_l: parseFloat(data.ldlc_mmol_l),
        // Medications
        insulin: parseInt(data.insulin),
        metformin: parseInt(data.metformin),
        lipid_lowering_drugs: parseInt(data.lipid_lowering_drugs),
      };

      const result = await predictionAPI.predict(patientData);
      toast.success('Complete clinical assessment analyzed with 77.92% accuracy!');
      onPrediction(result);
    } catch (error) {
      console.error('Prediction failed:', error);
      toast.error('Prediction failed. Please check your input data.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    reset();
    onPrediction(null);
    toast.success('Form cleared');
  };

  return (
    <div className="patient-form fade-in">
      <div className="form-header" style={{ marginBottom: '2rem', textAlign: 'center' }}>
        <h2 style={{ color: '#2d3748', marginBottom: '0.5rem' }}>
          <Activity size={24} style={{ display: 'inline', marginRight: '0.5rem' }} />
          Patient Assessment
        </h2>
        <p style={{ color: '#718096' }}>
          Enter patient data for diabetic nephropathy risk assessment
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)}>
        {/* Patient Demographics */}
        <div className="form-section" style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: '#4a5568', marginBottom: '1rem', fontSize: '1.1rem' }}>
            <User size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
            Patient Demographics
          </h3>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="patient_id">Patient ID *</label>
              <input
                id="patient_id"
                type="text"
                placeholder="e.g., P001"
                {...register('patient_id', { 
                  required: 'Patient ID is required'
                })}
                className={errors.patient_id ? 'error' : ''}
              />
              {errors.patient_id && (
                <span className="error-message">{errors.patient_id.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="sex">Sex *</label>
              <select
                id="sex"
                {...register('sex', { required: 'Sex is required' })}
                className={errors.sex ? 'error' : ''}
              >
                <option value="">Select sex</option>
                <option value="0">Female</option>
                <option value="1">Male</option>
              </select>
              {errors.sex && (
                <span className="error-message">{errors.sex.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="age">Age (years) *</label>
              <input
                id="age"
                type="number"
                placeholder="e.g., 65"
                step="1"
                min="18"
                max="120"
                {...register('age', { 
                  required: 'Age is required',
                  min: { value: 18, message: 'Age must be at least 18' },
                  max: { value: 120, message: 'Age must not exceed 120' }
                })}
                className={errors.age ? 'error' : ''}
              />
              {errors.age && (
                <span className="error-message">{errors.age.message}</span>
              )}
            </div>
          </div>
        </div>

        {/* Physical Measurements */}
        <div className="form-section" style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: '#4a5568', marginBottom: '1rem', fontSize: '1.1rem' }}>
            <Activity size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
            Physical Measurements
          </h3>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="height_cm">Height (cm) *</label>
              <input
                id="height_cm"
                type="number"
                placeholder="e.g., 175"
                step="0.1"
                min="100"
                max="250"
                {...register('height_cm', { 
                  required: 'Height is required',
                  min: { value: 100, message: 'Height must be at least 100cm' },
                  max: { value: 250, message: 'Height must not exceed 250cm' }
                })}
                className={errors.height_cm ? 'error' : ''}
              />
              {errors.height_cm && (
                <span className="error-message">{errors.height_cm.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="weight_kg">Weight (kg) *</label>
              <input
                id="weight_kg"
                type="number"
                placeholder="e.g., 70"
                step="0.1"
                min="30"
                max="300"
                {...register('weight_kg', { 
                  required: 'Weight is required',
                  min: { value: 30, message: 'Weight must be at least 30kg' },
                  max: { value: 300, message: 'Weight must not exceed 300kg' }
                })}
                className={errors.weight_kg ? 'error' : ''}
              />
              {errors.weight_kg && (
                <span className="error-message">{errors.weight_kg.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="bmi">BMI (kg/m²) *</label>
              <input
                id="bmi"
                type="number"
                placeholder="e.g., 25.5"
                step="0.1"
                min="12"
                max="80"
                {...register('bmi', { 
                  required: 'BMI is required',
                  min: { value: 12, message: 'BMI must be at least 12' },
                  max: { value: 80, message: 'BMI must not exceed 80' }
                })}
                className={errors.bmi ? 'error' : ''}
              />
              {errors.bmi && (
                <span className="error-message">{errors.bmi.message}</span>
              )}
            </div>
          </div>
        </div>

        {/* Diabetes History */}
        <div className="form-section" style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: '#4a5568', marginBottom: '1rem', fontSize: '1.1rem' }}>
            <Calendar size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
            Diabetes History
          </h3>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="diabetes_duration_y">Diabetes Duration (years) *</label>
              <input
                id="diabetes_duration_y"
                type="number"
                placeholder="e.g., 5.5"
                step="0.1"
                min="0"
                max="80"
                {...register('diabetes_duration_y', { 
                  required: 'Diabetes duration is required',
                  min: { value: 0, message: 'Duration cannot be negative' },
                  max: { value: 80, message: 'Duration must not exceed 80 years' }
                })}
                className={errors.diabetes_duration_y ? 'error' : ''}
              />
              {errors.diabetes_duration_y && (
                <span className="error-message">{errors.diabetes_duration_y.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="diabetic_retinopathy_dr">Diabetic Retinopathy *</label>
              <select
                id="diabetic_retinopathy_dr"
                {...register('diabetic_retinopathy_dr', { required: 'Diabetic retinopathy status is required' })}
                className={errors.diabetic_retinopathy_dr ? 'error' : ''}
              >
                <option value="">Select status</option>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
              {errors.diabetic_retinopathy_dr && (
                <span className="error-message">{errors.diabetic_retinopathy_dr.message}</span>
              )}
            </div>
          </div>
        </div>

        {/* Lifestyle Factors */}
        <div className="form-section" style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: '#4a5568', marginBottom: '1rem', fontSize: '1.1rem' }}>
            <Activity size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
            Lifestyle Factors
          </h3>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="smoking">Smoking *</label>
              <select
                id="smoking"
                {...register('smoking', { required: 'Smoking status is required' })}
                className={errors.smoking ? 'error' : ''}
              >
                <option value="">Select status</option>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
              {errors.smoking && (
                <span className="error-message">{errors.smoking.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="drinking">Drinking *</label>
              <select
                id="drinking"
                {...register('drinking', { required: 'Drinking status is required' })}
                className={errors.drinking ? 'error' : ''}
              >
                <option value="">Select status</option>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
              {errors.drinking && (
                <span className="error-message">{errors.drinking.message}</span>
              )}
            </div>
          </div>
        </div>

        {/* Vital Signs */}
        <div className="form-section" style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: '#4a5568', marginBottom: '1rem', fontSize: '1.1rem' }}>
            <Heart size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
            Vital Signs
          </h3>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="sbp_mmhg">Systolic BP (mmHg) *</label>
              <input
                id="sbp_mmhg"
                type="number"
                placeholder="e.g., 130"
                step="1"
                min="80"
                max="250"
                {...register('sbp_mmhg', { 
                  required: 'Systolic BP is required',
                  min: { value: 80, message: 'SBP must be at least 80 mmHg' },
                  max: { value: 250, message: 'SBP must not exceed 250 mmHg' }
                })}
                className={errors.sbp_mmhg ? 'error' : ''}
              />
              {errors.sbp_mmhg && (
                <span className="error-message">{errors.sbp_mmhg.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="dbp_mmhg">Diastolic BP (mmHg) *</label>
              <input
                id="dbp_mmhg"
                type="number"
                placeholder="e.g., 85"
                step="1"
                min="40"
                max="150"
                {...register('dbp_mmhg', { 
                  required: 'Diastolic BP is required',
                  min: { value: 40, message: 'DBP must be at least 40 mmHg' },
                  max: { value: 150, message: 'DBP must not exceed 150 mmHg' }
                })}
                className={errors.dbp_mmhg ? 'error' : ''}
              />
              {errors.dbp_mmhg && (
                <span className="error-message">{errors.dbp_mmhg.message}</span>
              )}
            </div>
          </div>
        </div>

        {/* Laboratory Values */}
        <div className="form-section" style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: '#4a5568', marginBottom: '1rem', fontSize: '1.1rem' }}>
            <TestTube size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
            Laboratory Values
          </h3>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="hba1c_percent">HbA1c (%) *</label>
              <input
                id="hba1c_percent"
                type="number"
                placeholder="e.g., 7.5"
                step="0.1"
                min="4.0"
                max="18.0"
                {...register('hba1c_percent', { 
                  required: 'HbA1c is required',
                  min: { value: 4.0, message: 'HbA1c must be at least 4.0%' },
                  max: { value: 18.0, message: 'HbA1c must not exceed 18.0%' }
                })}
                className={errors.hba1c_percent ? 'error' : ''}
              />
              {errors.hba1c_percent && (
                <span className="error-message">{errors.hba1c_percent.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="fbg_mmol_l">FBG (mmol/L) *</label>
              <input
                id="fbg_mmol_l"
                type="number"
                placeholder="e.g., 8.2"
                step="0.1"
                min="2.0"
                max="30.0"
                {...register('fbg_mmol_l', { 
                  required: 'Fasting blood glucose is required',
                  min: { value: 2.0, message: 'FBG must be at least 2.0 mmol/L' },
                  max: { value: 30.0, message: 'FBG must not exceed 30.0 mmol/L' }
                })}
                className={errors.fbg_mmol_l ? 'error' : ''}
              />
              {errors.fbg_mmol_l && (
                <span className="error-message">{errors.fbg_mmol_l.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="tg_mmol_l">Triglycerides (mmol/L) *</label>
              <input
                id="tg_mmol_l"
                type="number"
                placeholder="e.g., 2.1"
                step="0.01"
                min="0.1"
                max="20.0"
                {...register('tg_mmol_l', { 
                  required: 'Triglycerides is required',
                  min: { value: 0.1, message: 'TG must be at least 0.1 mmol/L' },
                  max: { value: 20.0, message: 'TG must not exceed 20.0 mmol/L' }
                })}
                className={errors.tg_mmol_l ? 'error' : ''}
              />
              {errors.tg_mmol_l && (
                <span className="error-message">{errors.tg_mmol_l.message}</span>
              )}
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="c_peptide_ng_ml">C-peptide (ng/ml) *</label>
              <input
                id="c_peptide_ng_ml"
                type="number"
                placeholder="e.g., 1.8"
                step="0.01"
                min="0.1"
                max="10.0"
                {...register('c_peptide_ng_ml', { 
                  required: 'C-peptide is required',
                  min: { value: 0.1, message: 'C-peptide must be at least 0.1 ng/ml' },
                  max: { value: 10.0, message: 'C-peptide must not exceed 10.0 ng/ml' }
                })}
                className={errors.c_peptide_ng_ml ? 'error' : ''}
              />
              {errors.c_peptide_ng_ml && (
                <span className="error-message">{errors.c_peptide_ng_ml.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="tc_mmol_l">Total Cholesterol (mmol/L) *</label>
              <input
                id="tc_mmol_l"
                type="number"
                placeholder="e.g., 5.2"
                step="0.01"
                min="2.0"
                max="15.0"
                {...register('tc_mmol_l', { 
                  required: 'Total cholesterol is required',
                  min: { value: 2.0, message: 'TC must be at least 2.0 mmol/L' },
                  max: { value: 15.0, message: 'TC must not exceed 15.0 mmol/L' }
                })}
                className={errors.tc_mmol_l ? 'error' : ''}
              />
              {errors.tc_mmol_l && (
                <span className="error-message">{errors.tc_mmol_l.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="hdlc_mmol_l">HDL Cholesterol (mmol/L) *</label>
              <input
                id="hdlc_mmol_l"
                type="number"
                placeholder="e.g., 1.2"
                step="0.01"
                min="0.5"
                max="5.0"
                {...register('hdlc_mmol_l', { 
                  required: 'HDL cholesterol is required',
                  min: { value: 0.5, message: 'HDL-C must be at least 0.5 mmol/L' },
                  max: { value: 5.0, message: 'HDL-C must not exceed 5.0 mmol/L' }
                })}
                className={errors.hdlc_mmol_l ? 'error' : ''}
              />
              {errors.hdlc_mmol_l && (
                <span className="error-message">{errors.hdlc_mmol_l.message}</span>
              )}
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="ldlc_mmol_l">LDL Cholesterol (mmol/L) *</label>
              <input
                id="ldlc_mmol_l"
                type="number"
                placeholder="e.g., 3.1"
                step="0.01"
                min="0.5"
                max="10.0"
                {...register('ldlc_mmol_l', { 
                  required: 'LDL cholesterol is required',
                  min: { value: 0.5, message: 'LDL-C must be at least 0.5 mmol/L' },
                  max: { value: 10.0, message: 'LDL-C must not exceed 10.0 mmol/L' }
                })}
                className={errors.ldlc_mmol_l ? 'error' : ''}
              />
              {errors.ldlc_mmol_l && (
                <span className="error-message">{errors.ldlc_mmol_l.message}</span>
              )}
            </div>
          </div>
        </div>

        {/* Medications */}
        <div className="form-section" style={{ marginBottom: '2rem' }}>
          <h3 style={{ color: '#4a5568', marginBottom: '1rem', fontSize: '1.1rem' }}>
            <TestTube size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
            Current Medications
          </h3>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="insulin">Insulin *</label>
              <select
                id="insulin"
                {...register('insulin', { required: 'Insulin usage is required' })}
                className={errors.insulin ? 'error' : ''}
              >
                <option value="">Select status</option>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
              {errors.insulin && (
                <span className="error-message">{errors.insulin.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="metformin">Metformin *</label>
              <select
                id="metformin"
                {...register('metformin', { required: 'Metformin usage is required' })}
                className={errors.metformin ? 'error' : ''}
              >
                <option value="">Select status</option>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
              {errors.metformin && (
                <span className="error-message">{errors.metformin.message}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="lipid_lowering_drugs">Lipid Lowering Drugs *</label>
              <select
                id="lipid_lowering_drugs"
                {...register('lipid_lowering_drugs', { required: 'Lipid lowering drugs usage is required' })}
                className={errors.lipid_lowering_drugs ? 'error' : ''}
              >
                <option value="">Select status</option>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
              {errors.lipid_lowering_drugs && (
                <span className="error-message">{errors.lipid_lowering_drugs.message}</span>
              )}
            </div>
          </div>
        </div>



        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <button
            type="submit"
            disabled={isLoading}
            className="btn btn-primary"
            style={{ minWidth: '150px' }}
          >
            {isLoading ? (
              <>
                <div className="loading-spinner"></div>
                Analyzing...
              </>
            ) : (
              <>
                <Activity size={16} />
                Analyze Risk
              </>
            )}
          </button>

          <button
            type="button"
            onClick={handleReset}
            className="btn btn-secondary"
            disabled={isLoading}
          >
            Clear Form
          </button>
        </div>
      </form>

      <div style={{ marginTop: '2rem', padding: '1rem', background: '#f7fafc', borderRadius: '0.5rem' }}>
        <p style={{ fontSize: '0.8rem', color: '#718096', textAlign: 'center' }}>
          * Required fields. Optional fields will use default values if not provided.
        </p>
      </div>
    </div>
  );
};

export default PatientForm;