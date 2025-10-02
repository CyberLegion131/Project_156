import React from 'react';
import { AlertTriangle, CheckCircle, AlertCircle, TrendingUp, Clock, X } from 'lucide-react';

const Dashboard = ({ predictionResult, onClearResult }) => {

  const getRiskColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'low':
        return '#48bb78';
      case 'medium':
        return '#ed8936';
      case 'high':
        return '#f56565';
      default:
        return '#718096';
    }
  };

  const getRiskIcon = (level) => {
    switch (level?.toLowerCase()) {
      case 'low':
        return <CheckCircle size={24} />;
      case 'medium':
        return <AlertCircle size={24} />;
      case 'high':
        return <AlertTriangle size={24} />;
      default:
        return <TrendingUp size={24} />;
    }
  };

  const getRiskClass = (level) => {
    switch (level?.toLowerCase()) {
      case 'low':
        return 'risk-low';
      case 'medium':
        return 'risk-medium';
      case 'high':
        return 'risk-high';
      default:
        return '';
    }
  };





  if (!predictionResult) {
    return (
      <div className="dashboard fade-in">
        <div className="dashboard-header">
          <h2>📊 Risk Assessment Dashboard</h2>
          <p>Results will appear here after patient assessment</p>
        </div>
        
        <div className="no-prediction">
          <TrendingUp size={48} style={{ color: '#cbd5e0', margin: '0 auto 1rem' }} />
          <h3 style={{ color: '#4a5568', marginBottom: '0.5rem' }}>No Assessment Data</h3>
          <p>Complete the patient form to view diabetic nephropathy risk analysis</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard fade-in">
      <div className="dashboard-header" style={{ position: 'relative' }}>
        <h2>📊 Risk Assessment Results</h2>
        <button 
          onClick={onClearResult}
          style={{
            position: 'absolute',
            top: 0,
            right: 0,
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: '#718096',
            padding: '0.5rem'
          }}
          title="Clear results"
        >
          <X size={20} />
        </button>
        <p style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', color: '#718096' }}>
          <Clock size={16} />
          Generated: {new Date(predictionResult.timestamp).toLocaleString()}
        </p>
      </div>

      <div className="prediction-result">
        {/* Prediction Results Card */}
        <div className={`risk-summary ${getRiskClass(predictionResult.risk_level)}`}>
          {/* Model Information Section */}
          <div style={{ 
            padding: '2rem',
            fontSize: '1rem'
          }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              marginBottom: '1.5rem',
              fontSize: '1.1rem'
            }}>
              <span>🤖 Model Accuracy:</span>
              <span style={{ fontWeight: '600' }}>
                {predictionResult.model_accuracy || 'Unknown'}
              </span>
            </div>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              fontSize: '1.1rem'
            }}>
              <span>🎯 Prediction:</span>
              <span style={{ 
                fontWeight: '600',
                color: predictionResult.binary_prediction === 1 ? '#e53e3e' : '#ffffff'
              }}>
                {predictionResult.binary_prediction === 1 ? 'Has DN' : 'No DN'}
              </span>
            </div>
          </div>
        </div>





        {/* Clinical Recommendations */}
        <div className="recommendations">
          <h3>🏥 Clinical Recommendations</h3>
          <ul>
            {predictionResult.recommendations?.map((recommendation, index) => (
              <li key={index}>
                <strong>#{index + 1}:</strong> {recommendation}
              </li>
            ))}
          </ul>
        </div>

        {/* Risk Interpretation */}
        <div style={{ background: '#edf2f7', borderRadius: '0.5rem', padding: '1.5rem' }}>
          <h3 style={{ color: '#2d3748', marginBottom: '1rem' }}>
            📋 Risk Level Interpretation
          </h3>
          
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <CheckCircle size={16} color="#48bb78" style={{ marginTop: '0.2rem', flexShrink: 0 }} />
              <div>
                <strong style={{ color: '#48bb78' }}>Low Risk (0-33%):</strong>
                <span style={{ color: '#4a5568', marginLeft: '0.5rem' }}>
                  Continue regular monitoring and diabetes management
                </span>
              </div>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <AlertCircle size={16} color="#ed8936" style={{ marginTop: '0.2rem', flexShrink: 0 }} />
              <div>
                <strong style={{ color: '#ed8936' }}>Medium Risk (34-66%):</strong>
                <span style={{ color: '#4a5568', marginLeft: '0.5rem' }}>
                  Enhanced monitoring and preventive measures recommended
                </span>
              </div>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <AlertTriangle size={16} color="#f56565" style={{ marginTop: '0.2rem', flexShrink: 0 }} />
              <div>
                <strong style={{ color: '#f56565' }}>High Risk (67-100%):</strong>
                <span style={{ color: '#4a5568', marginLeft: '0.5rem' }}>
                  Immediate clinical attention and specialist referral needed
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Model Information */}
        <div 
          style={{ 
            background: '#f7fafc', 
            borderRadius: '0.5rem', 
            padding: '1rem', 
            border: '1px solid #e2e8f0',
            fontSize: '0.85rem',
            color: '#718096'
          }}
        >
          <p style={{ textAlign: 'center' }}>
            ⚠️ This assessment is for clinical decision support only. 
            Results should be interpreted by qualified healthcare professionals 
            in conjunction with complete clinical evaluation.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;