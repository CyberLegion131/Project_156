import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import PatientForm from './components/PatientForm';
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  const [predictionResult, setPredictionResult] = useState(null);

  const handlePrediction = (result) => {
    setPredictionResult(result);
  };

  return (
    <Router>
      <div className="App">
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
          }}
        />
        
        <header className="app-header">
          <div className="header-content">
            <h1>🏥 Diabetic Nephropathy Detection System</h1>
            <p className="header-subtitle">AI-Powered Early Detection & Risk Assessment</p>
          </div>
        </header>

        <main className="app-main">
          <Routes>
            <Route 
              path="/" 
              element={
                <div className="main-content">
                  <div className="content-grid">
                    <div className="form-section">
                      <PatientForm onPrediction={handlePrediction} />
                    </div>
                    <div className="dashboard-section">
                      <Dashboard 
                        predictionResult={predictionResult}
                        onClearResult={() => setPredictionResult(null)}
                      />
                    </div>
                  </div>
                </div>
              } 
            />
          </Routes>
        </main>

        <footer className="app-footer">
          <p>&copy; 2024 DN Detection System. Advanced healthcare through AI.</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;