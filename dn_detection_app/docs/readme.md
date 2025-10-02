# 🏥 Diabetic Nephropathy Detection System

A comprehensive deep learning-based system for early detection and monitoring of diabetic nephropathy, featuring automated deployment pipelines and clinical decision support.

## 🚀 Features

- **Deep Learning Model**: Advanced neural network for DN risk assessment
- **Real-time Predictions**: Fast API for instant risk scoring
- **Clinical Dashboard**: Intuitive web interface for healthcare professionals
- **Automated CI/CD**: Continuous integration and deployment pipeline
- **Cloud-Ready**: Kubernetes and Docker support for scalable deployment
- **Comprehensive Monitoring**: Built-in health checks and performance metrics

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   ML Model      │
│   (React)       │────│   (FastAPI)     │────│   (ONNX)        │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │    │   Database      │    │  Training       │
│                 │    │   (SQLite/      │    │  Pipeline       │
│                 │    │   PostgreSQL)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠 Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: Database ORM
- **ONNX Runtime**: Model inference engine
- **Pydantic**: Data validation
- **JWT**: Authentication

### Frontend
- **React 18**: User interface framework
- **React Router**: Client-side routing
- **Recharts**: Data visualization
- **Axios**: HTTP client
- **React Hook Form**: Form management

### ML/AI
- **scikit-learn**: Machine learning algorithms
- **ONNX**: Cross-platform model format
- **Pandas/NumPy**: Data processing
- **Matplotlib/Seaborn**: Data visualization

### DevOps
- **Docker**: Containerization
- **Kubernetes**: Container orchestration
- **GitHub Actions**: CI/CD pipeline
- **Nginx**: Reverse proxy and load balancing

## 🚦 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (optional)
- Git

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dn_detection_app
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. **Train ML Model (Optional)**
   ```bash
   cd ml_model
   pip install -r requirements.txt
   python training_pipeline.py
   ```

### Using Docker Compose

```bash
docker-compose -f ci_cd/docker-compose.yml up -d
```

Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📋 API Documentation

### Authentication
```http
POST /api/v1/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}
```

### Prediction
```http
POST /api/v1/predict
Authorization: Bearer <token>
Content-Type: application/json

{
  "patient_id": "P001",
  "age": 65,
  "gender": 1,
  "glucose": 180,
  "hba1c": 8.5,
  "creatinine": 1.8,
  "urea": 45,
  "systolic_bp": 145,
  "diastolic_bp": 90
}
```

### Response
```json
{
  "risk_score": 75.5,
  "risk_level": "High",
  "confidence": 0.87,
  "recommendations": [
    "Immediate nephrology consultation recommended",
    "Intensive diabetes management required"
  ],
  "timestamp": "2024-01-01T10:00:00"
}
```

## 🔧 Configuration

### Environment Variables

**Backend**
```env
DATABASE_URL=sqlite:///./dn_detection.db
SECRET_KEY=your-secret-key-change-this
ACCESS_TOKEN_EXPIRE_MINUTES=30
LOG_DIR=logs
```

**Frontend**
```env
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_VERSION=1.0.0
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

### Frontend Tests
```bash
cd frontend
npm test -- --coverage
```

### ML Model Tests
```bash
cd ml_model
python -m pytest tests/ -v
```

## 📦 Deployment

### Kubernetes Deployment
```bash
kubectl apply -f ci_cd/k8s-deployment.yaml
```

### Docker Production Build
```bash
# Backend
docker build -t dn-detection-backend:latest backend/

# Frontend
docker build -t dn-detection-frontend:latest frontend/
```

## 📊 Model Information

### Input Features
- **Age**: Patient age in years (1-120)
- **Gender**: 0=Female, 1=Male
- **Glucose**: Blood glucose level (mg/dL, 50-500)
- **HbA1c**: Glycated hemoglobin percentage (4.0-15.0)
- **Creatinine**: Serum creatinine (mg/dL, 0.5-10.0)
- **Urea**: Blood urea nitrogen (mg/dL, 10-200, optional)
- **Systolic BP**: Systolic blood pressure (mmHg, 80-250, optional)
- **Diastolic BP**: Diastolic blood pressure (mmHg, 40-150, optional)

### Output
- **Risk Score**: 0-100% probability of diabetic nephropathy
- **Risk Level**: Low (0-33%), Medium (34-66%), High (67-100%)
- **Confidence**: Model confidence score (0-1)

### Model Performance
- **Accuracy**: ~85-90% on test set
- **Precision**: High precision for high-risk cases
- **Recall**: Optimized for early detection
- **F1-Score**: Balanced performance across risk levels

## 🔍 Monitoring and Observability

### Health Checks
- Backend: `GET /health`
- Model Status: `GET /api/v1/model/info`

### Metrics
- Request/response times
- Prediction accuracy
- System resource usage
- Error rates

### Logging
- Structured JSON logging
- Request/response logging
- Model prediction logging
- Error tracking

## 🛡 Security

### Authentication
- JWT-based authentication
- Role-based access control
- Session management

### Data Protection
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- HTTPS enforcement

### Privacy
- No PHI stored long-term
- Audit logging
- Data encryption at rest

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint for JavaScript/React
- Write comprehensive tests
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Troubleshooting

**Common Issues:**

1. **Model not loading**: Ensure ONNX model file exists in `ml_model/models/`
2. **Database connection**: Check DATABASE_URL configuration
3. **CORS errors**: Verify frontend/backend URL configuration
4. **Authentication issues**: Check JWT secret key and token expiration

**Getting Help:**
- Check the [Issues](../../issues) page
- Review the API documentation at `/docs`
- Contact the development team

### System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB
- Network: Stable internet connection

**Recommended:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 20GB+
- Network: High-speed internet

## 📈 Roadmap

- [ ] Advanced model architectures (LSTM, Transformer)
- [ ] Real-time data integration (EHR systems)
- [ ] Mobile application
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Federated learning capabilities

## 👥 Team

- **Development Team**: Full-stack and ML engineers
- **Clinical Advisors**: Nephrologists and endocrinologists
- **DevOps Engineers**: Infrastructure and deployment specialists

---

**Note**: This system is designed for clinical decision support and should be used in conjunction with professional medical judgment. It is not a substitute for comprehensive clinical evaluation.

For more detailed information, please refer to the individual component documentation in their respective directories.