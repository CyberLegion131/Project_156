from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database configuration
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dn_detection.db")

# Create engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

class PatientHistory(Base):
    """
    Database model for patient prediction history - EXACT 21 parameters from dataset
    """
    __tablename__ = "patient_history"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    # Demographics & Physical
    sex = Column(Integer)
    age = Column(Float)
    height_cm = Column(Float)
    weight_kg = Column(Float)
    bmi = Column(Float)
    # Diabetes History
    diabetes_duration_y = Column(Float)
    diabetic_retinopathy_dr = Column(Integer)
    # Lifestyle
    smoking = Column(Integer)
    drinking = Column(Integer)
    # Vital Signs
    sbp_mmhg = Column(Float)
    dbp_mmhg = Column(Float)
    # Laboratory Values
    hba1c_percent = Column(Float)
    fbg_mmol_l = Column(Float)
    tg_mmol_l = Column(Float)
    c_peptide_ng_ml = Column(Float)
    tc_mmol_l = Column(Float)
    hdlc_mmol_l = Column(Float)
    ldlc_mmol_l = Column(Float)
    # Medications
    insulin = Column(Integer)
    metformin = Column(Integer)
    lipid_lowering_drugs = Column(Integer)
    # Prediction Results
    risk_score = Column(Float)
    risk_level = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class User(Base):
    """
    Database model for users (clinicians)
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    """
    Initialize database tables
    """
    Base.metadata.create_all(bind=engine)
    
    # Create default user if not exists
    from sqlalchemy.orm import Session
    from passlib.context import CryptContext
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    db = SessionLocal()
    
    try:
        # Check if admin user exists
        existing_user = db.query(User).filter(User.username == "admin").first()
        if not existing_user:
            # Create default admin user with a simple password
            simple_password = "admin"
            admin_user = User(
                username="admin",
                email="admin@dndetection.com",
                full_name="System Administrator",
                hashed_password=pwd_context.hash(simple_password),
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            print("Default admin user created (username: admin, password: admin)")
    except Exception as e:
        print(f"Error creating default user: {e}")
    finally:
        db.close()

def get_db():
    """
    Dependency to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()