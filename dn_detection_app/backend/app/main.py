from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn
from contextlib import asynccontextmanager

from app.routes import router
from app.models.inference import DNPredictor
from app.utils.logger import setup_logger
from app.db.database import init_db

predictor = None
logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
   
    global predictor
    try:
        predictor = DNPredictor()
        try:
            await predictor.load_model()
            logger.info("DN Predictor model loaded successfully")
        except Exception as model_error:
            logger.warning(f"Model loading failed, running without model: {str(model_error)}")
            predictor = DNPredictor()  # Initialize without loading model
        
        
        init_db()
        logger.info("Database initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        # Don't raise - let the app start even if there are issues
        predictor = DNPredictor()  # Fallback predictor
        logger.warning("Running with fallback configuration")
    
    yield
    
   
    logger.info("Application shutting down")

app = FastAPI(
    title="Diabetic Nephropathy Detection API",
    description="Deep Learning-based system for early detection of Diabetic Nephropathy",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Diabetic Nephropathy Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )