import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import joblib

# Import your prediction function
from disease_predictor import predict_disease

# Create FastAPI app
app = FastAPI(
    title="Veterinary Disease Predictor API",
    description="API for predicting animal diseases based on symptoms and metrics",
    version="1.0.0"
)

# Load all models and encoders
try:
    model = joblib.load('model.pkl')
    animal_type_encoder = joblib.load('animal_type_encoder.pkl')
    breed_encoder = joblib.load('breed_encoder.pkl')
    age_group_encoder = joblib.load('age_group_encoder.pkl')
    weight_category_encoder = joblib.load('weight_category_encoder.pkl')
    disease_prediction_encoder = joblib.load('disease_prediction_encoder.pkl')
    print("Models loaded successfully")
except Exception as e:
    import os
    print(f"Error loading models: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")

# Define request model
class PredictionRequest(BaseModel):
    animal_type: str
    breed: str
    age: float
    gender: str
    weight: float
    symptoms: List[str]
    duration: str
    appetite_loss: Optional[int] = 0
    vomiting: Optional[int] = 0
    diarrhea: Optional[int] = 0
    coughing: Optional[int] = 0
    labored_breathing: Optional[int] = 0
    lameness: Optional[int] = 0
    skin_lesions: Optional[int] = 0
    nasal_discharge: Optional[int] = 0
    eye_discharge: Optional[int] = 0
    body_temp: Optional[float] = 0
    heart_rate: Optional[int] = 0

# Define response model
class Disease(BaseModel):
    disease: str
    probability: str

class PredictionResponse(BaseModel):
    predicted_disease: str
    top_diseases: List[Disease]

# Helper functions
def filter_diseases(probabilities, exclude_category='Other'):
    """Filter out the 'Other' category and return top diseases"""
    filtered_probs = {k: v for k, v in probabilities.items() if k != exclude_category}
    if not filtered_probs:  # If all diseases were filtered out
        return list(probabilities.items())[:2]  # Return top 2 from original
    
    # Get top 2 diseases from filtered list
    top_diseases = sorted(filtered_probs.items(), key=lambda x: x[1], reverse=True)[:2]
    return top_diseases

def format_percentage(value):
    """Convert probability to percentage with 1 decimal place"""
    return f"{value * 100:.1f}%"

# Endpoint for prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Make prediction using imported function
        disease, probabilities = predict_disease(
            request.animal_type, 
            request.breed, 
            request.age, 
            request.gender, 
            request.weight, 
            request.symptoms, 
            request.duration,
            request.appetite_loss, 
            request.vomiting, 
            request.diarrhea, 
            request.coughing, 
            request.labored_breathing,
            request.lameness, 
            request.skin_lesions, 
            request.nasal_discharge, 
            request.eye_discharge,
            request.body_temp, 
            request.heart_rate
        )
        
        # Filter out 'Other' and get top 2 diseases
        top_diseases = filter_diseases(probabilities)
        
        # If the original top disease was 'Other', replace it with the highest non-Other disease
        if disease == "Other" and top_diseases:
            disease = top_diseases[0][0]
        
        # Format the response
        response = PredictionResponse(
            predicted_disease=f"{disease} ({format_percentage(probabilities[disease])})",
            top_diseases=[
                Disease(
                    disease=name,
                    probability=format_percentage(prob)
                ) for name, prob in top_diseases
            ]
        )
        
        return response
    except Exception as e:
        import traceback
        error_detail = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail) 
        raise HTTPException(status_code=500, detail=error_detail)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Veterinary Disease Predictor API",
        "usage": "Send POST request to /predict endpoint with animal details"
    }

@app.get("/models")
async def check_models():
    import os
    files = os.listdir('.')
    pkl_files = [f for f in files if f.endswith('.pkl')]
    return {
        "available_files": files,
        "pkl_files": pkl_files,
        "current_directory": os.getcwd()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Get available animal types endpoint
@app.get("/animal-types")
async def get_animal_types():
    return {"animal_types": list(animal_type_encoder.classes_)}

# Get available breeds endpoint
@app.get("/breeds")
async def get_breeds():
    return {"breeds": list(breed_encoder.classes_)}

# Main entry point
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
