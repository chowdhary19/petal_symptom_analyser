# Veterinary Disease Predictor API

A FastAPI-based API for predicting animal diseases based on symptoms and metrics.

## Setup

1. Clone the repository
2. Download the model files (see below)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the API: `uvicorn app:app --host 0.0.0.0 --port 8000`

## Model Files

The application requires the following model files (.pkl) to function:
- model.pkl
- animal_type_encoder.pkl
- breed_encoder.pkl
- age_group_encoder.pkl
- weight_category_encoder.pkl
- disease_prediction_encoder.pkl

Download these files from: [Release Link] and place them in the root directory.

## API Usage

Send POST requests to `/predict` endpoint with JSON payload:

```json
{
  "animal_type": "Dog",
  "breed": "Labrador",
  "age": 5,
  "gender": "Male",
  "weight": 30.5,
  "symptoms": ["Coughing", "Lethargy"],
  "duration": "7 days",
  "body_temp": 39.5,
  "heart_rate": 120
}