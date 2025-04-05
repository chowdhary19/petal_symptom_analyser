import joblib
import pandas as pd
import numpy as np

def predict_disease(animal_type, breed, age, gender, weight, symptoms, duration, 
                   appetite_loss=0, vomiting=0, diarrhea=0, coughing=0, 
                   labored_breathing=0, lameness=0, skin_lesions=0, 
                   nasal_discharge=0, eye_discharge=0, body_temp=0, heart_rate=0):
    """
    Predict animal disease based on input parameters
    
    Parameters:
    -----------
    animal_type: str - Type of animal (Dog, Cat, Cow, etc.)
    breed: str - Breed of the animal
    age: int - Age in years
    gender: str - Male or Female
    weight: float - Weight in appropriate units
    symptoms: list - List of symptoms [symptom1, symptom2, symptom3, symptom4]
                     Pass None or empty string for no symptom
    duration: str - Duration of symptoms (e.g., "3 days", "1 week")
    appetite_loss, vomiting, etc.: Binary indicators (0 or 1)
    body_temp: float - Body temperature
    heart_rate: int - Heart rate
    
    Returns:
    --------
    predicted_disease: str - The predicted disease
    probabilities: dict - Probability for each disease class
    """
    # Load the model and encoders
    model = joblib.load('model.pkl')
    animal_type_encoder = joblib.load('animal_type_encoder.pkl')
    breed_encoder = joblib.load('breed_encoder.pkl')
    age_group_encoder = joblib.load('age_group_encoder.pkl')
    weight_category_encoder = joblib.load('weight_category_encoder.pkl')
    disease_prediction_encoder = joblib.load('disease_prediction_encoder.pkl')
    
    # Parse symptoms list from input
    if isinstance(symptoms, list):
        symptom_list = symptoms
    else:
        # Handle the case when symptoms are passed as individual parameters
        symptom_list = []
        for i, symptom in enumerate([symptoms.get(f'symptom{i+1}', None) for i in range(4)]):
            if symptom and symptom.lower() != 'none':
                symptom_list.append(symptom)
    
    # Determine age group
    if age < 1:
        age_group = 'Infant'
    elif 1 <= age < 3:
        age_group = 'Young'
    elif 3 <= age < 7:
        age_group = 'Adult'
    elif 7 <= age < 10:
        age_group = 'Middle_Aged'
    else:
        age_group = 'Senior'
    
    # Determine weight category based on animal type
    if animal_type == 'Dog':
        if weight < 10:
            weight_category = 'Small'
        elif 10 <= weight < 25:
            weight_category = 'Medium'
        elif 25 <= weight < 40:
            weight_category = 'Large'
        else:
            weight_category = 'Giant'
    elif animal_type == 'Cat':
        if weight < 4:
            weight_category = 'Small'
        elif 4 <= weight < 6:
            weight_category = 'Medium'
        else:
            weight_category = 'Large'
    else:
        weight_category = 'NA'
    
    # Parse duration to numerical value (days)
    duration_num = 0
    if 'day' in duration:
        duration_num = float(duration.split()[0])
    elif 'week' in duration:
        duration_num = float(duration.split()[0]) * 7
    
    # Create initial DataFrame with basic features
    data = {
        'Animal_Type': [animal_type],
        'Weight': [float(weight)],
        'Eye_Discharge': [int(eye_discharge)],
        'Body_Temperature': [float(body_temp)],
        'Heart_Rate': [int(heart_rate)],
        'Duration_Num': [duration_num],
        'Weight_Category': [weight_category],
    }
    
    df = pd.DataFrame(data)
    
    # Encode categorical variables
    df['Animal_Type'] = animal_type_encoder.transform([animal_type])[0]  
    df['Weight_Category'] = weight_category_encoder.transform([weight_category])[0]
    
    # Initialize all symptom features to 0
    symptom_features = [
        'Symptom_1_Coughing', 'Symptom_1_Lameness', 'Symptom_1_Lethargy', 'Symptom_1_Vomiting',
        'Symptom_2_Diarrhea', 'Symptom_2_Swollen Legs', 'Symptom_2_Vomiting',
        'Symptom_3_Diarrhea', 'Symptom_3_Lethargy', 'Symptom_3_Swollen Legs',
        'Symptom_4_Appetite Loss'
    ]
    
    for feature in symptom_features:
        df[feature] = 0
    
    # Set symptom values based on input
    if len(symptom_list) >= 1:
        symptom1 = symptom_list[0]
        if symptom1 == 'Coughing': df['Symptom_1_Coughing'] = 1
        elif symptom1 == 'Lameness': df['Symptom_1_Lameness'] = 1
        elif symptom1 == 'Lethargy': df['Symptom_1_Lethargy'] = 1
        elif symptom1 == 'Vomiting': df['Symptom_1_Vomiting'] = 1
    
    if len(symptom_list) >= 2:
        symptom2 = symptom_list[1]
        if symptom2 == 'Diarrhea': df['Symptom_2_Diarrhea'] = 1
        elif symptom2 == 'Swollen Legs': df['Symptom_2_Swollen Legs'] = 1
        elif symptom2 == 'Vomiting': df['Symptom_2_Vomiting'] = 1
    
    if len(symptom_list) >= 3:
        symptom3 = symptom_list[2]
        if symptom3 == 'Diarrhea': df['Symptom_3_Diarrhea'] = 1
        elif symptom3 == 'Lethargy': df['Symptom_3_Lethargy'] = 1
        elif symptom3 == 'Swollen Legs': df['Symptom_3_Swollen Legs'] = 1
    
    if len(symptom_list) >= 4:
        symptom4 = symptom_list[3]
        if symptom4 == 'Appetite Loss': df['Symptom_4_Appetite Loss'] = 1
    
    # Add duration features
    df['Duration_2 days'] = 1 if duration == '2 days' else 0
    df['Duration_7 days'] = 1 if duration == '7 days' else 0
    
    # Create feature array with only the selected features the model expects
    # From the model selection results: Index(['Animal_Type', 'Weight', 'Eye_Discharge', 'Body_Temperature', 'Heart_Rate', 
    # 'Duration_Num', 'Weight_Category', 'Symptom_1_Coughing', 'Symptom_1_Lameness', 'Symptom_1_Lethargy', 
    # 'Symptom_1_Vomiting', 'Symptom_2_Diarrhea', 'Symptom_2_Swollen Legs', 'Symptom_2_Vomiting', 
    # 'Symptom_3_Diarrhea', 'Symptom_3_Lethargy', 'Symptom_3_Swollen Legs', 'Symptom_4_Appetite Loss', 
    # 'Duration_2 days', 'Duration_7 days']
    
    expected_features = [
        'Animal_Type', 'Weight', 'Eye_Discharge', 'Body_Temperature', 'Heart_Rate', 
        'Duration_Num', 'Weight_Category', 'Symptom_1_Coughing', 'Symptom_1_Lameness', 
        'Symptom_1_Lethargy', 'Symptom_1_Vomiting', 'Symptom_2_Diarrhea', 
        'Symptom_2_Swollen Legs', 'Symptom_2_Vomiting', 'Symptom_3_Diarrhea', 
        'Symptom_3_Lethargy', 'Symptom_3_Swollen Legs', 'Symptom_4_Appetite Loss', 
        'Duration_2 days', 'Duration_7 days'
    ]
    
    # Ensure all expected features are in the DataFrame
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Keep only the expected features in the correct order
    X = df[expected_features]
    
    # Verify feature count
    if X.shape[1] != 20:
        raise ValueError(f"Feature mismatch: Model expects 20 features, but {X.shape[1]} were prepared.")
        
    # Predict disease
    prediction_idx = model.predict(X)[0]
    predicted_disease = disease_prediction_encoder.inverse_transform([prediction_idx])[0]
    
    # Get probabilities for each disease class
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
        prob_dict = {disease: prob for disease, prob in 
                   zip(disease_prediction_encoder.inverse_transform(range(len(probabilities))), 
                       probabilities)}
    else:
        prob_dict = {predicted_disease: 1.0}
    
    return predicted_disease, prob_dict