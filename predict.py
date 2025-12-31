import pandas as pd
import joblib
from diabetes_prediction import DiabetesPredictor


def predict_single_patient(patient_data):
    """
    Predict diabetes for a single patient.
    
    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient features
        
    Returns:
    --------
    int
        Prediction (0: No Diabetes, 1: Diabetes)
    """
    # Load trained model
    predictor = DiabetesPredictor()
    predictor.load_data()
    predictor.prepare_data()
    predictor.train_model()
    
    # Convert patient data to DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Make prediction
    prediction = predictor.model.predict(patient_df)[0]
    probability = predictor.model.predict_proba(patient_df)[0]
    
    result = {
        'prediction': 'Diabetes' if prediction == 1 else 'No Diabetes',
        'probability_no_diabetes': probability[0],
        'probability_diabetes': probability[1]
    }
    
    return result


if __name__ == "__main__":
    # Example patient data
    example_patient = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    result = predict_single_patient(example_patient)
    
    print("Prediction Result:")
    print(f"  Diagnosis: {result['prediction']}")
    print(f"  Probability of No Diabetes: {result['probability_no_diabetes']:.2%}")
    print(f"  Probability of Diabetes: {result['probability_diabetes']:.2%}")
