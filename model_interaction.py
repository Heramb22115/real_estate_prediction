import pandas as pd
import numpy as np
import json
import joblib
from tensorflow.keras.models import load_model

MODEL_LOAD_PATH = "housing_price_model.keras"
PREPROCESSOR_LOAD_PATH = "preprocessor.joblib"
TIER_MAPPING_FILE = "city_tier_mapping.json"

model = None
preprocessor = None
city_tier_mapping = None
tier_start_index = -1

def load_resources():
    global model, preprocessor, city_tier_mapping, tier_start_index
    
    model = load_model(MODEL_LOAD_PATH)
    preprocessor = joblib.load(PREPROCESSOR_LOAD_PATH)
    
    with open(TIER_MAPPING_FILE, 'r') as f:
        city_tier_mapping = json.load(f)
        
    feature_names = preprocessor.get_feature_names_out()
    tier_size = len([name for name in feature_names if name.startswith('tier__')])
    tier_start_index = len(feature_names) - tier_size
    
    return tier_start_index

def preprocess_single_input(input_data, tier_start_index):
    df_single = pd.DataFrame([input_data])
    
    city_name = df_single['City'].iloc[0]
    df_single['City_Tier'] = city_tier_mapping.get(city_name, 'Tier_2')
    
    df_single.drop(['City'], axis=1, inplace=True)
    
    trained_columns = preprocessor.feature_names_in_.tolist()
    missing_cols = set(trained_columns) - set(df_single.columns)
    for col in missing_cols:
        if df_single.dtypes.get(col) == object:
            df_single[col] = ''
        else:
            df_single[col] = 0.0
            
    df_single = df_single[trained_columns]
    
    X_processed = preprocessor.transform(df_single)
    
    X_core = X_processed[:, :tier_start_index]
    X_tier = X_processed[:, tier_start_index:]
    
    return [X_core, X_tier]

def predict_house_price(input_data):
    global tier_start_index
    
    if model is None or preprocessor is None:
        tier_start_index = load_resources()
    
    X_inputs = preprocess_single_input(input_data, tier_start_index)
    
    y_pred_log = model.predict(X_inputs, verbose=0).flatten()[0]
    
    y_pred_real = np.expm1(y_pred_log)
    
    city_name = input_data['City']
    assigned_tier = city_tier_mapping.get(city_name, 'Tier_2')

    return y_pred_real, assigned_tier

def get_user_input():
    print("\n--- Enter Property Details for Prediction ---")
    
    city = input("1. City (e.g., Mumbai, Bangalore, Chennai): ")
    bhk = int(input("2. BHK (Number of bedrooms): "))
    size = int(input("3. Size in SqFt: "))
    furnished_status = input("4. Furnished Status (Furnished/Semi-furnished/Unfurnished): ")
    age = int(input("5. Age of Property (in years): "))
    total_floors = int(input("6. Total Floors in Building: "))
    owner_type = input("7. Owner Type (Owner/Builder/Broker): ")

    # Using safe defaults for less critical or hard-to-input features
    user_property = {
        'City': city,
        'BHK': bhk,
        'Size_in_SqFt': size,
        'Furnished_Status': furnished_status,
        'Age_of_Property': age,
        'Total_Floors': total_floors,
        'Owner_Type': owner_type,
        
        # Default values for other features
        'State': 'Unknown',
        'Property_Type': 'Apartment',
        'Price_per_SqFt': 0.08,
        'Year_Built': 2010,
        'Floor_No': 5,
        'Nearby_Schools': 5,
        'Nearby_Hospitals': 5,
        'Public_Transport_Accessibility': 'Medium',
        'Parking_Space': 'Yes',
        'Security': 'Yes',
        'Amenities': 'Gym, Pool',
        'Facing': 'East',
        'Availability_Status': 'Ready_to_Move'
    }
    return user_property

if __name__ == "__main__":
    
    user_data = get_user_input()
    
    predicted_price, assigned_tier = predict_house_price(user_data)
    
    print("\n=======================================================")
    print(f"Prediction for City: {user_data['City']} (Assigned Tier: {assigned_tier})")
    print(f"PREDICTED HOUSE PRICE: {predicted_price:.2f} Lakhs")
    print("=======================================================")