import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam

DATA_FILE = "india_housing_prices.csv"
TIER_MAPPING_FILE = "city_tier_mapping.json"
MODEL_SAVE_PATH = "housing_price_model.keras"
PREPROCESSOR_SAVE_PATH = "preprocessor.joblib"

def load_and_prepare_data(data_path, mapping_path):
    df = pd.read_csv(data_path)
    
    with open(mapping_path, 'r') as f:
        tier_mapping = json.load(f)
        
    df['City_Tier'] = df['City'].map(tier_mapping)
    
    df.drop(['ID', 'Locality', 'City'], axis=1, inplace=True)
    
    y = df.pop('Price_in_Lakhs').values
    X = df
    
    y = np.log1p(y)
    
    return X, y

def build_preprocessor(X):
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    tier_feature = ['City_Tier']
    categorical_features.remove('City_Tier')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('tier', OneHotEncoder(handle_unknown='ignore', sparse_output=False), tier_feature)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    return preprocessor

def build_keras_model(preprocessor, X):
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    
    tier_cols = [name for name in feature_names if name.startswith('tier__')]
    tier_size = len(tier_cols)
    
    tier_start_index = X_processed.shape[1] - tier_size
    core_input_size = tier_start_index
    
    core_input = Input(shape=(core_input_size,), name='core_input')
    tier_input = Input(shape=(tier_size,), name='tier_input')
    
    x = Dense(64, activation='relu')(core_input)
    x = Dense(32, activation='relu')(x)
    
    merged = concatenate([x, tier_input])
    
    final = Dense(16, activation='relu')(merged)
    output = Dense(1, activation='linear', name='price_output')(final) 
    
    model = Model(inputs=[core_input, tier_input], outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model, tier_start_index

if __name__ == "__main__":
    X, y = load_and_prepare_data(DATA_FILE, TIER_MAPPING_FILE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = build_preprocessor(X)
    
    model, tier_start_index = build_keras_model(preprocessor, X)
    
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    X_train_core = X_train_processed[:, :tier_start_index]
    X_train_tier = X_train_processed[:, tier_start_index:]
    
    X_test_core = X_test_processed[:, :tier_start_index]
    X_test_tier = X_test_processed[:, tier_start_index:]
    
    history = model.fit(
        [X_train_core, X_train_tier], y_train,
        validation_data=([X_test_core, X_test_tier], y_test),
        epochs=10,
        batch_size=32,
        verbose=0
    )
    
    model.save(MODEL_SAVE_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)
    
    y_pred_log = model.predict([X_test_core, X_test_tier], verbose=0).flatten()
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)
    
    total_variance = np.sum((y_test_real - np.mean(y_test_real))**2)
    residual_variance = np.sum((y_test_real - y_pred_real)**2)
    r2 = 1 - (residual_variance / total_variance)
    
    mae = np.mean(np.abs(y_test_real - y_pred_real))
    
    print("--- Keras Model Training Complete ---")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_SAVE_PATH}")
    print(f"Test R-squared: {r2:.4f}")
    print(f"Test MAE (Lakhs): {mae:.2f} Lakhs")