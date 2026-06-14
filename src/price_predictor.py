import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def load_price_data(csv_path):
    df = pd.read_csv(csv_path)
    print(df.head())
    print(df.columns.tolist())
    return df

def train(csv_path):
    df = load_price_data(csv_path)
    
    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R²:   {r2_score(y_test, y_pred):.4f}")
    
    joblib.dump(model, 'models/price_model.pkl')
    joblib.dump(le_dict, 'models/price_encoders.pkl')
    print("Price model saved.")
    
    return model

def predict_price(input_dict, model=None, le_dict=None):
    if model is None:
        model   = joblib.load('models/price_model.pkl')
        le_dict = joblib.load('models/price_encoders.pkl')
    
    df = pd.DataFrame([input_dict])
    for col, le in le_dict.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    return model.predict(df)[0]