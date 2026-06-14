import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import load_images, extract_features, encode_labels

def train(data_dir):
    print("Loading images...")
    images, labels = load_images(data_dir)
    
    print(f"Loaded {len(images)} images, {len(set(labels))} classes")
    
    print("Extracting features...")
    features = extract_features(images)
    encoded_labels, le = encode_labels(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42
    )
    
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    joblib.dump(model, 'models/disease_model.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print("Model saved.")
    
    return model, le

def predict(image_array, model=None, le=None):
    if model is None:
        model = joblib.load('models/disease_model.pkl')
        le    = joblib.load('models/label_encoder.pkl')
    
    from preprocessing import extract_features
    features = extract_features([image_array])
    pred = model.predict(features)
    label = le.inverse_transform(pred)[0]
    return label