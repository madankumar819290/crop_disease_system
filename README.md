# Crop Disease Detection & Price Prediction System

## Problem Statement
Indian farmers lose 20-30% of crops annually due to undetected diseases 
and poor price forecasting. This system solves both problems using ML.

## Features
- Crop disease detection from leaf images (15 disease classes)
- Mandi price prediction using historical data
- REST API for real-time predictions

## Tech Stack
- Python, Scikit-learn, Random Forest, Gradient Boosting
- FastAPI for deployment
- Dataset: PlantVillage (54,000+ images)

## Model Performance
- Disease Detection Accuracy: 73%

## How to Run
pip install -r requirements.txt
python main.py

## Project Structure
crop-disease-system/
├── data/
├── models/
├── src/
│   ├── preprocessing.py
│   ├── disease_detector.py
│   └── price_predictor.py
├── app.py
├── main.py
└── requirements.txt

## Author
Your Name — B.Tech CS (AI), GL Bajaj Institutepip 