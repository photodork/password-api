import joblib
import re
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("time_to_crack.pkl")
scaler = joblib.load("scaler.pkl")

class Password(BaseModel):
    password: str

def extract_features(pw: str):
    length = len(pw)
    digits = len(re.findall(r'[0-9]', pw))
    symbols = len(re.findall(r'[^A-Za-z0-9]', pw))
    upper = len(re.findall(r'[A-Z]', pw))
    lower = len(re.findall(r'[a-z]', pw))
    return [length, digits, symbols, upper, lower]

@app.post("/predict")
def predict(data: Password):
    features = np.array([extract_features(data.password)])
    features_scaled = scaler.transform(features)
    predicted_log_time = model.predict(features_scaled)
    predicted_time_seconds = np.expm1(predicted_log_time)[0]
    return {"time_to_crack": predicted_time_seconds}

@app.get("/")
def root():
    return {"message": "Password Crack-Time API is running!"}