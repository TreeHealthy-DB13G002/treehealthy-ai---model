from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# inisiasi fastAPI
app = FastAPI(title="TreeHealthy AI model API", version="2.0")

# load model dan scaler
MODEL_PATH = "model_xgb_treehealthy.pkl"
SCALER_PATH = "scaler_treehealthy.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# struktur data input kuis
class KuisInput(BaseModel):
    HighBP: float
    BMI: float
    Smoker: float
    Stroke: float
    HeartDiseaseorAttack: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    NoDocbcCost: float
    MentalHlth: float  # Skala 0 - 30 hari stres
    DiffWalk: float
    Sex: float
    Age: float
    HighChol: float

# endpoint utama hitung resiko
@app.post("/api/predict")
def predict_risk(data: KuisInput):
    fitur_mentah = [
        data.HighBP, data.BMI, data.Smoker, data.Stroke, 
        data.HeartDiseaseorAttack, data.PhysActivity, data.Fruits, 
        data.Veggies, data.NoDocbcCost, data.MentalHlth, 
        data.DiffWalk, data.Sex, data.Age, data.HighChol
    ]
    # ubah ke aray 2 dimensi
    fitur_array = np.array([fitur_mentah])
    fitur_terpola = scaler.transform(fitur_array)
    #prediksi skor probabilitas
    probabilitas = model.predict_proba(fitur_terpola)[0][1] # Ambil probabilitas kelas 1 (Risiko)
    skor_persen = round(probabilitas * 100, 1)

    #label berdasarkan skor
    if skor_persen < 30:
        status = "Low Risk"
    elif skor_persen <= 70:
        status = "Moderate Risk"
    else:
        status = "High Risk"
    # kalkulasi bar UI
    bar_stress = round((data.MentalHlth / 30) * 100)
    # bar nutrisi
    bar_nutrition = round(((data.Fruits + data.Veggies) / 2) * 100)
    if bar_nutrition == 0: bar_nutrition = 10
    # bar actifity
    kemampuan_jalan = 1 if data.DiffWalk == 0 else 0 # 1 jika sehat, 0 jika susah jalan
    bar_activity = round(((data.PhysActivity + kemampuan_jalan) / 2) * 100)
    if bar_activity == 0: bar_activity = 15
    
    return {
        "status": "success",
        "message": "Kalkulasi risiko berhasil dihitung",
        "data": {
            "risk_percentage": skor_persen,
            "label_status": status,
            "ui_bars": {
                "stress_level": bar_stress,
                "physical_activity": bar_activity,
                "nutritional_habits": bar_nutrition
            }
        }
    }

# Endpoint testing sederhana untuk memastikan server hidup
@app.get("/")
def index():
    return {"message": "Server TreeHealthy Backend is Running!"}