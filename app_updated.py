
import streamlit as st
import pandas as pd
import joblib
import os

st.title("Vergleich mehrerer ML-Modelle zur Kaufabsichtsvorhersage")

# Wichtige Eingabe-Features (aus Training bekannt)
used_features = [
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "VisitorType"
]

# Eingabemaske für Features
month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'June':6,
             'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
visitor_map = {"Returning_Visitor": 1, "New_Visitor": 0, "Other": 0}

st.sidebar.header("🧾 Nutzereingaben")
input_dict = {
    "Administrative_Duration": st.sidebar.slider("Administrative_Duration", 0.0, 1000.0, 50.0),
    "Informational_Duration": st.sidebar.slider("Informational_Duration", 0.0, 500.0, 20.0),
    "ProductRelated_Duration": st.sidebar.slider("ProductRelated_Duration", 0.0, 5000.0, 200.0),
    "BounceRates": st.sidebar.slider("BounceRates", 0.0, 1.0, 0.2),
    "ExitRates": st.sidebar.slider("ExitRates", 0.0, 1.0, 0.25),
    "PageValues": st.sidebar.slider("PageValues", 0.0, 100.0, 5.0),
    "SpecialDay": st.sidebar.slider("SpecialDay", 0.0, 1.0, 0.0),
    "Month": month_map[st.sidebar.selectbox("Month", list(month_map.keys()))],
    "VisitorType": visitor_map[st.sidebar.selectbox("VisitorType", list(visitor_map.keys()))]
}

input_df = pd.DataFrame([input_dict])

# Modellnamen und Pfade
model_dir = "mnt/data"
model_files = {
    "Logistische Regression": "logreg_model.pkl",
    "Baseline Modell": "baseline_model.pkl",
    "Entscheidungsbaum": "tree_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "LightGBM": "lgbm_model.pkl",
    "Random Forest (Best)": "best_rf_model.pkl",
    "Stacking Modell": "stacking_model.pkl"
}

# Vorhersagen aller Modelle
results = []
for name, path in model_files.items():
    try:
        model = joblib.load(f"/{model_dir}/{path}")
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else "-"
        results.append((name, "JA" if pred == 1 else "NEIN", f"{prob:.2f}" if prob != "-" else "-"))
    except Exception as e:
        results.append((name, "Fehler", str(e)))

# Ausgabe
st.subheader("🔍 Modellvergleich")
result_df = pd.DataFrame(results, columns=["Modell", "Vorhersage", "Wahrscheinlichkeit (für JA)"])
st.table(result_df)
