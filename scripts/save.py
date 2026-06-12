import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

CSV_PATH = "C:\\Users\\vishw\\ml tut\\Project\\outputs\\features.csv"
MODEL_PATH = "C:\\Users\\vishw\\ml tut\\Project\\outputs\\rf_model.pkl"

df = pd.read_csv(CSV_PATH)

FEATURES = [
    "mean_intensity",
    "edge_density",
    "texture_variance",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
]

X = df[FEATURES].values
y = df["E_x"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# save model AND feature names together so predict.py knows the right order
model_data = {
    "model":         rf,
    "feature_names": FEATURES,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print(f"Model saved to {MODEL_PATH}")