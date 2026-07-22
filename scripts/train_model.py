import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
OUTPUT_DIR = "C:\\Users\\vishw\\ml tut\\Project\\outputs"
df = pd.read_csv("C:\\Users\\vishw\\ml tut\\Project\\outputs\\features.csv")
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
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples")
rf = RandomForestRegressor(
    n_estimators=100,   # 100 decision trees
    max_depth=None,     # trees grow until pure
    random_state=42,
    n_jobs=-1           # use all CPU cores
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mean_y = y_test.mean()

print(f"\n── Results ──────────────────────────")
print(f"R²:   {r2:.4f}  ")
print(f"RMSE: {rmse:.2f} units")
print(f"Mean E_x: {mean_y:.2f} units")
print(f"RMSE as % of mean: {(rmse/mean_y)*100:.1f}%")
importances = rf.feature_importances_
importance_df = pd.DataFrame({
    "feature":    FEATURES,
    "importance": importances
}).sort_values("importance", ascending=False)

print(f"\n── Feature Importance ───────────────")
print(importance_df.to_string(index=False))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "r--", linewidth=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual E_x")
axes[0].set_ylabel("Predicted E_x")
axes[0].set_title(f"Predicted vs Actual  (R² = {r2:.3f})")
axes[0].legend()
axes[1].barh(importance_df["feature"], importance_df["importance"], color="steelblue")
axes[1].set_xlabel("Importance")
axes[1].set_title("Feature importance")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}\\model_results.png", dpi=120)
plt.show()
