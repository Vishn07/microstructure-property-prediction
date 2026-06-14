import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
 
CSV_PATH    = r"C:\Users\vishw\ml tut\Project\data\cr9 data\tensile-properties-from-jeffs-ml-sheet-20220208.csv"
MAPPING_CSV = r"C:\Users\vishw\ml tut\Project\data\cr9 data\phase8_alloy_mapping.csv"
OUTPUT_DIR  = r"C:\Users\vishw\ml tut\Project\outputs"
 def extract_features(img):
    if img.dtype != np.uint8:
        img = (255 * (img - img.min()) / (img.ptp() + 1e-8)).astype(np.uint8)
    img_f  = img.astype(np.float64) / 255.0
    img_64 = (img // 4).astype(np.uint8)
 
    mean_int = img_f.mean()
    edges    = cv2.Canny(img, 100, 200)
    edge_den = edges.mean() / 255.0
    tex_var  = float(img_f.var() * 255 * 255)
 
    feats = {}
    for d in [1, 8]:
        gcm = graycomatrix(img_64, distances=[d], angles=[0],
                           levels=64, symmetric=True, normed=True)
        feats[f"contrast_d{d}"]    = graycoprops(gcm, "contrast")[0, 0]
        feats[f"homogeneity_d{d}"] = graycoprops(gcm, "homogeneity")[0, 0]
        feats[f"energy_d{d}"]      = graycoprops(gcm, "energy")[0, 0]
        feats[f"correlation_d{d}"] = graycoprops(gcm, "correlation")[0, 0]
 
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    phase_frac = float((binary > 0).sum()) / binary.size
 
    f     = np.fft.fft2(img_f)
    mag   = np.abs(np.fft.fftshift(f)) ** 2
    h, w  = mag.shape
    cx, cy = h // 2, w // 2
    radius = min(h, w) // 10
    y, x  = np.ogrid[:h, :w]
    mask  = (x - cx)**2 + (y - cy)**2 <= radius**2
    fft_r = float(mag[mask].sum() / (mag.sum() + 1e-10))
 
    bin3 = (binary > 0).astype(int)
    run_lengths = []
    for row in bin3:
        count = 1
        for j in range(1, len(row)):
            if row[j] == row[j-1]:
                count += 1
            else:
                run_lengths.append(count)
                count = 1
        run_lengths.append(count)
    run_len_mean = float(np.mean(run_lengths)) if run_lengths else 0.0
    run_len_var  = float(np.var(run_lengths))  if run_lengths else 0.0
 
    return np.array([
        mean_int, edge_den, tex_var,
        feats["contrast_d1"], feats["homogeneity_d1"],
        feats["energy_d1"],   feats["correlation_d1"],
        feats["contrast_d8"], feats["homogeneity_d8"],
        feats["energy_d8"],   feats["correlation_d8"],
        phase_frac, fft_r, run_len_mean, run_len_var,
    ], dtype=np.float32)
 
FEATURE_NAMES = [
    "mean_intensity", "edge_density", "texture_variance",
    "contrast_d1", "homogeneity_d1", "energy_d1", "correlation_d1",
    "contrast_d8", "homogeneity_d8", "energy_d8", "correlation_d8",
    "phase_fraction", "fft_ratio", "run_len_mean", "run_len_var",
    "temperature",   # <-- the new 16th feature
]
 
 
def grouped_loocv(X, y, groups, label):
    """
    Leave-One-Alloy-Out. Hold out all rows of one alloy, train on the rest.
    """
    unique_groups = np.unique(groups)
    preds = np.zeros(len(y))
    for g in unique_groups:
        test_mask  = groups == g
        train_mask = ~test_mask
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X[train_mask], y[train_mask])
        preds[test_mask] = rf.predict(X[test_mask])
    r2   = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae  = mean_absolute_error(y, preds)
    print(f"\n{label}:")
    print(f"  R2:   {r2:.4f}")
    print(f"  RMSE: {rmse:.1f} MPa")
    print(f"  MAE:  {mae:.1f} MPa")
    print(f"  Mean {label}: {y.mean():.0f} MPa  (RMSE {rmse/y.mean()*100:.1f}% of mean)")
    return preds, r2, rmse
 
 
def main():
    # 1. Load CSV and mapping
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Alloy"])
    df["Alloy"] = df["Alloy"].astype(str)
 
    mapping = pd.read_csv(MAPPING_CSV)
    alloy_to_path = dict(zip(mapping["alloy"].astype(str), mapping["path"]))
 
    df = df[df["Alloy"].isin(alloy_to_path.keys())].copy()
    print(f"Total measurements (all temps): {len(df)} across {df['Alloy'].nunique()} alloys")
 
    # 2. Extract image features ONCE per alloy (cache them)
    alloy_features = {}
    for alloy, folder in alloy_to_path.items():
        if not os.path.isdir(folder):
            continue
        bmps = [f for f in os.listdir(folder) if f.lower().endswith(".bmp")]
        feats = []
        for fname in bmps:
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (256, 256))
            feats.append(extract_features(img))
        if feats:
            alloy_features[alloy] = np.mean(feats, axis=0)
            print(f"  {alloy:<12} {len(feats)} images")
 
    # 3. Build the full dataset: each CSV row = image features + temperature
    X, y_ys, y_uts, groups = [], [], [], []
    for _, row in df.iterrows():
        alloy = row["Alloy"]
        if alloy not in alloy_features:
            continue
        temp = row["Temperature, [C]"]
        feat_vec = np.append(alloy_features[alloy], temp)   # 15 + 1 = 16 features
        X.append(feat_vec)
        y_ys.append(row["Yield Stress, [MPa]"])
        y_uts.append(row["Ultimate Tensile Stress, [MPa]"])
        groups.append(alloy)
 
    X      = np.array(X)
    y_ys   = np.array(y_ys)
    y_uts  = np.array(y_uts)
    groups = np.array(groups)
 
    print(f"\nFinal dataset: {X.shape[0]} measurements, {X.shape[1]} features "
          f"(15 image + 1 temperature)")
 
    # 4. Grouped LOOCV
    ys_preds,  ys_r2,  ys_rmse  = grouped_loocv(X, y_ys,  groups, "Yield Strength")
    uts_preds, uts_r2, uts_rmse = grouped_loocv(X, y_uts, groups, "Ultimate Tensile Strength")
 
    # 5. Feature importance
    rf_full = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_full.fit(X, y_ys)
    imp = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": rf_full.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop features for YS prediction:")
    print(imp.head(8).to_string(index=False))
 
    # 6. Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 8b: Temperature-Aware YS/UTS Prediction (241 measurements, 29 alloys)",
                 fontsize=12, fontweight="bold")
 
    ax = axes[0]
    sc = ax.scatter(y_ys, ys_preds, c=X[:, -1], cmap="coolwarm", alpha=0.7, s=30)
    lims = [min(y_ys.min(), ys_preds.min()), max(y_ys.max(), ys_preds.max())]
    ax.plot(lims, lims, "k--", linewidth=1.2)
    ax.set_xlabel("Actual YS (MPa)")
    ax.set_ylabel("Predicted YS (MPa)")
    ax.set_title(f"Yield Strength (R2={ys_r2:.3f}, RMSE={ys_rmse:.0f})")
    plt.colorbar(sc, ax=ax, label="Test temp (C)")
 
    ax = axes[1]
    sc = ax.scatter(y_uts, uts_preds, c=X[:, -1], cmap="coolwarm", alpha=0.7, s=30)
    lims = [min(y_uts.min(), uts_preds.min()), max(y_uts.max(), uts_preds.max())]
    ax.plot(lims, lims, "k--", linewidth=1.2)
    ax.set_xlabel("Actual UTS (MPa)")
    ax.set_ylabel("Predicted UTS (MPa)")
    ax.set_title(f"UTS (R2={uts_r2:.3f}, RMSE={uts_rmse:.0f})")
    plt.colorbar(sc, ax=ax, label="Test temp (C)")
 
    ax = axes[2]
    top = imp.head(8)
    colors = ["tomato" if f == "temperature" else "steelblue" for f in top["feature"]]
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top Features (temperature in red)")
 
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase8b_results.png")
    plt.savefig(out, dpi=130)
    plt.show()
 
 
if __name__ == "__main__":
    main()