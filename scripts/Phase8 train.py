import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
MAPPING_CSV = r"C:\Users\vishw\ml tut\Project\data\cr9 data\phase8_alloy_mapping.csv"
OUTPUT_DIR  = r"C:\Users\vishw\ml tut\Project\outputs"

def extract_features(img):
    """
    img: grayscale uint8 image (real micrograph, NOT binary)
    Returns 15 hand-crafted features.
    Note: unlike MICRO2D these are real grayscale images, so phase_fraction
    uses Otsu thresholding to separate phases rather than assuming 0/1.
    """
    # Ensure uint8
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
 
    # Phase fraction via Otsu threshold (real images are grayscale)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    phase_frac = float((binary > 0).sum()) / binary.size
 
    # FFT low-freq ratio
    f     = np.fft.fft2(img_f)
    mag   = np.abs(np.fft.fftshift(f)) ** 2
    h, w  = mag.shape
    cx, cy = h // 2, w // 2
    radius = min(h, w) // 10
    y, x  = np.ogrid[:h, :w]
    mask  = (x - cx)**2 + (y - cy)**2 <= radius**2
    fft_r = float(mag[mask].sum() / (mag.sum() + 1e-10))
 
    # Run length on Otsu-binarized image
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
]
def main():
    # 1. Load the verified alloy mapping
    print("Loading alloy mapping...")
    mapping = pd.read_csv(MAPPING_CSV)
    print(f"Alloys to process: {len(mapping)}")
 
    # 2. Extract features per alloy (average across that alloy's images)
    print("\nExtracting features (averaged per alloy)...")
    X, ys_ys, ys_uts, alloy_names = [], [], [], []
 
    for _, row in mapping.iterrows():
        folder = row["path"]
        if not os.path.isdir(folder):
            print(f"  WARNING: folder missing for {row['alloy']}: {folder}")
            continue
 
        bmps = [f for f in os.listdir(folder) if f.lower().endswith(".bmp")]
        if not bmps:
            continue
 
        # Extract features for every image, then average
        alloy_feats = []
        for fname in bmps:
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (256, 256))
            alloy_feats.append(extract_features(img))
 
        if not alloy_feats:
            continue
 
        avg_feat = np.mean(alloy_feats, axis=0)
        X.append(avg_feat)
        ys_ys.append(row["YS"])
        ys_uts.append(row["UTS"])
        alloy_names.append(row["alloy"])
        print(f"  {row['alloy']:<12} {len(alloy_feats)} images averaged")
 
    X      = np.array(X)
    ys_ys  = np.array(ys_ys)
    ys_uts = np.array(ys_uts)
 
    print(f"\nFinal dataset: {X.shape[0]} alloys, {X.shape[1]} features each")
 
    # 3. Leave-One-Alloy-Out cross validation
    # With 29 alloys, LOOCV is the most data-efficient honest evaluation.
    # Each alloy is held out once; model trains on the other 28.
    print("\nRunning Leave-One-Alloy-Out cross validation...")
 
    def loocv(X, y, label):
        loo = LeaveOneOut()
        preds = np.zeros(len(y))
        for train_idx, test_idx in loo.split(X):
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X[train_idx], y[train_idx])
            preds[test_idx] = rf.predict(X[test_idx])
        r2   = r2_score(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        mae  = mean_absolute_error(y, preds)
        print(f"\n{label}:")
        print(f"  R2:   {r2:.4f}")
        print(f"  RMSE: {rmse:.1f} MPa")
        print(f"  MAE:  {mae:.1f} MPa")
        print(f"  Mean {label}: {y.mean():.0f} MPa  (RMSE is {rmse/y.mean()*100:.1f}% of mean)")
        return preds, r2, rmse
 
    ys_preds,  ys_r2,  ys_rmse  = loocv(X, ys_ys,  "Yield Strength")
    uts_preds, uts_r2, uts_rmse = loocv(X, ys_uts, "Ultimate Tensile Strength")
 
    # 4. Feature importance (train once on all data just for inspection)
    rf_full = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_full.fit(X, ys_ys)
    imp = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": rf_full.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop features for YS prediction:")
    print(imp.head(8).to_string(index=False))
 
    # 5. Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 8: Real 9Cr Steel YS/UTS Prediction (LOOCV, 29 alloys)",
                 fontsize=12, fontweight="bold")
 
    ax = axes[0]
    ax.scatter(ys_ys, ys_preds, alpha=0.7, s=40, color="steelblue")
    lims = [min(ys_ys.min(), ys_preds.min()), max(ys_ys.max(), ys_preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_xlabel("Actual YS (MPa)")
    ax.set_ylabel("Predicted YS (MPa)")
    ax.set_title(f"Yield Strength (R2={ys_r2:.3f}, RMSE={ys_rmse:.0f})")
 
    ax = axes[1]
    ax.scatter(ys_uts, uts_preds, alpha=0.7, s=40, color="seagreen")
    lims = [min(ys_uts.min(), uts_preds.min()), max(ys_uts.max(), uts_preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_xlabel("Actual UTS (MPa)")
    ax.set_ylabel("Predicted UTS (MPa)")
    ax.set_title(f"UTS (R2={uts_r2:.3f}, RMSE={uts_rmse:.0f})")
 
    ax = axes[2]
    top = imp.head(8)
    ax.barh(top["feature"], top["importance"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top Features (YS)")
 
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase8_results.png")
    plt.savefig(out, dpi=130)
    plt.show()
    print(f"\nPlot saved to {out}")
 
    # 6. Save predictions table
    results_df = pd.DataFrame({
        "alloy":      alloy_names,
        "YS_actual":  ys_ys,
        "YS_pred":    ys_preds.round(0),
        "UTS_actual": ys_uts,
        "UTS_pred":   uts_preds.round(0),
    })
    results_csv = os.path.join(OUTPUT_DIR, "phase8_predictions.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Predictions saved to {results_csv}")
    print("\n" + results_df.to_string(index=False))
 
 
if __name__ == "__main__":
    main()