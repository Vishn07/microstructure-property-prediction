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
 
# Composition columns in the CSV (skip ones that are mostly empty)
COMP_COLS = ["Fe", "C", "Cr", "Mn", "Si", "Ni", "Co", "Mo", "W",
             "Nb", "Al", "P", "Cu", "V", "B", "N"]
 
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
                run_lengths.append(count); count = 1
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
 
 
def grouped_loocv(X, y, groups):
    """Leave-One-Alloy-Out CV. Returns predictions, R2, RMSE, MAE."""
    preds = np.zeros(len(y))
    for g in np.unique(groups):
        test = groups == g
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X[~test], y[~test])
        preds[test] = rf.predict(X[test])
    return (preds,
            r2_score(y, preds),
            np.sqrt(mean_squared_error(y, preds)),
            mean_absolute_error(y, preds))
 
 
def main():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Alloy"])
    df["Alloy"] = df["Alloy"].astype(str)
 
    mapping = pd.read_csv(MAPPING_CSV)
    alloy_to_path = dict(zip(mapping["alloy"].astype(str), mapping["path"]))
    df = df[df["Alloy"].isin(alloy_to_path.keys())].copy()
 
    # Fill missing composition values with 0 (means element not added)
    for col in COMP_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
 
    print(f"Measurements: {len(df)} across {df['Alloy'].nunique()} alloys")
 
    # Extract image features once per alloy
    alloy_img_feats = {}
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
            alloy_img_feats[alloy] = np.mean(feats, axis=0)
 
    # Build feature blocks
    img_block, comp_block, temp_block = [], [], []
    y_ys, y_uts, groups = [], [], []
 
    for _, row in df.iterrows():
        alloy = row["Alloy"]
        if alloy not in alloy_img_feats:
            continue
        img_block.append(alloy_img_feats[alloy])
        comp_block.append([row[c] for c in COMP_COLS if c in df.columns])
        temp_block.append([row["Temperature, [C]"]])
        y_ys.append(row["Yield Stress, [MPa]"])
        y_uts.append(row["Ultimate Tensile Stress, [MPa]"])
        groups.append(alloy)
 
    img_block  = np.array(img_block)
    comp_block = np.array(comp_block)
    temp_block = np.array(temp_block)
    y_ys   = np.array(y_ys)
    y_uts  = np.array(y_uts)
    groups = np.array(groups)
 
    print(f"Image features: {img_block.shape[1]}, "
          f"Composition: {comp_block.shape[1]}, Temperature: 1")
 
    # Define the 4 models
    models = {
        "A: Comp + Temp":        np.hstack([comp_block, temp_block]),
        "B: Image + Temp":       np.hstack([img_block, temp_block]),
        "C: Image+Comp+Temp":    np.hstack([img_block, comp_block, temp_block]),
        "D: Comp only":          comp_block,
    }
 
    print("ABLATION RESULTS (Leave-One-Alloy-Out CV)")
 
    results = {}
    for name, X in models.items():
        _, ys_r2, ys_rmse, ys_mae    = grouped_loocv(X, y_ys,  groups)
        _, uts_r2, uts_rmse, uts_mae = grouped_loocv(X, y_uts, groups)
        results[name] = {
            "ys_r2": ys_r2, "ys_rmse": ys_rmse,
            "uts_r2": uts_r2, "uts_rmse": uts_rmse,
        }
        print(f"\n{name}")
        print(f"  YS:  R2={ys_r2:.3f}  RMSE={ys_rmse:.0f} MPa")
        print(f"  UTS: R2={uts_r2:.3f}  RMSE={uts_rmse:.0f} MPa")
 
    # The key comparison
    print("KEY QUESTION: Do images add value beyond composition?")
    a_ys = results["A: Comp + Temp"]["ys_r2"]
    c_ys = results["C: Image+Comp+Temp"]["ys_r2"]
    diff = c_ys - a_ys
    print(f"Comp+Temp (A) YS R2:        {a_ys:.3f}")
    print(f"Image+Comp+Temp (C) YS R2:  {c_ys:.3f}")
    print(f"Image contribution:         {diff:+.3f} R2")
    if diff > 0.02:
        print("=> Images ADD meaningful predictive value beyond composition.")
    elif diff > 0:
        print("=> Images add marginal value beyond composition.")
    else:
        print("=> Images are redundant once composition is known.")
 
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase 9: Image vs Composition Ablation (LOOCV)",
                 fontsize=12, fontweight="bold")
 
    names = list(results.keys())
    ys_r2s  = [results[n]["ys_r2"]  for n in names]
    uts_r2s = [results[n]["uts_r2"] for n in names]
    colors = ["#888", "#e67e22", "#27ae60", "#aaa"]
 
    for a, vals, title in [(ax[0], ys_r2s, "Yield Strength R2"),
                            (ax[1], uts_r2s, "UTS R2")]:
        bars = a.bar(range(len(names)), vals, color=colors)
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        a.set_ylabel("R2")
        a.set_title(title)
        a.set_ylim(min(0, min(vals) - 0.05), max(vals) + 0.1)
        for b, v in zip(bars, vals):
            a.text(b.get_x()+b.get_width()/2, v + 0.01, f"{v:.3f}",
                   ha="center", va="bottom", fontsize=8, fontweight="bold")
 
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase9_ablation.png")
    plt.savefig(out, dpi=130)
    plt.show()
 
 
if __name__ == "__main__":
    main()