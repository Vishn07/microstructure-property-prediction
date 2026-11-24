"""
Phase 10: Quantifying the Synthetic-to-Real Domain Gap
=======================================================
The project showed R2 = 0.94 on synthetic MICRO2D but only R2 = 0.47 on real
steel. How much of that drop is caused by IMAGING ARTIFACTS alone, versus
other factors (different material, different target, fewer samples)?

This isolates the imaging variable by degrading synthetic images with the same
physically motivated artifacts validated in Phase 7b, then running the classic
2x2 domain adaptation matrix:

                    TEST clean    TEST degraded
  TRAIN clean          (1)            (2)          <- (2) is the domain gap
  TRAIN degraded       (3)            (4)          <- (4) is domain-matched training

Key readings:
  (1) baseline, should reproduce ~0.94
  (2) how badly a clean-trained model fails on realistic images
  (1)-(2) = the pure imaging-artifact penalty
  (4) whether training on degraded data recovers performance
  (4)-(2) = how much domain-matched training buys you

Degradation pipeline (same physics as Phase 7b):
  binary -> grayscale phases -> etch variation -> illumination -> blur -> noise
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

FILE_PATH  = r"C:\Users\vishw\ml tut\Project\data\MICRO2D_homogenized.h5"
OUTPUT_DIR = r"C:\Users\vishw\ml tut\Project\outputs"

CLASS      = "GRF"
N_SAMPLES  = 10000    
SEED       = 42


# ── Degradation: make a synthetic binary image look like a real micrograph ────
def degrade_image(binary_img, rng):
    """
    binary_img: 256x256 array of 0/1 ints (MICRO2D format)
    Returns: uint8 grayscale image with realistic imaging artifacts.

    Each step models something that physically happens when a real
    micrograph is collected. Same artifact set validated in Phase 7b.
    """
    # 1. Binary -> grayscale phase intensities.
    #    Real micrographs never have perfectly uniform phases. Each phase has
    #    a mean gray level with some internal variation.
    dark_mean  = rng.uniform(55, 85)     # etched/darker phase
    light_mean = rng.uniform(170, 205)   # lighter phase
    img = np.where(binary_img > 0, light_mean, dark_mean).astype(np.float32)

    # Within-phase intensity variation (polishing, local composition)
    img += rng.normal(0, 6, img.shape)

    # 2. Soften phase boundaries. Real interfaces are never single-pixel sharp
    #    because of etch undercut and optical diffraction.
    img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.6, 1.4))

    # 3. Etching variation -> gamma correction (nonlinear contrast change)
    img_u = np.clip(img, 0, 255).astype(np.uint8)
    gamma = rng.uniform(0.75, 1.35)
    table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
    img_u = cv2.LUT(img_u, table)

    # 4. Illumination gradient across the field of view
    h, w = img_u.shape
    strength = rng.uniform(0.08, 0.25)
    angle    = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx - w / 2) / (w / 2)
    yy = (yy - h / 2) / (h / 2)
    field = 1.0 + strength * (np.cos(angle) * xx + np.sin(angle) * yy)
    img_f = img_u.astype(np.float32) * field

    # 5. Vignette
    r = np.sqrt((xx * w / 2) ** 2 + (yy * h / 2) ** 2)
    img_f *= (1.0 - rng.uniform(0.05, 0.20) * (r / r.max()) ** 2)

    # 6. Focus variation
    img_u = np.clip(img_f, 0, 255).astype(np.uint8)
    sigma = rng.uniform(0.0, 1.0)
    if sigma > 0.1:
        k = int(2 * round(3 * sigma) + 1)
        img_u = cv2.GaussianBlur(img_u, (k, k), sigma)

    # 7. Sensor noise
    img_u = np.clip(img_u.astype(np.float32) + rng.normal(0, rng.uniform(2, 7),
                                                          img_u.shape), 0, 255)
    return img_u.astype(np.uint8)


# ── Feature extraction (identical for clean and degraded) ────────────────────
def extract_features(img_uint8):
    """
    Same 15 features used throughout the project.
    Uses Otsu thresholding for phase fraction so it works on BOTH
    clean binary images and degraded grayscale images without modification.
    """
    img_f  = img_uint8.astype(np.float64) / 255.0
    img_64 = (img_uint8 // 4).astype(np.uint8)

    mean_int = img_f.mean()
    edge_den = cv2.Canny(img_uint8, 100, 200).mean() / 255.0
    tex_var  = float(img_f.var() * 255 * 255)

    feats = {}
    for d in [1, 8]:
        gcm = graycomatrix(img_64, distances=[d], angles=[0],
                           levels=64, symmetric=True, normed=True)
        feats[f"contrast_d{d}"]    = graycoprops(gcm, "contrast")[0, 0]
        feats[f"homogeneity_d{d}"] = graycoprops(gcm, "homogeneity")[0, 0]
        feats[f"energy_d{d}"]      = graycoprops(gcm, "energy")[0, 0]
        feats[f"correlation_d{d}"] = graycoprops(gcm, "correlation")[0, 0]

    _, binary = cv2.threshold(img_uint8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    phase_frac = float((binary > 0).sum()) / binary.size

    f    = np.fft.fft2(img_f)
    mag  = np.abs(np.fft.fftshift(f)) ** 2
    h, w = mag.shape
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= (min(h, w) // 10) ** 2
    fft_r = float(mag[mask].sum() / (mag.sum() + 1e-10))

    bin3 = (binary > 0).astype(int)
    runs = []
    for row in bin3:
        c = 1
        for j in range(1, len(row)):
            if row[j] == row[j - 1]:
                c += 1
            else:
                runs.append(c); c = 1
        runs.append(c)
    run_mean = float(np.mean(runs)) if runs else 0.0
    run_var  = float(np.var(runs))  if runs else 0.0

    return np.array([
        mean_int, edge_den, tex_var,
        feats["contrast_d1"], feats["homogeneity_d1"],
        feats["energy_d1"],   feats["correlation_d1"],
        feats["contrast_d8"], feats["homogeneity_d8"],
        feats["energy_d8"],   feats["correlation_d8"],
        phase_frac, fft_r, run_mean, run_var,
    ], dtype=np.float32)

FEATURE_NAMES = [
    "mean_intensity", "edge_density", "texture_variance",
    "contrast_d1", "homogeneity_d1", "energy_d1", "correlation_d1",
    "contrast_d8", "homogeneity_d8", "energy_d8", "correlation_d8",
    "phase_fraction", "fft_ratio", "run_len_mean", "run_len_var",
]


def main():
    rng = np.random.default_rng(SEED)

    print("Loading MICRO2D...")
    with h5py.File(FILE_PATH, "r") as f:
        images = f[f"{CLASS}/{CLASS}"][:N_SAMPLES]
        mech   = f[f"{CLASS}/homogenized_mechanical"][:N_SAMPLES]
    E_x = mech[:, 0, 0].astype(np.float32)
    print(f"Loaded {len(images)} {CLASS} images")

    # Extract features from BOTH clean and degraded versions of every image
    print("\nExtracting features (clean + degraded)...")
    X_clean, X_degraded = [], []
    example_clean, example_degraded = None, None

    for i in range(len(images)):
        binary = images[i]
        clean_u = (binary * 255).astype(np.uint8)     # MICRO2D 0/1 -> 0/255
        deg_u   = degrade_image(binary, rng)

        X_clean.append(extract_features(clean_u))
        X_degraded.append(extract_features(deg_u))

        if i == 0:
            example_clean, example_degraded = clean_u, deg_u
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(images)}")

    X_clean    = np.array(X_clean)
    X_degraded = np.array(X_degraded)

    # Same split indices for both domains so the comparison is exact
    idx = np.arange(len(E_x))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)

    # ── The 2x2 domain adaptation matrix ──────────────────────────────────
    print("\n" + "=" * 62)
    print("DOMAIN ADAPTATION MATRIX")
    print("=" * 62)

    def run(Xtr, Xte, label):
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(Xtr[tr], E_x[tr])
        p    = rf.predict(Xte[te])
        r2   = r2_score(E_x[te], p)
        rmse = np.sqrt(mean_squared_error(E_x[te], p))
        print(f"{label:<34} R2={r2:7.4f}   RMSE={rmse:7.1f}")
        return r2, rmse, p

    r2_cc, rmse_cc, p_cc = run(X_clean,    X_clean,    "train clean    -> test clean")
    r2_cd, rmse_cd, p_cd = run(X_clean,    X_degraded, "train clean    -> test degraded")
    r2_dc, rmse_dc, _    = run(X_degraded, X_clean,    "train degraded -> test clean")
    r2_dd, rmse_dd, p_dd = run(X_degraded, X_degraded, "train degraded -> test degraded")

    print("\n" + "=" * 62)
    print("INTERPRETATION")
    print("=" * 62)
    print(f"Baseline (clean/clean):              R2 = {r2_cc:.4f}")
    print(f"Domain gap (clean-trained on real):  R2 = {r2_cd:.4f}"
          f"   drop of {r2_cc - r2_cd:.4f}")
    print(f"Domain-matched training:             R2 = {r2_dd:.4f}"
          f"   recovers {r2_dd - r2_cd:+.4f}")
    print()
    print(f"Pure imaging-artifact penalty:  {r2_cc - r2_cd:.4f} R2")
    print(f"Recoverable by matched training: {r2_dd - r2_cd:.4f} R2")
    residual = r2_cc - r2_dd
    print(f"Irrecoverable information loss:  {residual:.4f} R2")
    print()
    print(f"For reference, real 9Cr steel YS reached R2 = 0.466.")
    print(f"The clean-to-degraded drop measured here ({r2_cc - r2_cd:.3f}) quantifies")
    print(f"how much of the synthetic-to-real gap is imaging artifacts alone.")

    # Feature stability: which features survive degradation?
    print("\n" + "=" * 62)
    print("FEATURE STABILITY UNDER DEGRADATION")
    print("=" * 62)
    stab = []
    for i, name in enumerate(FEATURE_NAMES):
        c, d = X_clean[:, i], X_degraded[:, i]
        corr = np.corrcoef(c, d)[0, 1] if c.std() > 0 and d.std() > 0 else 0.0
        stab.append({"feature": name, "correlation": corr})
    stab_df = pd.DataFrame(stab).sort_values("correlation", ascending=False)
    print(stab_df.to_string(index=False))
    print("\nFeatures with low correlation are destroyed by imaging artifacts.")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Phase 10: Synthetic-to-Real Domain Gap", fontsize=13, fontweight="bold")

    # Example images
    ax = plt.subplot(2, 3, 1)
    ax.imshow(example_clean, cmap="gray"); ax.axis("off")
    ax.set_title("Clean synthetic (MICRO2D)", fontsize=10)

    ax = plt.subplot(2, 3, 2)
    ax.imshow(example_degraded, cmap="gray"); ax.axis("off")
    ax.set_title("Degraded (simulated micrograph)", fontsize=10)

    # 2x2 matrix heatmap
    ax = plt.subplot(2, 3, 3)
    mat = np.array([[r2_cc, r2_cd], [r2_dc, r2_dd]])
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["test clean", "test degraded"], fontsize=9)
    ax.set_yticklabels(["train clean", "train degraded"], fontsize=9)
    ax.set_title("R² matrix", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontweight="bold", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Predicted vs actual, clean
    ax = plt.subplot(2, 3, 4)
    ax.scatter(E_x[te], p_cc, alpha=0.2, s=6, color="steelblue")
    lims = [E_x[te].min(), E_x[te].max()]
    ax.plot(lims, lims, "r--", lw=1.2)
    ax.set_xlabel("Actual E_x"); ax.set_ylabel("Predicted")
    ax.set_title(f"Clean → clean (R²={r2_cc:.3f})", fontsize=10)

    # Predicted vs actual, domain gap
    ax = plt.subplot(2, 3, 5)
    ax.scatter(E_x[te], p_cd, alpha=0.2, s=6, color="tomato")
    ax.plot(lims, lims, "r--", lw=1.2)
    ax.set_xlabel("Actual E_x"); ax.set_ylabel("Predicted")
    ax.set_title(f"Clean → degraded (R²={r2_cd:.3f})", fontsize=10)

    # Feature stability
    ax = plt.subplot(2, 3, 6)
    top = stab_df.head(10)
    colors = ["#27ae60" if c > 0.8 else "#e67e22" if c > 0.5 else "#c0392b"
              for c in top["correlation"]]
    ax.barh(top["feature"], top["correlation"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Clean vs degraded correlation")
    ax.set_title("Feature stability", fontsize=10)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase10_domain_gap.png")
    plt.savefig(out, dpi=130)
    plt.show()
    print(f"\nPlot saved to {out}")

    stab_df.to_csv(os.path.join(OUTPUT_DIR, "phase10_feature_stability.csv"),
                   index=False)


if __name__ == "__main__":
    main()
