import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from skimage.feature import graycomatrix, graycoprops
import cv2
FILE_PATH  = r"C:\Users\vishw\ml tut\Project\data\MICRO2D_homogenized.h5"
OUTPUT_DIR = r"C:\Users\vishw\ml tut\Project\outputs"
CNN_MODEL  = r"C:\Users\vishw\ml tut\Project\outputs\resnet18_v2_best.pth"
 
CLASSES = [
    "AngEllipse", "GRF", "NBSA", "RandomEllipse", "VoidSmall",
    "VoidSmallBig", "VoronoiLarge", "VoronoiMedium",
    "VoronoiMediumSpaced", "VoronoiSmall"
]
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
#RF
def extract_features(img):
    img_f = img.astype(np.float64)
    img_u = (img_f * 255).astype(np.uint8)
    img_64 = (img_u // 4).astype(np.uint8)
 
    mi = img_f.mean()
    edges = cv2.Canny(img_u, 100, 200)
    ed = edges.mean() / 255.0
    tv = float(img_f.var() * 255 * 255)
 
    gcm = graycomatrix(img_64, distances=[1], angles=[0],
                       levels=64, symmetric=True, normed=True)
    c  = graycoprops(gcm, "contrast")[0, 0]
    h  = graycoprops(gcm, "homogeneity")[0, 0]
    en = graycoprops(gcm, "energy")[0, 0]
    co = graycoprops(gcm, "correlation")[0, 0]
 
    pf = float((img == 1).sum()) / img.size
 
    f = np.fft.fft2(img_f)
    mag = np.abs(np.fft.fftshift(f)) ** 2
    hh, w = mag.shape
    cx, cy = hh // 2, w // 2
    r = min(hh, w) // 10
    y, x = np.ogrid[:hh, :w]
    mask = (x - cx)**2 + (y - cy)**2 <= r**2
    fft_r = float(mag[mask].sum() / (mag.sum() + 1e-10))
 
    return [mi, ed, tv, c, h, en, co, pf, fft_r]
 
RF_FEATURES = [
    "mean_intensity", "edge_density", "texture_variance",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy",
    "glcm_correlation", "phase_fraction", "fft_energy_ratio"
]
#CNN
class SimpleDataset(Dataset):
    def __init__(self, images):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
        # pre-process all at once (per-class batches are small enough)
        imgs = torch.from_numpy(images.astype(np.float32))   # (N,256,256)
        imgs = imgs.unsqueeze(1).expand(-1, 3, -1, -1).clone()
        self.images = self.normalize(imgs)
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, idx):
        return self.images[idx]
 
 
def cnn_predict(model, images, E_x_mean, E_x_std):
    ds     = SimpleDataset(images)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    preds  = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch).squeeze(1).cpu().numpy()
            preds.extend(out)
    preds = np.array(preds) * E_x_std + E_x_mean
    return preds
 
 
def build_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )
    return model
 
 #Main
def main():
    # Load CNN
    print("Loading CNN model...")
    checkpoint = torch.load(CNN_MODEL, map_location=DEVICE, weights_only=True)
    cnn = build_model().to(DEVICE)
    cnn.load_state_dict(checkpoint["model_state"])
    E_x_mean = checkpoint["E_x_mean"]
    E_x_std  = checkpoint["E_x_std"]
    print(f"  Loaded from epoch {checkpoint['epoch']}")
 
    results = []
 
    with h5py.File(FILE_PATH, "r") as f:
        for cls in CLASSES:
            print(f"\nProcessing {cls}...")
            images = f[f"{cls}/{cls}"][:]
            mech   = f[f"{cls}/homogenized_mechanical"][:]
            E_x    = mech[:, 0, 0].astype(np.float32)
            n      = len(images)
 
            # train/test split — use 20% for test, consistent across classes
            idx = np.arange(n)
            train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
 
            # ── Random Forest ──
            print(f"  Extracting features for RF ({n} images)...")
            feats = []
            for i in idx:
                feats.append(extract_features(images[i]))
            feats = np.array(feats)
 
            X_train = feats[train_idx]
            X_test  = feats[test_idx]
            y_train = E_x[train_idx]
            y_test  = E_x[test_idx]
 
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            rf_r2   = r2_score(y_test, rf_preds)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
 
            # ── CNN ──
            print(f"  Running CNN inference...")
            cnn_preds = cnn_predict(cnn, images[test_idx], E_x_mean, E_x_std)
            cnn_r2   = r2_score(y_test, cnn_preds)
            cnn_rmse = np.sqrt(mean_squared_error(y_test, cnn_preds))
 
            results.append({
                "class":    cls,
                "n":        n,
                "rf_r2":    round(rf_r2,   4),
                "rf_rmse":  round(rf_rmse,  1),
                "cnn_r2":   round(cnn_r2,   4),
                "cnn_rmse": round(cnn_rmse,  1),
            })
 
            print(f"  RF:  R²={rf_r2:.4f}  RMSE={rf_rmse:.1f}")
            print(f"  CNN: R²={cnn_r2:.4f}  RMSE={cnn_rmse:.1f}")
 
    # ── Summary table ──
    df = pd.DataFrame(results)
    print("\n\n── Per-class Results ──────────────────────────────────────────")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUT_DIR, "phase6_per_class.csv"), index=False)
 
    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 6: Per-class R² — Random Forest vs CNN", fontsize=13, fontweight="bold")
 
    x = np.arange(len(CLASSES))
    w = 0.35
 
    ax = axes[0]
    ax.bar(x - w/2, df["rf_r2"],  width=w, label="Random Forest", color="#2e86c1")
    ax.bar(x + w/2, df["cnn_r2"], width=w, label="CNN ResNet-18", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("R² per class")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(0.91, color="#2e86c1", linestyle="--", alpha=0.4, label="RF overall")
 
    ax = axes[1]
    ax.bar(x - w/2, df["rf_rmse"],  width=w, label="Random Forest", color="#2e86c1")
    ax.bar(x + w/2, df["cnn_rmse"], width=w, label="CNN ResNet-18", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE per class")
    ax.legend()
 
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase6_per_class.png")
    plt.savefig(out, dpi=130)
    plt.show()
    print(f"\nPlot saved to {out}")
    print(f"CSV  saved to {os.path.join(OUTPUT_DIR, 'phase6_per_class.csv')}")
 
 
if __name__ == "__main__":
    main()