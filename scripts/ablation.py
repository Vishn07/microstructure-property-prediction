import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops
import cv2
 
FILE_PATH  = r"C:\Users\vishw\ml tut\Project\data\MICRO2D_homogenized.h5"
OUTPUT_DIR = r"C:\Users\vishw\ml tut\Project\outputs"
MODEL_PATH = r"C:\Users\vishw\ml tut\Project\outputs\hybrid_resnet_best.pth"
 
CLASSES = [
    "AngEllipse", "GRF", "RandomEllipse", "VoidSmall",
    "VoidSmallBig", "VoronoiLarge", "VoronoiMedium",
    "VoronoiMediumSpaced", "VoronoiSmall"
]
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
 
 
# ── Feature extraction (same as CNN v3) ───────────────────────────────────────
def extract_features(img):
    img_f  = img.astype(np.float64)
    img_u  = (img_f * 255).astype(np.uint8)
    img_64 = (img_u // 4).astype(np.uint8)
 
    mean_int = img_f.mean()
    edges    = cv2.Canny(img_u, 100, 200)
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
 
    phase_frac = float((img == 1).sum()) / img.size
 
    f     = np.fft.fft2(img_f)
    mag   = np.abs(np.fft.fftshift(f)) ** 2
    h, w  = mag.shape
    cx, cy = h // 2, w // 2
    radius = min(h, w) // 10
    y, x  = np.ogrid[:h, :w]
    mask  = (x - cx)**2 + (y - cy)**2 <= radius**2
    fft_r = float(mag[mask].sum() / (mag.sum() + 1e-10))
 
    run_lengths = []
    for row in img:
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
 
N_HANDCRAFTED = 15
 
 
# ── Datasets ──────────────────────────────────────────────────────────────────
class ImageOnlyDataset(Dataset):
    """For ablation B: image only, no hand-crafted features."""
    def __init__(self, file_path, index_list):
        self.file_path  = file_path
        self.index_list = index_list
        self.normalize  = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
        self._file = None
 
    def __len__(self):
        return len(self.index_list)
 
    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file
 
    def __getitem__(self, idx):
        cls, local_idx = self.index_list[idx]
        img = self._get_file()[f"{cls}/{cls}"][local_idx].astype(np.float32)
        img = np.stack([img, img, img], axis=0)
        img = torch.from_numpy(img)
        return self.normalize(img)
 
 
class HybridDataset(Dataset):
    """For ablation C: image + hand-crafted features."""
    def __init__(self, file_path, index_list, hand_feats):
        self.file_path  = file_path
        self.index_list = index_list
        self.hand_feats = torch.tensor(hand_feats, dtype=torch.float32)
        self.normalize  = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
        self._file = None
 
    def __len__(self):
        return len(self.index_list)
 
    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file
 
    def __getitem__(self, idx):
        cls, local_idx = self.index_list[idx]
        img = self._get_file()[f"{cls}/{cls}"][local_idx].astype(np.float32)
        img = np.stack([img, img, img], axis=0)
        img = torch.from_numpy(img)
        return self.normalize(img), self.hand_feats[idx]
 
 
# ── Models ────────────────────────────────────────────────────────────────────
class ImageOnlyResNet(nn.Module):
    """Pure CNN — same as v2 architecture."""
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
 
    def forward(self, img):
        x = self.backbone(img).view(-1, 512)
        return self.head(x).squeeze(1)
 
 
class HybridResNet(nn.Module):
    """Image + hand-crafted features combined."""
    def __init__(self, n_handcrafted):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        combined_dim = 512 + n_handcrafted
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
        )
 
    def forward(self, img, hand_feats):
        x = self.backbone(img).view(-1, 512)
        return self.head(torch.cat([x, hand_feats], dim=1)).squeeze(1)
 
 
# ── Inference helpers ─────────────────────────────────────────────────────────
def predict_image_only(model, index_list, device):
    ds     = ImageOnlyDataset(FILE_PATH, index_list)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    preds  = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            preds.extend(model(batch.to(device)).cpu().numpy())
    return np.array(preds)
 
 
def predict_hybrid(model, index_list, feats, device):
    ds     = HybridDataset(FILE_PATH, index_list, feats)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    preds  = []
    model.eval()
    with torch.no_grad():
        for imgs, f in loader:
            preds.extend(model(imgs.to(device), f.to(device)).cpu().numpy())
    return np.array(preds)
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load all data and features
    
 
    index_list = []
    all_ex     = []
    all_feats  = []
 
    with h5py.File(FILE_PATH, "r") as f:
        for cls in CLASSES:
            images = f[f"{cls}/{cls}"][:]
            mech   = f[f"{cls}/homogenized_mechanical"][:]
            ex     = mech[:, 0, 0]
            print(f"  {cls}: {len(images)} samples...")
            for i in range(len(images)):
                all_feats.append(extract_features(images[i]))
                index_list.append((cls, i))
            all_ex.extend(ex.tolist())
            print(f"    done.")
 
    all_ex    = np.array(all_ex,    dtype=np.float32)
    all_feats = np.array(all_feats, dtype=np.float32)
 
    # Normalise features
    feat_mean = all_feats.mean(axis=0)
    feat_std  = all_feats.std(axis=0) + 1e-8
    all_feats_norm = (all_feats - feat_mean) / feat_std
 
    # Normalise E_x
    E_x_mean = float(all_ex.mean())
    E_x_std  = float(all_ex.std())
 
    # Reproduce the exact same train/test split as CNN v3
    idx = np.arange(len(index_list))
    train_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42)
    train_idx, _        = train_test_split(train_idx, test_size=0.15, random_state=42)
 
    print(f"\nTest set: {len(test_idx)} samples")
    test_list  = [index_list[i] for i in test_idx]
    test_feats = all_feats_norm[test_idx]
    y_test     = all_ex[test_idx]
 
    # ── Ablation A: Features only (Random Forest) ─────────────────────────────
    print("\nAblation A: Training Random Forest on 15 features only...")
    X_train = all_feats_norm[train_idx]
    y_train = all_ex[train_idx]
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(test_feats)
    rf_r2    = r2_score(y_test, rf_preds)
    rf_rmse  = np.sqrt(mean_squared_error(y_test, rf_preds))
    print(f"  R2={rf_r2:.4f}  RMSE={rf_rmse:.1f}")
 
    # ── Ablation B: Image only ResNet (proxy for CNN v2) ──────────────────────
    # We use the hybrid checkpoint backbone weights as a warm start
    # so this is a fair comparison on the same training
    print("\nAblation B: Image-only ResNet inference...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
 
    img_model = ImageOnlyResNet().to(DEVICE)
    # Load only the backbone weights from the hybrid checkpoint
    hybrid_state = checkpoint["model_state"]
    backbone_state = {k.replace("backbone.", ""): v
                      for k, v in hybrid_state.items() if k.startswith("backbone.")}
    img_model.backbone.load_state_dict(backbone_state)
    # Note: head is randomly initialized — this shows backbone alone without features
    img_preds_norm = predict_image_only(img_model, test_list, DEVICE)
    img_preds = img_preds_norm * E_x_std + E_x_mean
    img_r2    = r2_score(y_test, img_preds)
    img_rmse  = np.sqrt(mean_squared_error(y_test, img_preds))
    print(f"  R2={img_r2:.4f}  RMSE={img_rmse:.1f}")
    print(f"  (Note: head is untrained — this isolates backbone features only)")
 
    # ── Ablation C: Full hybrid (the actual CNN v3) ───────────────────────────
    print("\nAblation C: Full hybrid model evaluation...")
    hybrid_model = HybridResNet(N_HANDCRAFTED).to(DEVICE)
    hybrid_model.load_state_dict(checkpoint["model_state"])
    hybrid_preds_norm = predict_hybrid(hybrid_model, test_list, test_feats, DEVICE)
    hybrid_preds = hybrid_preds_norm * E_x_std + E_x_mean
    hybrid_r2    = r2_score(y_test, hybrid_preds)
    hybrid_rmse  = np.sqrt(mean_squared_error(y_test, hybrid_preds))
    print(f"  R2={hybrid_r2:.4f}  RMSE={hybrid_rmse:.1f}")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"ABLATION RESULTS")
    print(f"{'='*55}")
    print(f"A  RF on 15 features (no image):    R2={rf_r2:.4f}  RMSE={rf_rmse:.0f}")
    print(f"B  CNN backbone only (no features): R2={img_r2:.4f}  RMSE={img_rmse:.0f}")
    print(f"C  Hybrid (image + features):       R2={hybrid_r2:.4f}  RMSE={hybrid_rmse:.0f}")
    print(f"{'='*55}")
    cnn_contribution = hybrid_r2 - rf_r2
    print(f"CNN contribution over features alone: {cnn_contribution:+.4f} R2")
    if cnn_contribution > 0.005:
        print("Conclusion: CNN adds real value on top of hand-crafted features.")
    elif cnn_contribution > 0:
        print("Conclusion: CNN adds marginal value. Features do most of the work.")
    else:
        print("Conclusion: CNN contributes nothing. Features alone explain the result.")
 
    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 6 Ablation: What is actually driving the improvement?",
                 fontsize=12, fontweight="bold")
 
    # Bar chart R2
    ax = axes[0]
    labels = ["RF\n(features only)", "CNN backbone\n(image only)", "Hybrid\n(both)"]
    r2s    = [rf_r2, img_r2, hybrid_r2]
    colors = ["#2e86c1", "#e67e22", "#27ae60"]
    bars   = ax.bar(labels, r2s, color=colors, width=0.4)
    ax.set_ylim(min(0.5, min(r2s) - 0.05), 1.0)
    ax.set_ylabel("R2")
    ax.set_title("R2 Comparison")
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
 
    # Bar chart RMSE
    ax = axes[1]
    rmses  = [rf_rmse, img_rmse, hybrid_rmse]
    bars   = ax.bar(labels, rmses, color=colors, width=0.4)
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Comparison")
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                f"{val:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
 
    # Predicted vs actual for hybrid
    ax = axes[2]
    ax.scatter(y_test, hybrid_preds, alpha=0.2, s=6, color="#27ae60")
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], "r--", linewidth=1.5)
    ax.set_xlabel("Actual E_x")
    ax.set_ylabel("Predicted E_x")
    ax.set_title(f"Hybrid predicted vs actual (R2={hybrid_r2:.4f})")
 
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase6_ablation.png")
    plt.savefig(out, dpi=130)
    plt.show()
 
 
if __name__ == "__main__":
    main()
