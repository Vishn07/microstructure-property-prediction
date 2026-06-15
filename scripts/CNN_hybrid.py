"""
Phase 6 fixes:
1. NBSA dropped (too few samples, both models fail)
2. Voronoi classes get extra grain-aware features fed into a hybrid model
3. Hybrid architecture: CNN image features + hand-crafted features combined
"""
 
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from skimage.feature import graycomatrix, graycoprops
import cv2
 
FILE_PATH  = r"C:\Users\vishw\ml tut\Project\data\MICRO2D_homogenized.h5"
OUTPUT_DIR = r"C:\Users\vishw\ml tut\Project\outputs"
 
# NBSA dropped: too few samples (1634), both RF and CNN failed on it
CLASSES = [
    "AngEllipse", "GRF", "RandomEllipse", "VoidSmall",
    "VoidSmallBig", "VoronoiLarge", "VoronoiMedium",
    "VoronoiMediumSpaced", "VoronoiSmall"
]
 
BATCH_SIZE   = 64
EPOCHS       = 40
LR_HEAD      = 1e-3
LR_FULL      = 1e-5
FREEZE_EPOCHS = 5
PATIENCE     = 8
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

#RF features for hybrid model
def extract_features(img):
    """
    9 physically meaningful features.
    For Voronoi classes, grain size statistics involved
    """
    img_f  = img.astype(np.float64)
    img_u  = (img_f * 255).astype(np.uint8)
    img_64 = (img_u // 4).astype(np.uint8)
 
    # Basic
    mean_int = img_f.mean()
    edges    = cv2.Canny(img_u, 100, 200)
    edge_den = edges.mean() / 255.0
    tex_var  = float(img_f.var() * 255 * 255)
 
    # GLCM at distance 1 and 8 (captures both fine and coarse structure)
    feats = {}
    for d in [1, 8]:
        gcm = graycomatrix(img_64, distances=[d], angles=[0],
                           levels=64, symmetric=True, normed=True)
        feats[f"contrast_d{d}"]    = graycoprops(gcm, "contrast")[0, 0]
        feats[f"homogeneity_d{d}"] = graycoprops(gcm, "homogeneity")[0, 0]
        feats[f"energy_d{d}"]      = graycoprops(gcm, "energy")[0, 0]
        feats[f"correlation_d{d}"] = graycoprops(gcm, "correlation")[0, 0]
 
    # Phase fraction
    phase_frac = float((img == 1).sum()) / img.size
 
    # FFT low-frequency ratio (captures grain scale)
    f     = np.fft.fft2(img_f)
    mag   = np.abs(np.fft.fftshift(f)) ** 2
    h, w  = mag.shape
    cx, cy = h // 2, w // 2
    radius = min(h, w) // 10
    y, x  = np.ogrid[:h, :w]
    mask  = (x - cx)**2 + (y - cy)**2 <= radius**2
    fft_r = float(mag[mask].sum() / (mag.sum() + 1e-10))
 
    # Run length mean (critical for Voronoi — measures grain size directly)
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
 
    # Chord length variance (how irregular are the grain sizes?)
    run_len_var = float(np.var(run_lengths)) if run_lengths else 0.0
 
    return np.array([
        mean_int,
        edge_den,
        tex_var,
        feats["contrast_d1"],
        feats["homogeneity_d1"],
        feats["energy_d1"],
        feats["correlation_d1"],
        feats["contrast_d8"],
        feats["homogeneity_d8"],
        feats["energy_d8"],
        feats["correlation_d8"],
        phase_frac,
        fft_r,
        run_len_mean,
        run_len_var,
    ], dtype=np.float32)
 
N_HANDCRAFTED = 15
 #Hybrid dataset
class HybridDataset(Dataset):
    """
    Returns (image_tensor, feature_vector, target) for each sample.
    The model receives both the raw image AND the hand-crafted features.
    This is the core fix: the CNN no longer has to rediscover phase fraction,
    grain size, and GLCM from scratch. They are handed to it directly.
    """
    def __init__(self, file_path, index_list, hand_feats, targets, augment=False):
        self.file_path  = file_path
        self.index_list = index_list
        self.hand_feats = torch.tensor(hand_feats, dtype=torch.float32)
        self.targets    = torch.tensor(targets, dtype=torch.float32)
        self.augment    = augment
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
        f   = self._get_file()
        img = f[f"{cls}/{cls}"][local_idx].astype(np.float32)
 
        img = np.stack([img, img, img], axis=0)
        img = torch.from_numpy(img)
        img = self.normalize(img)
 
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            img = torch.rot90(img, k, dims=[1, 2])
 
        return img, self.hand_feats[idx], self.targets[idx]
 
#hybrid model
class HybridResNet(nn.Module):
   
    def __init__(self, n_handcrafted):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC — keep everything up to the 512-dim pool output
        self.backbone = nn.Sequential(*list(base.children())[:-1])
 
        combined_dim = 512 + n_handcrafted
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
 
    def forward(self, img, hand_feats):
        visual = self.backbone(img)           # (B, 512, 1, 1)
        visual = visual.view(visual.size(0), -1)  # (B, 512)
        combined = torch.cat([visual, hand_feats], dim=1)  # (B, 527)
        return self.head(combined).squeeze(1)
 
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
 
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
 
 
# ── Train / eval ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, feats, targets in loader:
        imgs, feats, targets = imgs.to(device), feats.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(imgs, feats)
        loss  = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)
 
 
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, feats, targets in loader:
            imgs, feats, targets = imgs.to(device), feats.to(device), targets.to(device)
            preds = model(imgs, feats)
            total_loss += criterion(preds, targets).item() * len(imgs)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_targets)
#main run
def main():
   
 
    index_list = []
    all_ex     = []
    all_feats  = []
 
    with h5py.File(FILE_PATH, "r") as f:
        for cls in CLASSES:
            images = f[f"{cls}/{cls}"][:]
            mech   = f[f"{cls}/homogenized_mechanical"][:]
            ex     = mech[:, 0, 0]
            n      = len(images)
            print(f"  {cls}: {n} samples — extracting features...")
            for i in range(n):
                feat = extract_features(images[i])
                all_feats.append(feat)
                index_list.append((cls, i))
            all_ex.extend(ex.tolist())
            print("done.")
 
    all_ex    = np.array(all_ex,   dtype=np.float32)
    all_feats = np.array(all_feats, dtype=np.float32)
 
    # Normalise hand-crafted features to zero mean unit variance
    feat_mean = all_feats.mean(axis=0)
    feat_std  = all_feats.std(axis=0) + 1e-8
    all_feats_norm = (all_feats - feat_mean) / feat_std
 
    # Normalise E_x
    E_x_mean = float(all_ex.mean())
    E_x_std  = float(all_ex.std())
    ex_norm  = (all_ex - E_x_mean) / E_x_std
 
    total = len(index_list)
    print(f"\nTotal: {total} samples (NBSA excluded)")
    print(f"E_x range: {all_ex.min():.1f} to {all_ex.max():.1f}")
 
    # Split
    idx = np.arange(total)
    train_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42)
    train_idx, val_idx  = train_test_split(train_idx, test_size=0.15, random_state=42)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
 
    train_list = [index_list[i] for i in train_idx]
    val_list   = [index_list[i] for i in val_idx]
    test_list  = [index_list[i] for i in test_idx]
 
    train_ds = HybridDataset(FILE_PATH, train_list, all_feats_norm[train_idx], ex_norm[train_idx], augment=True)
    val_ds   = HybridDataset(FILE_PATH, val_list,   all_feats_norm[val_idx],   ex_norm[val_idx],   augment=False)
    test_ds  = HybridDataset(FILE_PATH, test_list,  all_feats_norm[test_idx],  ex_norm[test_idx],  augment=False)
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
 
    # Model
    model = HybridResNet(N_HANDCRAFTED).to(DEVICE)
    model.freeze_backbone()
 
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nHybrid ResNet-18  |  Total params: {total_params:,}")
    print(f"Strategy: freeze backbone {FREEZE_EPOCHS} epochs, then unfreeze")
    print(f"Hand-crafted features: {N_HANDCRAFTED} (including Voronoi grain stats)\n")
 
    train_losses, val_losses = [], []
    best_val_loss   = float("inf")
    best_model_path = os.path.join(OUTPUT_DIR, "hybrid_resnet_best.pth")
    patience_counter = 0
    unfrozen = False
 
    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1 and not unfrozen:
            print(f"\nEpoch {epoch}: Unfreezing backbone (LR -> {LR_FULL})")
            model.unfreeze_backbone()
            for g in optimizer.param_groups:
                g["lr"] = LR_FULL
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )
            patience_counter = 0
            best_val_loss = float("inf")
            unfrozen = True
 
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_preds, val_targets = eval_epoch(model, val_loader, criterion, DEVICE)
 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
 
        val_preds_orig   = val_preds   * E_x_std + E_x_mean
        val_targets_orig = val_targets * E_x_std + E_x_mean
        val_r2   = r2_score(val_targets_orig, val_preds_orig)
        val_rmse = np.sqrt(mean_squared_error(val_targets_orig, val_preds_orig))
 
        phase = "frozen" if not unfrozen else "full  "
        print(f"[{phase}] Epoch {epoch:02d}/{EPOCHS}  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
              f"R2: {val_r2:.4f}  RMSE: {val_rmse:.1f}")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "E_x_mean":   E_x_mean,
                "E_x_std":    E_x_std,
                "feat_mean":  feat_mean.tolist(),
                "feat_std":   feat_std.tolist(),
            }, best_model_path)
            print(f"  Best model saved (epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and unfrozen:
                print(f"\nEarly stopping after epoch {epoch}.")
                break
 
    # Final evaluation
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
 
    _, test_preds, test_targets = eval_epoch(model, test_loader, criterion, DEVICE)
    test_preds_orig   = test_preds   * E_x_std + E_x_mean
    test_targets_orig = test_targets * E_x_std + E_x_mean
 
    r2   = r2_score(test_targets_orig, test_preds_orig)
    rmse = np.sqrt(mean_squared_error(test_targets_orig, test_preds_orig))
    mean_y = test_targets_orig.mean()
 
    print(f"\nFinal Test Results")
    print(f"R2:             {r2:.4f}")
    print(f"RMSE:           {rmse:.2f} units")
    print(f"RMSE % of mean: {(rmse/mean_y)*100:.1f}%")
    print(f"Best epoch:     {checkpoint['epoch']}")
    print(f"\nComparison")
    print(f"Random Forest (7 features):       R2=0.9077  RMSE=186")
    print(f"CNN v2 (87k, pure image):         R2=0.8400  RMSE=189")
    print(f"Hybrid CNN v3 (image + features): R2={r2:.4f}  RMSE={rmse:.0f}")
 
    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 6: Hybrid ResNet (image + hand-crafted features)", fontsize=13, fontweight="bold")
 
    ax = axes[0]
    ax.plot(train_losses, label="Train loss", color="steelblue")
    ax.plot(val_losses,   label="Val loss",   color="tomato")
    ax.axvline(x=FREEZE_EPOCHS - 0.5, color="gray", linestyle="--", label="Unfreeze")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised)")
    ax.set_title("Learning Curves")
    ax.legend()
 
    ax = axes[1]
    ax.scatter(test_targets_orig, test_preds_orig, alpha=0.2, s=6, color="steelblue")
    ax.plot([test_targets_orig.min(), test_targets_orig.max()],
            [test_targets_orig.min(), test_targets_orig.max()],
            "r--", linewidth=1.5)
    ax.set_xlabel("Actual E_x")
    ax.set_ylabel("Predicted E_x")
    ax.set_title(f"Predicted vs Actual (R2={r2:.4f})")
 
    ax = axes[2]
    model_names = ["RF\n7 feat", "CNN v2\n87k", "Hybrid\nv3"]
    r2_vals     = [0.9077, 0.8400, r2]
    colors      = ["#b0c4de", "#e67e22", "#2e86c1"]
    bars = ax.bar(model_names, r2_vals, color=colors, width=0.4)
    ax.set_ylim(min(0.80, r2 - 0.05), 1.0)
    ax.set_ylabel("R2")
    ax.set_title("Model Comparison")
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
 
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase6_hybrid_results.png")
    plt.savefig(out, dpi=130)
    plt.show()
 
 
if __name__ == "__main__":
    main()
 
