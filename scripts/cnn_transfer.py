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

FILE_PATH  = r"C:\Users\vishw\ml tut\Project\data\MICRO2D_homogenized.h5"
OUTPUT_DIR = r"C:\Users\vishw\ml tut\Project\outputs"
 
BATCH_SIZE  = 64
EPOCHS      = 40
LR_HEAD     = 1e-3    # phase A: head only — can be aggressive
LR_FULL     = 1e-5    # phase B: full network — must be gentle
WEIGHT_DECAY = 1e-4
FREEZE_EPOCHS = 5     
PATIENCE    = 8
CLASSES = [
    "AngEllipse", "GRF", "NBSA", "RandomEllipse", "VoidSmall",
    "VoidSmallBig", "VoronoiLarge", "VoronoiMedium",
    "VoronoiMediumSpaced", "VoronoiSmall"
]
 
print("Starting...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
class MicrostructureDataset(Dataset):
   
    def __init__(self, file_path, index_list, targets, augment=False):
      
        self.file_path  = file_path
        self.index_list = index_list
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
        f = self._get_file()
        img = f[f"{cls}/{cls}"][local_idx].astype(np.float32)  # (256,256) 0/1
 
        # 3-channel
        img = np.stack([img, img, img], axis=0)   # (3,256,256)
        img = torch.from_numpy(img)
        img = self.normalize(img)
 
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            img = torch.rot90(img, k, dims=[1, 2])
 
        return img, self.targets[idx]
#model
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )
    return model
 
 
def freeze_backbone(model):
    """Freeze everything except the final FC head."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone frozen. Trainable params: {trainable:,} (head only)")
 
 
def unfreeze_all(model, optimizer, new_lr):
    """Unfreeze all layers and reset optimizer with lower LR."""
    for param in model.parameters():
        param.requires_grad = True
    # Update optimizer LR for all param groups
    for g in optimizer.param_groups:
        g["lr"] = new_lr
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  All layers unfrozen. Trainable params: {trainable:,}. LR → {new_lr}")
#train
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(imgs).squeeze(1)
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
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs).squeeze(1)
            total_loss += criterion(preds, targets).item() * len(imgs)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_targets)
#main loop
def main():
    # 1. Load all indices and E_x values (not images — just metadata)
    print("Reading metadata from HDF5 (fast)...")
    index_list = []   # (class_name, local_idx)
    all_ex     = []   # E_x values
 
    with h5py.File(FILE_PATH, "r") as f:
        for cls in CLASSES:
            mech = f[f"{cls}/homogenized_mechanical"][:]
            ex   = mech[:, 0, 0]
            n    = len(ex)
            for i in range(n):
                index_list.append((cls, i))
            all_ex.extend(ex.tolist())
            print(f"  {cls}: {n} samples")
 
    all_ex = np.array(all_ex, dtype=np.float32)
    print(f"\nTotal: {len(index_list)} samples")
    print(f"E_x range: {all_ex.min():.1f} – {all_ex.max():.1f}")
 
    # Normalise E_x
    E_x_mean = float(all_ex.mean())
    E_x_std  = float(all_ex.std())
    ex_norm  = (all_ex - E_x_mean) / E_x_std
 
    # 2. Split indices (not images)
    idx = np.arange(len(index_list))
    train_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42)
    train_idx, val_idx  = train_test_split(train_idx, test_size=0.15, random_state=42)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
 
    train_list = [index_list[i] for i in train_idx]
    val_list   = [index_list[i] for i in val_idx]
    test_list  = [index_list[i] for i in test_idx]
 
    train_ds = MicrostructureDataset(FILE_PATH, train_list, ex_norm[train_idx], augment=True)
    val_ds   = MicrostructureDataset(FILE_PATH, val_list,   ex_norm[val_idx],   augment=False)
    test_ds  = MicrostructureDataset(FILE_PATH, test_list,  ex_norm[test_idx],  augment=False)
 
    # num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
 
    # 3. Model — start with frozen backbone
    model = build_model().to(DEVICE)
    freeze_backbone(model)
 
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: ResNet-18  |  Total params: {total_params:,}")
    print(f"Strategy: freeze backbone for {FREEZE_EPOCHS} epochs, then unfreeze all")
    print(f"Training for up to {EPOCHS} epochs (patience={PATIENCE})...\n")
 
    train_losses, val_losses = [], []
    best_val_loss   = float("inf")
    best_model_path = os.path.join(OUTPUT_DIR, "resnet18_v2_best.pth")
    patience_counter = 0
    unfrozen = False
 
    for epoch in range(1, EPOCHS + 1):
 
        # Phase B: unfreeze after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1 and not unfrozen:
            print(f"\n── Epoch {epoch}: Unfreezing backbone ──")
            unfreeze_all(model, optimizer, LR_FULL)
            unfrozen = True
            # Reset scheduler and patience after unfreeze
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )
            patience_counter = 0
            best_val_loss = float("inf")   # reset so we save a fresh best
 
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_preds, val_targets = eval_epoch(model, val_loader, criterion, DEVICE)
 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
 
        val_preds_orig   = val_preds   * E_x_std + E_x_mean
        val_targets_orig = val_targets * E_x_std + E_x_mean
        val_r2   = r2_score(val_targets_orig, val_preds_orig)
        val_rmse = np.sqrt(mean_squared_error(val_targets_orig, val_preds_orig))
 
        phase = "A-frozen" if not unfrozen else "B-full  "
        print(f"[{phase}] Epoch {epoch:02d}/{EPOCHS}  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
              f"R²: {val_r2:.4f}  RMSE: {val_rmse:.1f}")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "E_x_mean":    E_x_mean,
                "E_x_std":     E_x_std,
            }, best_model_path)
            print(f"  ✓ Best model saved (epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and unfrozen:
                print(f"\nEarly stopping after epoch {epoch}.")
                break
 
    # 4. Final evaluation
    print("\nLoading best model...")
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
 
    _, test_preds, test_targets = eval_epoch(model, test_loader, criterion, DEVICE)
    test_preds_orig   = test_preds   * E_x_std + E_x_mean
    test_targets_orig = test_targets * E_x_std + E_x_mean
 
    r2   = r2_score(test_targets_orig, test_preds_orig)
    rmse = np.sqrt(mean_squared_error(test_targets_orig, test_preds_orig))
    mean_y = test_targets_orig.mean()
 
    print(f"\n── Final Test Results ──────────────────────────────")
    print(f"R²:             {r2:.4f}")
    print(f"RMSE:           {rmse:.2f} units")
    print(f"RMSE % of mean: {(rmse/mean_y)*100:.1f}%")
    print(f"Best epoch:     {checkpoint['epoch']}")
    print(f"\n── Comparison ──────────────────────────────────────")
    print(f"Random Forest (7 features):  R²=0.9077, RMSE=186")
    print(f"CNN ResNet-18 v1 (10k GRF):  R²=0.7642, RMSE=306")
    print(f"CNN ResNet-18 v2 (87k all):  R²={r2:.4f}, RMSE={rmse:.0f}")
 
    # 5. Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 5 v2: ResNet-18 on all 87k samples", fontsize=13, fontweight="bold")
 
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
    ax.set_title(f"Predicted vs Actual  (R²={r2:.4f})")
 
    ax = axes[2]
    models_cmp = ["RF\n7 feat", "RF\n22 feat", "CNN v1\n10k", "CNN v2\n87k"]
    r2_cmp     = [0.9077, 0.9061, 0.7642, r2]
    bar_colors = ["#b0c4de", "#7fb3d3", "#5b9ec9", "#2e86c1"]
    bars = ax.bar(models_cmp, r2_cmp, color=bar_colors, width=0.5)
    ax.set_ylim(min(0.70, r2 - 0.05), 1.0)
    ax.set_ylabel("R²")
    ax.set_title("Model Comparison")
    for bar, val in zip(bars, r2_cmp):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=8)
 
    plt.tight_layout()
    out_plot = os.path.join(OUTPUT_DIR, "phase5_v2_results.png")
    plt.savefig(out_plot, dpi=130)
    plt.show()
    print(f"\nPlot saved to {out_plot}")
 
 
if __name__ == "__main__":
    main()
 