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
from tqdm import tqdm
FILE_PATH  = r"C:\Users\vishw\ml tut\Project\data\MICRO2D_homogenized.h5"
OUTPUT_DIR = r"C:\Users\vishw\ml tut\Project\outputs"
size=64
LR=3e-5
EP=30
decay=1e-4
timer=10
print("starting ")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("starting")
class MicrostructureDataset(Dataset):
    """
    Loads binary 0/1 microstructure images from HDF5.
    Converts to 3-channel float32 for ResNet (which expects RGB).
    Normalises with ImageNet stats so pretrained features transfer well.
    """
    def __init__(self, images, targets, augment=False):
        self.images  = images          # numpy array (N, 256, 256) int64
        self.targets = targets.astype(np.float32)
        self.augment = augment
 
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, idx):
        # Binary 0/1 → float32 [0, 1]
        img = self.images[idx].astype(np.float32)
 
        # Stack to 3 channels (ResNet expects RGB)
        img = np.stack([img, img, img], axis=0)     # (3, 256, 256)
        img = torch.from_numpy(img)                 # tensor
 
        # Data augmentation on training set only
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[1])
            # Random 90° rotation (microstructures are isotropic — safe)
            k = torch.randint(0, 4, (1,)).item()
            img = torch.rot90(img, k, dims=[1, 2])
 
        img = self.normalize(img)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img, target
def build_model():
    """
    ResNet-18 pretrained on ImageNet
    
     fine-tune the ENTIRE network  because microstructure
    textures are very different from natural images — deeper layers need to adapt.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
 
    # Replace final FC layer: 512 → 256 → 1
    in_features = model.fc.in_features          # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )
    return model
 # ── Training loop ─────────────────────────────────────────────────────────────
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
def main():
    # 1. Load data
    print("Loading data from HDF5...")
    with h5py.File(FILE_PATH, "r") as f:
        images = f[f"{"GRF"}/{"GRF"}"][:10000]          # (N, 256, 256)
        mech   = f[f"{"GRF"}/homogenized_mechanical"][:10000]
    E_x = mech[:, 0, 0].astype(np.float32)
    print(f"Loaded {len(images)} images. E_x range: {E_x.min():.1f} – {E_x.max():.1f}")
 
    # Normalise E_x to ~[0,1] for stable training, remember scale for inverse transform
    E_x_mean = E_x.mean()
    E_x_std  = E_x.std()
    E_x_norm = (E_x - E_x_mean) / E_x_std
 
    # 2. Split
    idx = np.arange(len(images))
    train_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42)
    train_idx, val_idx  = train_test_split(train_idx, test_size=0.15, random_state=42)
 
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
 
    train_ds = MicrostructureDataset(images[train_idx], E_x_norm[train_idx], augment=True)
    val_ds   = MicrostructureDataset(images[val_idx],   E_x_norm[val_idx],   augment=False)
    test_ds  = MicrostructureDataset(images[test_idx],  E_x_norm[test_idx],  augment=False)
 
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
 
    # 3. Model, optimizer, scheduler
    model     = build_model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
 
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ResNet-18  |  Total params: {total_params:,}  |  Trainable: {trainable_params:,}")
 
    # 4. Training loop
    print(f"\nTraining for up to {EP} epochs (early stopping patience={timer})...\n")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_path = os.path.join(OUTPUT_DIR, "resnet18_best.pth")
    patience_counter = 0
 
    for epoch in range(1, EP + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_preds, val_targets = eval_epoch(model, val_loader, criterion, DEVICE)
 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
 
        # Convert back to original scale for RMSE display
        val_preds_orig   = val_preds   * E_x_std + E_x_mean
        val_targets_orig = val_targets * E_x_std + E_x_mean
        val_r2   = r2_score(val_targets_orig, val_preds_orig)
        val_rmse = np.sqrt(mean_squared_error(val_targets_orig, val_preds_orig))
 
        print(f"Epoch {epoch:02d}/{EP}  "
              f"Train loss: {train_loss:.4f}  "
              f"Val loss: {val_loss:.4f}  "
              f"Val R²: {val_r2:.4f}  "
              f"Val RMSE: {val_rmse:.1f}")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "E_x_mean": float(E_x_mean),   
                "E_x_std":  float(E_x_std), 
            }, best_model_path)
            print(f"  ✓ Best model saved (epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= timer:
                print(f"\nEarly stopping triggered after epoch {epoch}.")
                break
 
    # 5. Evaluate on test set using best model
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
 
    _, test_preds, test_targets = eval_epoch(model, test_loader, criterion, DEVICE)
    test_preds_orig   = test_preds   * E_x_std + E_x_mean
    test_targets_orig = test_targets * E_x_std + E_x_mean
 
    r2   = r2_score(test_targets_orig, test_preds_orig)
    rmse = np.sqrt(mean_squared_error(test_targets_orig, test_preds_orig))
    mean_y = test_targets_orig.mean()
 
    print(f"\n── Final Test Results ────────────────────────────────")
    print(f"R²:               {r2:.4f}")
    print(f"RMSE:             {rmse:.2f} units")
    print(f"RMSE % of mean:   {(rmse/mean_y)*100:.1f}%")
    print(f"Best epoch:       {checkpoint['epoch']}")
    print(f"\n── Comparison ────────────────────────────────────────")
    print(f"Random Forest (7 features):  R²=0.9077, RMSE=186")
    print(f"CNN ResNet-18 (this run):    R²={r2:.4f}, RMSE={rmse:.0f}")
 
    # 6. Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 5: ResNet-18 CNN Results", fontsize=13, fontweight="bold")
 
    # Learning curves
    ax = axes[0]
    ax.plot(train_losses, label="Train loss", color="steelblue")
    ax.plot(val_losses,   label="Val loss",   color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised)")
    ax.set_title("Learning Curves")
    ax.legend()
 
    # Predicted vs Actual
    ax = axes[1]
    ax.scatter(test_targets_orig, test_preds_orig, alpha=0.3, s=10, color="steelblue")
    ax.plot([test_targets_orig.min(), test_targets_orig.max()],
            [test_targets_orig.min(), test_targets_orig.max()],
            "r--", linewidth=1.5)
    ax.set_xlabel("Actual E_x")
    ax.set_ylabel("Predicted E_x")
    ax.set_title(f"Predicted vs Actual  (R²={r2:.4f})")
 
    # R² comparison bar chart
    ax = axes[2]
    models_cmp = ["RF\n7 features", "RF\n22 features", "ResNet-18\nCNN"]
    r2_cmp     = [0.9077, 0.9061, r2]
    bar_colors = ["#b0c4de", "#7fb3d3", "#2e86c1"]
    bars = ax.bar(models_cmp, r2_cmp, color=bar_colors, width=0.4)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("R²")
    ax.set_title("Model Comparison")
    for bar, val in zip(bars, r2_cmp):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
 
    plt.tight_layout()
    out_plot = os.path.join(OUTPUT_DIR, "phase5_cnn_results.png")
    plt.savefig(out_plot, dpi=130)
    plt.show()
    print(f"\nPlot saved to {out_plot}")
    print(f"Model saved to {best_model_path}")
 
 
if __name__ == "__main__":
    main()