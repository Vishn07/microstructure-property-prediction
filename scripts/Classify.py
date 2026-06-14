"""
Phase 7: Real Steel Micrograph Classification
==============================================
Predicts microstructure class (spheroidite / network / pearlite)
from real UHCS micrograph images using ResNet-18.

Key differences from Phase 5/6:
- Classification not regression (3 classes not a continuous value)
- Real grayscale images not synthetic binary images
- Magnification fed as extra input to handle multi-scale images
- Class imbalance handled with weighted loss
- Much smaller dataset (~598 images vs 87k)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR  = r"C:\Users\vishw\.cache\kagglehub\datasets\safi842\highcarbon-micrographs\versions\2\For Training\Cropped"
METADATA   = r"C:\Users\vishw\.cache\kagglehub\datasets\safi842\highcarbon-micrographs\versions\2\new_metadata.xlsx"
OUTPUT_DIR = r"C:\Users\vishw\ml tut\Project\outputs"

BATCH_SIZE = 16      # small — only ~598 images total
EPOCHS  = 40
LR_HEAD  = 1e-3
LR_FULL  = 1e-5
FREEZE_EPOCHS = 8      # more freeze epochs — tiny dataset needs stable head first
PATIENCE = 10
IMG_SIZE = 224     # ResNet standard input size

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
(f"Device: {DEVICE}")


CLASS_MAP = {
    "spheroidite":             "spheroidite",
    "spheroidite+widmanstatten": "spheroidite",
    "network":                 "network",
    "pearlite":                "pearlite",
    "pearlite+spheroidite":    "pearlite",
    "pearlite+widmanstatten":  "pearlite",
}
CLASSES    = ["spheroidite", "network", "pearlite"]
CLASS2IDX  = {c: i for i, c in enumerate(CLASSES)}
 
 
# ── Magnification parsing ─────────────────────────────────────────────────────
def parse_magnification(mag_str):
    """
    Parse magnification string like '1964X', '982x', '98X' into a float.
    Returns NaN if unparseable.
    """
    if pd.isna(mag_str):
        return np.nan
    s = str(mag_str).upper().replace("X", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan
 
 
# Data
class UHCSDataset(Dataset):
    """
    Loads real UHCS steel micrographs.
    Returns image_tensor, magnification_tensor, class_label
    Magnification is log-normalized 
    """
    def __init__(self, records, mag_mean, mag_std, augment=False):
        """
        records: list of dicts with keys: path, mag_norm, label
        """
        self.records  = records
        self.mag_mean = mag_mean
        self.mag_std  = mag_std
        self.augment  = augment
 
        # Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
 
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            ])
        else:
            self.transform = transforms.CenterCrop(IMG_SIZE)
 
    def __len__(self):
        return len(self.records)
 
    def __getitem__(self, idx):
        rec = self.records[idx]
 
        # Load grayscale image and convert to 3-channel
        img = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return zeros if image not found
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        
        # Resize to slightly larger than IMG_SIZE then crop
        img = cv2.resize(img, (256, 256))
        img = np.stack([img, img, img], axis=2)   # (256,256,3)
 
        # Convert to tensor (H,W,C) -> (C,H,W)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = self.transform(img)
        img = self.normalize(img)
 
        # Log-normalized magnification
        mag = torch.tensor([rec["mag_norm"]], dtype=torch.float32)
 
        label = torch.tensor(rec["label"], dtype=torch.long)
        return img, mag, label
 
 
# Model 
class MicrographClassifier(nn.Module):
    """
    ResNet-18 backbone + magnification input -> 3-class classifier.
    Magnification is concatenated after the backbone pool layer,
    same hybrid approach as CNN v3 but for classification.
    """
    def __init__(self, n_classes=3):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # -> (B,512,1,1)
 
        # mag input: 1 raw value -> 32 dim embedding
        self.mag_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )
 
        self.head = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )
 
    def forward(self, img, mag):
        visual  = self.backbone(img).view(-1, 512)
        mag_emb = self.mag_embed(mag)
        combined = torch.cat([visual, mag_emb], dim=1)
        return self.head(combined)
 
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
 
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
 
 
# Train
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, mags, labels in loader:
        imgs, mags, labels = imgs.to(device), mags.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs, mags)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(imgs)
    return total_loss / total, correct / total
 
 
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, mags, labels in loader:
            imgs, mags, labels = imgs.to(device), mags.to(device), labels.to(device)
            logits = model(imgs, mags)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * len(imgs)
            preds       = logits.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += len(imgs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)
 
 
# Main
def main():
    # 1. Load and prepare metadata
    df = pd.read_excel(METADATA)
    df["class"]  = df["primary_microconstituent"].map(CLASS_MAP)
    df["mag_raw"] = df["magnification"].apply(parse_magnification)
 
    # Drop rows with unmapped classes
    df = df.dropna(subset=["class"])
    print(f"Total usable images: {len(df)}")
    print(df["class"].value_counts().to_string())
 
    # Log-normalize magnification (fill missing with median)
    median_mag = np.nanmedian(df["mag_raw"].values)
    df["mag_raw"] = df["mag_raw"].fillna(median_mag)
    df["mag_log"] = np.log1p(df["mag_raw"])
    mag_mean = df["mag_log"].mean()
    mag_std  = df["mag_log"].std() + 1e-8
    df["mag_norm"] = (df["mag_log"] - mag_mean) / mag_std
 
    # 2. Build records list
    # Metadata has 'micrograph2.png', folder has 'Croppedmicrograph2' or 'Croppedmicrograph2.png'
    records = []
    missing = 0
    for _, row in df.iterrows():
        num  = row["path"].replace("micrograph", "").replace(".png", "")
        base = f"Croppedmicrograph{num}"
        img_path = None
        for candidate in [
            os.path.join(IMAGE_DIR, base + ".png"),
            os.path.join(IMAGE_DIR, base + ".jpg"),
            os.path.join(IMAGE_DIR, base),
        ]:
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            missing += 1
            continue
        records.append({
            "path":     img_path,
            "mag_norm": row["mag_norm"],
            "label":    CLASS2IDX[row["class"]],
            "class":    row["class"],
        })
 
    print(f"\nImages found: {len(records)}  Missing: {missing}")
    if len(records) == 0:
        print("ERROR: No images found.")
        return
 
    # 3. Split
    labels_arr = [r["label"] for r in records]
    train_rec, test_rec = train_test_split(
        records, test_size=0.15, random_state=42, stratify=labels_arr
    )
    train_labels = [r["label"] for r in train_rec]
    train_rec, val_rec = train_test_split(
        train_rec, test_size=0.15, random_state=42, stratify=train_labels
    )
    print(f"Train: {len(train_rec)}  Val: {len(val_rec)}  Test: {len(test_rec)}")
 
    # 4. Datasets
    train_ds = UHCSDataset(train_rec, mag_mean, mag_std, augment=True)
    val_ds   = UHCSDataset(val_rec,   mag_mean, mag_std, augment=False)
    test_ds  = UHCSDataset(test_rec,  mag_mean, mag_std, augment=False)
 
    # Weighted sampler to handle class imbalance
    # Gives rarer classes (pearlite, network) a higher chance of being sampled
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[r["label"]] for r in train_rec]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,   num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,     num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,     num_workers=0)
 
    # 5. Model
    model = MicrographClassifier(n_classes=3).to(DEVICE)
    model.freeze_backbone()
 
    # Class-weighted loss — same idea as sampler but at loss level too
    loss_weights = torch.tensor(class_weights / class_weights.sum() * 3,
                                dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: ResNet-18 + mag embedding  |  Total params: {total_params:,}")
    print(f"Classes: {CLASSES}")
    print(f"Freeze backbone for {FREEZE_EPOCHS} epochs then unfreeze")
    print(f"Training for up to {EPOCHS} epochs (patience={PATIENCE})...\n")
 
    train_losses, val_losses = [], []
    train_accs, val_accs     = [], []
    best_val_loss   = float("inf")
    best_model_path = os.path.join(OUTPUT_DIR, "phase7_classifier_best.pth")
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
 
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)
 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(val_loss)
 
        phase = "frozen" if not unfrozen else "full  "
        print(f"[{phase}] Epoch {epoch:02d}/{EPOCHS}  "
              f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f}  "
              f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "classes":    CLASSES,
                "mag_mean":   float(mag_mean),
                "mag_std":    float(mag_std),
                "median_mag": float(median_mag),
            }, best_model_path)
            print(f"  Best model saved (epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and unfrozen:
                print(f"\nEarly stopping after epoch {epoch}.")
                break
 
    # 6. Final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
 
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, DEVICE)
 
    print(f"\nFinal Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"\nPer-class Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASSES))
 
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
 
    # 7. Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 7: Real Steel Micrograph Classification", fontsize=13, fontweight="bold")
 
    ax = axes[0]
    ax.plot(train_losses, label="Train loss", color="steelblue")
    ax.plot(val_losses,   label="Val loss",   color="tomato")
    ax.axvline(x=FREEZE_EPOCHS - 0.5, color="gray", linestyle="--", label="Unfreeze")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Learning Curves")
    ax.legend()
 
    ax = axes[1]
    ax.plot(train_accs, label="Train acc", color="steelblue")
    ax.plot(val_accs,   label="Val acc",   color="tomato")
    ax.axvline(x=FREEZE_EPOCHS - 0.5, color="gray", linestyle="--", label="Unfreeze")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
 
    ax = axes[2]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (acc={test_acc:.3f})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontweight="bold")
    plt.colorbar(im, ax=ax)
 
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase7_results.png")
    plt.savefig(out, dpi=130)
    plt.show()
   
 
 
if __name__ == "__main__":
    main()
