"""
Phase 7c: Multi-Seed Verification of Realistic Augmentation
============================================================
Phase 7b showed realistic augmentation beat generic by +7.8 points on a single
run. But the test set is only 90 images, so 7.8 points is about 7 images.
That could be luck.

This script runs BOTH augmentation strategies across N different random seeds
and reports mean, standard deviation, and per-seed results. If realistic wins
consistently across seeds, the effect is real. If the seeds overlap heavily,
the single-run result was noise.

Each seed changes BOTH the train/val/test split AND model initialization,
so it is a genuine independent replication.

RUNTIME WARNING: trains 2 models x N_SEEDS times. With N_SEEDS=5 and
EPOCHS=100 this is a long run. Set N_SEEDS=3 for a faster check.
Recommended to run overnight.
"""

import os
import json
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

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGE_DIR  = r"C:\Users\vishw\.cache\kagglehub\datasets\safi842\highcarbon-micrographs\versions\2\For Training\Cropped"
METADATA   = r"C:\Users\vishw\.cache\kagglehub\datasets\safi842\highcarbon-micrographs\versions\2\new_metadata.xlsx"
OUTPUT_DIR = r"C:\Users\vishw\ml tut\Project\outputs"

SEEDS         = [42, 123, 777, 2024, 31337]   # set to [42, 123, 777] for a faster run
BATCH_SIZE    = 16
EPOCHS        = 100
LR_HEAD       = 1e-3
LR_FULL       = 1e-5
FREEZE_EPOCHS = 8
PATIENCE      = 15
IMG_SIZE      = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Seeds to run: {SEEDS}")
print(f"Total models to train: {len(SEEDS) * 2}\n")

CLASS_MAP = {
    "spheroidite": "spheroidite",
    "spheroidite+widmanstatten": "spheroidite",
    "network": "network",
    "pearlite": "pearlite",
    "pearlite+spheroidite": "pearlite",
    "pearlite+widmanstatten": "pearlite",
}
CLASSES   = ["spheroidite", "network", "pearlite"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}


# ── Materials-realistic augmentations ─────────────────────────────────────────
def aug_etching_variation(img, rng):
    """Etch strength varies -> nonlinear contrast change between phases."""
    gamma = rng.uniform(0.7, 1.4)
    table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def aug_illumination_gradient(img, rng):
    """Uneven microscope illumination across the field of view."""
    h, w = img.shape
    strength = rng.uniform(0.10, 0.30)
    angle    = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx - w / 2) / (w / 2)
    yy = (yy - h / 2) / (h / 2)
    field = 1.0 + strength * (np.cos(angle) * xx + np.sin(angle) * yy)
    return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)


def aug_vignette(img, rng):
    """Radial edge darkening, common in optical microscopy."""
    h, w = img.shape
    strength = rng.uniform(0.10, 0.30)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - w/2)**2 + (yy - h/2)**2)
    field = 1.0 - strength * (r / r.max()) ** 2
    return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)


def aug_focus_variation(img, rng):
    """Slight defocus. Real micrographs vary in sharpness."""
    sigma = rng.uniform(0.0, 1.2)
    if sigma < 0.1:
        return img
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def aug_sensor_noise(img, rng):
    """Camera sensor grain."""
    sigma = rng.uniform(0, 8)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape),
                   0, 255).astype(np.uint8)


def aug_scale_jitter(img, rng):
    """Small magnification differences between imaging sessions."""
    scale = rng.uniform(1.0, 1.25)
    h, w = img.shape
    nh, nw = int(h * scale), int(w * scale)
    big = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0 = int(rng.integers(0, max(1, nh - h + 1)))
    x0 = int(rng.integers(0, max(1, nw - w + 1)))
    return big[y0:y0+h, x0:x0+w]


def apply_realistic_augmentation(img, rng):
    if rng.random() < 0.5: img = aug_scale_jitter(img, rng)
    if rng.random() < 0.6: img = aug_etching_variation(img, rng)
    if rng.random() < 0.5: img = aug_illumination_gradient(img, rng)
    if rng.random() < 0.3: img = aug_vignette(img, rng)
    if rng.random() < 0.4: img = aug_focus_variation(img, rng)
    if rng.random() < 0.4: img = aug_sensor_noise(img, rng)
    return img


# ── Dataset ───────────────────────────────────────────────────────────────────
class UHCSDataset(Dataset):
    def __init__(self, records, augment_mode="none", seed=42):
        self.records      = records
        self.augment_mode = augment_mode
        self.rng          = np.random.default_rng(seed)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.generic = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        ])
        self.center = transforms.CenterCrop(IMG_SIZE)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((256, 256), dtype=np.uint8)
        img = cv2.resize(img, (256, 256))

        if self.augment_mode == "realistic":
            img = apply_realistic_augmentation(img, self.rng)
            if self.rng.random() < 0.5: img = np.fliplr(img).copy()
            if self.rng.random() < 0.5: img = np.flipud(img).copy()
            img = np.rot90(img, int(self.rng.integers(0, 4))).copy()

        img3 = np.stack([img, img, img], axis=2)
        t = torch.from_numpy(img3).permute(2, 0, 1).float() / 255.0
        t = self.generic(t) if self.augment_mode == "generic" else self.center(t)
        t = self.normalize(t)
        return (t,
                torch.tensor([rec["mag_norm"]], dtype=torch.float32),
                torch.tensor(rec["label"], dtype=torch.long))


# ── Model ─────────────────────────────────────────────────────────────────────
class MicrographClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone  = nn.Sequential(*list(base.children())[:-1])
        self.mag_embed = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(512 + 32, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes))

    def forward(self, img, mag):
        v = self.backbone(img).view(-1, 512)
        return self.head(torch.cat([v, self.mag_embed(mag)], dim=1))

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    preds_all, labels_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, mags, labels in loader:
            imgs, mags, labels = imgs.to(device), mags.to(device), labels.to(device)
            if train: optimizer.zero_grad()
            logits = model(imgs, mags)
            loss = criterion(logits, labels)
            if train:
                loss.backward(); optimizer.step()
            total_loss += loss.item() * len(imgs)
            p = logits.argmax(1)
            correct += (p == labels).sum().item(); total += len(imgs)
            preds_all.extend(p.cpu().numpy()); labels_all.extend(labels.cpu().numpy())
    return total_loss/total, correct/total, np.array(preds_all), np.array(labels_all)


def train_one(train_rec, val_rec, test_rec, augment_mode, class_weights, seed, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = UHCSDataset(train_rec, augment_mode, seed=seed)
    val_ds   = UHCSDataset(val_rec,   "none",       seed=seed)
    test_ds  = UHCSDataset(test_rec,  "none",       seed=seed)

    tl     = [r["label"] for r in train_rec]
    counts = np.bincount(tl, minlength=3)
    inv    = 1.0 / np.maximum(counts, 1)
    sampler = WeightedRandomSampler([inv[r["label"]] for r in train_rec], len(train_rec))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    model = MicrographClassifier().to(DEVICE)
    model.freeze_backbone()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4)

    best_val, patience_ct, unfrozen, best_state = float("inf"), 0, False, None
    final_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        final_epoch = epoch
        if epoch == FREEZE_EPOCHS + 1 and not unfrozen:
            model.unfreeze_backbone()
            for g in optimizer.param_groups: g["lr"] = LR_FULL
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=4)
            best_val, patience_ct, unfrozen = float("inf"), 0, True

        _, tr_acc, _, _ = run_epoch(model, train_loader, criterion, DEVICE, optimizer)
        vl_loss, vl_acc, _, _ = run_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step(vl_loss)

        if epoch % 20 == 0:
            print(f"    [{tag}] epoch {epoch:03d}  train {tr_acc:.3f}  val {vl_acc:.3f}")

        if vl_loss < best_val:
            best_val, patience_ct = vl_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE and unfrozen:
                break

    model.load_state_dict(best_state)
    _, test_acc, preds, labels = run_epoch(model, test_loader, criterion, DEVICE)
    return test_acc, preds, labels, final_epoch


def build_records():
    df = pd.read_excel(METADATA)
    df["class"] = df["primary_microconstituent"].map(CLASS_MAP)
    df = df.dropna(subset=["class"])

    def parse_mag(m):
        if pd.isna(m): return np.nan
        try: return float(str(m).upper().replace("X", "").strip())
        except: return np.nan

    df["mag_raw"] = df["magnification"].apply(parse_mag)
    df["mag_raw"] = df["mag_raw"].fillna(np.nanmedian(df["mag_raw"].values))
    lg = np.log1p(df["mag_raw"])
    df["mag_norm"] = (lg - lg.mean()) / (lg.std() + 1e-8)

    records = []
    for _, row in df.iterrows():
        num  = row["path"].replace("micrograph", "").replace(".png", "")
        base = f"Croppedmicrograph{num}"
        p = None
        for cand in [os.path.join(IMAGE_DIR, base + ".png"),
                     os.path.join(IMAGE_DIR, base + ".jpg"),
                     os.path.join(IMAGE_DIR, base)]:
            if os.path.exists(cand):
                p = cand; break
        if p is None: continue
        records.append({"path": p, "mag_norm": row["mag_norm"],
                        "label": CLASS2IDX[row["class"]]})
    return records


def main():
    records = build_records()
    print(f"Images found: {len(records)}\n")

    results = {"generic": [], "realistic": []}
    per_seed_detail = []

    for si, seed in enumerate(SEEDS, 1):
        print(f"{'='*62}")
        print(f"SEED {seed}  ({si}/{len(SEEDS)})")
        print(f"{'='*62}")

        # Seed changes the split too, so this is a true independent replication
        labels_arr = [r["label"] for r in records]
        train_rec, test_rec = train_test_split(
            records, test_size=0.15, random_state=seed, stratify=labels_arr)
        tl = [r["label"] for r in train_rec]
        train_rec, val_rec = train_test_split(
            train_rec, test_size=0.15, random_state=seed, stratify=tl)

        counts = np.bincount([r["label"] for r in train_rec], minlength=3)
        cw = 1.0 / np.maximum(counts, 1)
        cw = torch.tensor(cw / cw.sum() * 3, dtype=torch.float32).to(DEVICE)

        row = {"seed": seed}
        for mode in ["generic", "realistic"]:
            acc, preds, labels, ep = train_one(
                train_rec, val_rec, test_rec, mode, cw, seed, mode[:4])
            results[mode].append(acc)
            row[mode] = acc
            row[f"{mode}_epochs"] = ep
            print(f"  {mode:<10} test acc: {acc*100:.1f}%  (stopped epoch {ep})")

        row["diff"] = row["realistic"] - row["generic"]
        per_seed_detail.append(row)
        print(f"  difference: {row['diff']*100:+.1f} points\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    gen  = np.array(results["generic"])
    real = np.array(results["realistic"])
    diffs = real - gen

    print(f"{'='*62}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*62}")
    print(f"{'Seed':<10}{'Generic':<12}{'Realistic':<12}{'Diff':<10}")
    print("-" * 44)
    for r in per_seed_detail:
        print(f"{r['seed']:<10}{r['generic']*100:<12.1f}{r['realistic']*100:<12.1f}"
              f"{r['diff']*100:+.1f}")
    print("-" * 44)
    print(f"{'MEAN':<10}{gen.mean()*100:<12.1f}{real.mean()*100:<12.1f}"
          f"{diffs.mean()*100:+.1f}")
    print(f"{'STD':<10}{gen.std()*100:<12.1f}{real.std()*100:<12.1f}"
          f"{diffs.std()*100:.1f}")
    print()
    print(f"Generic:   {gen.mean()*100:.1f}% +/- {gen.std()*100:.1f}%")
    print(f"Realistic: {real.mean()*100:.1f}% +/- {real.std()*100:.1f}%")
    print(f"Mean improvement: {diffs.mean()*100:+.1f} points")
    print(f"Realistic won in {(diffs > 0).sum()} of {len(diffs)} seeds")

    # Simple significance check (paired)
    if len(diffs) >= 3 and diffs.std() > 0:
        t_stat = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))
        print(f"Paired t-statistic: {t_stat:.2f}")
        if abs(t_stat) > 2.78 and len(diffs) >= 5:
            print("  -> Significant at p<0.05 (5 seeds, two-tailed)")
        elif abs(t_stat) > 2.0:
            print("  -> Suggestive but not conclusive; more seeds would help")
        else:
            print("  -> Not statistically distinguishable from noise")

    # Save
    out_csv = os.path.join(OUTPUT_DIR, "phase7c_multiseed.csv")
    pd.DataFrame(per_seed_detail).to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 7c: Multi-Seed Verification of Realistic Augmentation",
                 fontsize=12, fontweight="bold")

    # Per-seed paired lines
    ax = axes[0]
    for r in per_seed_detail:
        ax.plot([0, 1], [r["generic"]*100, r["realistic"]*100],
                marker="o", color="#27ae60" if r["diff"] > 0 else "#c0392b",
                alpha=0.7, linewidth=1.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Generic", "Realistic"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Paired results per seed")
    ax.grid(axis="y", alpha=0.3)

    # Mean with error bars
    ax = axes[1]
    ax.bar([0, 1], [gen.mean()*100, real.mean()*100],
           yerr=[gen.std()*100, real.std()*100], capsize=8,
           color=["#888888", "#27ae60"], width=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Generic", "Realistic"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Mean +/- SD across {len(SEEDS)} seeds")
    ax.set_ylim(0, 100)
    for i, (m, s) in enumerate([(gen.mean()*100, gen.std()*100),
                                 (real.mean()*100, real.std()*100)]):
        ax.text(i, m + s + 2, f"{m:.1f}%", ha="center", fontweight="bold")

    # Difference distribution
    ax = axes[2]
    ax.bar(range(len(diffs)), diffs*100,
           color=["#27ae60" if d > 0 else "#c0392b" for d in diffs])
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(diffs.mean()*100, color="blue", linestyle="--",
               label=f"mean {diffs.mean()*100:+.1f}")
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([str(s) for s in SEEDS], rotation=30, fontsize=8)
    ax.set_xlabel("Seed"); ax.set_ylabel("Improvement (points)")
    ax.set_title("Per-seed improvement")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase7c_multiseed.png")
    plt.savefig(out, dpi=130)
    plt.show()
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
