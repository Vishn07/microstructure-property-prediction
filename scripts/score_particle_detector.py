"""
score_particle_detector.py
==========================

Scores the particle detector against your hand annotations.

Run after you have painted crops from make_annotation_crops.py:

    python score_particle_detector.py

For every crop that has a <id>_annotated.png it runs the detector on the
FULL source image (so the background estimate is the same as in a real run),
cuts out the matching window and compares it with your painted particles.

Reported per crop and overall:
    annotated_fraction / detected_fraction   carbide area fraction
    fraction_ratio                           detected / annotated (1.0 is perfect)
    iou, precision, recall                   pixel level overlap
    particle_recall                          share of your particles the detector found
    n_annotated / n_detected                 particle counts

Writes score_results.csv and a side by side picture for each crop in
annotation_kit/scored/ so you can see where it goes wrong.
"""
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

KIT = "annotation_kit"
import spheroidite_particle_analysis as pa


def read_painted(path, size):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)           # BGR
    if im is None:
        return None
    b, g, r = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    red = ((r > 150) & (g < 110) & (b < 110)).astype(np.uint8)
    small = cv2.resize(red.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return (small > 0.5).astype(np.uint8)


def components(mask, min_area=4):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area], lab


def main():
    kit = Path(KIT)
    idx = pd.read_csv(kit / "index.csv")
    (kit / "scored").mkdir(exist_ok=True)
    rows = []
    for _, r in idx.iterrows():
        ann_path = kit / f"{r.crop_id}_annotated.png"
        if not ann_path.exists():
            continue
        size = int(r["size"])
        gt = read_painted(ann_path, size)
        if gt is None:
            continue
        img = cv2.imread(r.source_file, cv2.IMREAD_GRAYSCALE)
        full_mask, valid = pa.detect_mask(img)
        x0, y0 = int(r.x0), int(r.y0)
        det = full_mask[y0:y0 + size, x0:x0 + size]
        tp = int((det & gt).sum()); fp = int((det & (1 - gt)).sum()); fn = int(((1 - det) & gt).sum())
        iou = tp / max(1, tp + fp + fn)
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        comps, lab = components(gt)
        hit = sum(1 for i, a in comps if (det[lab == i] > 0).mean() >= 0.3)
        dcomps, _ = components(det)
        rows.append(dict(crop_id=r.crop_id, magnification=r.magnification,
                         annotated_fraction=gt.mean(), detected_fraction=det.mean(),
                         fraction_ratio=det.mean() / max(1e-9, gt.mean()),
                         iou=iou, precision=prec, recall=rec,
                         particle_recall=hit / max(1, len(comps)),
                         n_annotated=len(comps), n_detected=len(dcomps)))
        base = cv2.cvtColor(img[y0:y0 + size, x0:x0 + size], cv2.COLOR_GRAY2BGR)
        a = base.copy(); a[gt > 0] = (0, 0, 255)
        d = base.copy(); d[det > 0] = (0, 200, 255)
        tile = np.hstack([base, a, d])
        cv2.imwrite(str(kit / "scored" / f"{r.crop_id}_compare.png"),
                    cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    if not rows:
        print("No <id>_annotated.png files found in", kit)
        return
    df = pd.DataFrame(rows)
    df.to_csv("score_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\nOVERALL (mean over crops)")
    print(df[["annotated_fraction", "detected_fraction", "fraction_ratio", "iou",
              "precision", "recall", "particle_recall"]].mean().round(3).to_string())
    print("\nLeft to right in each picture: original, your annotation (red), detector (orange).")


if __name__ == "__main__":
    main()
