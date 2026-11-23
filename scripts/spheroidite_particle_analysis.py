"""
spheroidite_particle_analysis.py
================================

Particle analysis for SEM micrographs of spheroidized / particle-in-matrix
steel (the bulk of UHCS). It answers a different question from the
professor's pearlite classifier. That tool asks "lamellar or spheroidized?"
This one asks "what do the carbide particles look like?" and reports numbers
that can feed a property model directly:

    area_fraction            carbide area / valid image area
    n_particles, density     count and count per 1000 px^2 (and per um^2 if
                             a scale is supplied)
    d_eq_*                   equivalent circular diameter (mean, median,
                             p10, p90, coefficient of variation)
    aspect_mean              major/minor axis of a fitted ellipse
    frac_elongated           share of particles with aspect ratio above 2
                             (separates rods and plates from spheroids)
    circularity_mean         4*pi*A/P^2, 1.0 is a perfect circle
    nn_mean                  mean nearest neighbour spacing between centres
    clark_evans_R            observed / expected nearest neighbour spacing for
                             a random pattern. Below 1 means clustered
                             (boundary decorated), near 1 random, above 1 even
    frac_area_in_large       share of carbide area held by particles larger
                             than LARGE_D_FACTOR times the median diameter
    contrast_p99             99th percentile of the particle signal, a quality
                             flag. Low values mean faint particles and the
                             row should be treated with caution

Why it does not need to know polarity
-------------------------------------
Particles are found as features brighter than a local median background.
That measures how much brighter a feature is than the surrounding material,
so it works whatever the overall brightness of the matrix is. It assumes cementite is the
BRIGHT feature, which is true for secondary electron images of etched steel.
Set BRIGHT_PARTICLES = False for optical images, where it is dark.

Scale
-----
Everything is reported in pixels. Pass um_per_px to also get micron columns.
Without it the micron columns are left empty (NaN), so you cannot mix up
calibrated and uncalibrated rows.

Usage
-----
    python spheroidite_particle_analysis.py  path\\to\\image.png  [more images]
    python spheroidite_particle_analysis.py  --folder path\\to\\folder

Writes <out>/particle_results.csv and one QC overlay per image. LOOK AT THE
OVERLAYS before trusting any number.
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
BRIGHT_PARTICLES = True       # True for SEM, False for etched optical
BACKGROUND_WINDOW_FRAC = 0.15  # median window for the local background, fraction of the short image side
SMOOTH_SIGMA = 1.0            # px, light denoise before the top-hat
THRESH_MAD_K = 4.0            # threshold = median + K * 1.4826 * MAD of the top-hat signal
THRESH_FLOOR = 8.0            # grey levels (0-255), never threshold below this
MIN_AREA_PX = 8               # smaller blobs are noise
MAX_HOLE_FILL_FRAC = 0.01     # enclosed holes up to this fraction of the valid area are filled
EDGE_AT_HALF_PEAK = True      # place each particle edge at half its own peak signal (removes blur bias)
SPLIT_TOUCHING = True         # watershed split of touching particles
SPLIT_MIN_DIST_FRAC = 0.5     # a split peak must be at least this fraction of the max distance
LARGE_D_FACTOR = 3.0          # "large" particle = this many times the median diameter
D_MIN_UM = 1.0                # "comparable" particle set: only particles at least this wide (um)
RESOLVED_PX_ACROSS = 4        # a D_MIN_UM particle must span at least this many pixels to count as resolved
ELONGATED_ASPECT = 2.0
BORDER_MARGIN = 2             # px trimmed around the black data bar region


# ----------------------------------------------------------------------
# Image loading and the valid region
# ----------------------------------------------------------------------
def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def valid_region(img):
    """True where pixels are real micrograph. Removes the black rectangle left
    where the data bar was cropped out. Only large, solid, near-black blocks
    are removed, so dark features inside the micrograph survive."""
    h, w = img.shape
    black = (img <= 2).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    valid = np.ones((h, w), dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # a data bar block is big and almost perfectly rectangular
        if area > 0.01 * h * w and area / float(bw * bh) > 0.95:
            valid[lab == i] = False
    if BORDER_MARGIN > 0 and not valid.all():
        k = 2 * BORDER_MARGIN + 1
        valid = ~cv2.dilate((~valid).astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    return valid


# ----------------------------------------------------------------------
# Particle segmentation
# ----------------------------------------------------------------------
def particle_signal(img, valid):
    """Signal in grey levels, large where something is brighter than the local
    matrix. The local matrix level is a large-window MEDIAN, which is robust in
    both directions: big bright particles (under half the window) do not raise
    it, and big dark patches do not pull it down and create false bright rims
    around them."""
    work = img.astype(np.float32)
    if not BRIGHT_PARTICLES:
        work = 255.0 - work
    work = work.copy()
    work[~valid] = np.median(work[valid])
    if SMOOTH_SIGMA > 0:
        work = cv2.GaussianBlur(work, (0, 0), SMOOTH_SIGMA)

    # median on a quarter-size copy, then scale back up (fast and smooth enough)
    scale = 4
    small = cv2.resize(np.clip(work, 0, 255).astype(np.uint8),
                       (max(8, img.shape[1] // scale), max(8, img.shape[0] // scale)),
                       interpolation=cv2.INTER_AREA)
    k = int(round(BACKGROUND_WINDOW_FRAC * min(img.shape) / scale))
    k = max(5, k | 1)                                   # odd, at least 5
    med = cv2.medianBlur(small, k).astype(np.float32)
    bg = cv2.resize(med, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (0, 0), 4)
    return work - bg, k * scale


def fill_enclosed_holes(mask, max_area):
    """Fill background regions that are fully enclosed by particles and no
    larger than max_area pixels. SEM edge brightening makes the rim of a big
    particle brighter than its flat top, so a rim-only outline must be filled."""
    inv = (1 - mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = mask.shape
    out = mask.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches and area <= max_area:
            out[lab == i] = 1
    return out


def segment(signal, valid):
    s = signal[valid]
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    thr = max(THRESH_FLOOR, med + THRESH_MAD_K * 1.4826 * mad)
    mask = ((signal > thr) & valid).astype(np.uint8)

    # close pinholes, drop specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # fill holes
    inv = (1 - mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches and area < 200:
            mask[lab == i] = 1
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA_PX
    mask = keep[lab].astype(np.uint8)

    if EDGE_AT_HALF_PEAK and mask.any():
        # A fixed low threshold swells every particle by its blur halo. Put each
        # particle's edge at half of its own peak signal instead.
        n, lab, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        peak = np.zeros(n, dtype=np.float32)
        np.maximum.at(peak, lab[mask > 0], signal[mask > 0])
        grown = cv2.dilate(lab.astype(np.float32),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.int32)
        local_thr = np.maximum(thr, 0.5 * peak[grown])
        refined = ((signal > local_thr) & (grown > 0) & valid).astype(np.uint8)
        n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
        keep2 = np.zeros(n2, dtype=bool)
        keep2[1:] = stats2[1:, cv2.CC_STAT_AREA] >= MIN_AREA_PX
        mask = keep2[lab2].astype(np.uint8)
    mask = fill_enclosed_holes(mask, max_area=int(MAX_HOLE_FILL_FRAC * valid.sum()))
    mask = (mask & valid).astype(np.uint8)
    return mask, thr


def split_touching(mask):
    """Watershed on the distance transform to separate touching particles."""
    if mask.sum() == 0:
        return mask, 0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peak_thr = SPLIT_MIN_DIST_FRAC * dist.max()
    # markers: connected blobs of high distance, smoothed so one particle gives one marker
    dist_s = cv2.GaussianBlur(dist, (0, 0), 1.5)
    local_max = (dist_s == cv2.dilate(dist_s, np.ones((7, 7), np.uint8))) & (dist_s > 2.0)
    n_m, markers = cv2.connectedComponents(local_max.astype(np.uint8))
    if n_m <= 2:
        return mask, 0
    markers = markers.astype(np.int32) + 1          # background is 1, markers 2..
    markers[mask == 0] = 1
    bgr = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(bgr, markers.copy())
    out = mask.copy()
    out[ws == -1] = 0                                # watershed lines become gaps
    # re-clean tiny fragments created by the lines
    n, lab, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA_PX
    out = keep[lab].astype(np.uint8)
    return out, int(n_m - 2)


# ----------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------
def measure(mask, valid):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rows = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX:
            continue
        per = cv2.arcLength(c, True)
        (cx, cy), _ = cv2.minEnclosingCircle(c)
        if len(c) >= 5:
            (ex, ey), (ma, mi), _ = cv2.fitEllipse(c)
            major, minor = max(ma, mi), max(min(ma, mi), 1e-6)
            aspect = major / minor
            mom = cv2.moments(c)
            cx, cy = (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]) if mom["m00"] else (cx, cy)
        else:
            aspect, cx, cy = 1.0, cx, cy
        rows.append(dict(
            area=area,
            d_eq=2.0 * np.sqrt(area / np.pi),
            aspect=aspect,
            circ=min(1.0, 4.0 * np.pi * area / (per * per)) if per > 0 else np.nan,
            x=cx, y=cy,
        ))
    return pd.DataFrame(rows)


def summarize(parts, valid, um_per_px):
    nan = float("nan")
    valid_px = int(valid.sum())
    out = dict(valid_area_px=valid_px, n_particles=int(len(parts)))
    keys = ["area_fraction", "density_per_1000px2", "d_eq_mean_px", "d_eq_median_px",
            "d_eq_p10_px", "d_eq_p90_px", "d_eq_cv", "aspect_mean", "frac_elongated",
            "circularity_mean", "nn_mean_px", "clark_evans_R", "frac_area_in_large"]
    if len(parts) == 0:
        out.update({k: nan for k in keys})
        out["area_fraction"] = 0.0
    else:
        d = parts["d_eq"].to_numpy()
        a = parts["area"].to_numpy()
        out["area_fraction"] = float(a.sum() / valid_px)
        out["density_per_1000px2"] = float(1000.0 * len(parts) / valid_px)
        out["d_eq_mean_px"] = float(d.mean())
        out["d_eq_median_px"] = float(np.median(d))
        out["d_eq_p10_px"] = float(np.percentile(d, 10))
        out["d_eq_p90_px"] = float(np.percentile(d, 90))
        out["d_eq_cv"] = float(d.std() / d.mean()) if d.mean() > 0 else nan
        out["aspect_mean"] = float(parts["aspect"].mean())
        out["frac_elongated"] = float((parts["aspect"] > ELONGATED_ASPECT).mean())
        out["circularity_mean"] = float(parts["circ"].mean())
        out["frac_area_in_large"] = float(a[d > LARGE_D_FACTOR * np.median(d)].sum() / a.sum())
        if len(parts) >= 3:
            pts = parts[["x", "y"]].to_numpy()
            dd, _ = cKDTree(pts).query(pts, k=2)
            nn = dd[:, 1]
            out["nn_mean_px"] = float(nn.mean())
            expected = 0.5 / np.sqrt(len(parts) / valid_px)   # Clark-Evans random expectation
            out["clark_evans_R"] = float(nn.mean() / expected)
        else:
            out["nn_mean_px"] = nan
            out["clark_evans_R"] = nan

    # micron versions, empty unless a scale was given
    if um_per_px:
        out["um_per_px"] = float(um_per_px)
        out["density_per_um2"] = (out["density_per_1000px2"] / 1000.0) / um_per_px ** 2 \
            if np.isfinite(out.get("density_per_1000px2", nan)) else nan
        for k in ("d_eq_mean", "d_eq_median", "d_eq_p10", "d_eq_p90", "nn_mean"):
            v = out.get(k + "_px", nan)
            out[k + "_um"] = v * um_per_px if np.isfinite(v) else nan
    else:
        out["um_per_px"] = nan
        out["density_per_um2"] = nan
        for k in ("d_eq_mean", "d_eq_median", "d_eq_p10", "d_eq_p90", "nn_mean"):
            out[k + "_um"] = nan
    return out


def detect_mask(img):
    """Run the detection pipeline on a grayscale array and return
    (particle_mask, valid_region_mask). Used by the scoring script."""
    valid = valid_region(img)
    signal, _ = particle_signal(img, valid)
    mask, _ = segment(signal, valid)
    if SPLIT_TOUCHING:
        mask, _ = split_touching(mask)
    return mask, valid


# ----------------------------------------------------------------------
# One image
# ----------------------------------------------------------------------
def analyze_image(path, out_dir=None, um_per_px=None, save_overlay=True):
    path = Path(path)
    img = load_gray(path)
    valid = valid_region(img)
    signal, radius = particle_signal(img, valid)
    mask, thr = segment(signal, valid)
    n_split = 0
    if SPLIT_TOUCHING:
        mask, n_split = split_touching(mask)
    parts = measure(mask, valid)
    row = summarize(parts, valid, um_per_px)

    # Size-comparable set. Without a scale these stay empty (NaN). With one, only
    # particles at least D_MIN_UM wide are counted, so images shot at different
    # magnifications are judged on the same physical size range.
    cmp_keys = ["n_particles", "area_fraction", "d_eq_median_um", "aspect_mean",
                "frac_elongated", "circularity_mean"]
    if um_per_px:
        keep = parts["d_eq"].to_numpy() * um_per_px >= D_MIN_UM if len(parts) else np.zeros(0, bool)
        srow = summarize(parts[keep] if len(parts) else parts, valid, um_per_px)
        for k in cmp_keys:
            row["cmp_" + k] = srow.get(k, float("nan"))
        row["scale_resolved"] = bool(um_per_px <= D_MIN_UM / RESOLVED_PX_ACROSS)
    else:
        for k in cmp_keys:
            row["cmp_" + k] = float("nan")
        row["scale_resolved"] = float("nan")
    row["filename"] = path.name
    row["contrast_p99"] = float(np.percentile(signal[valid], 99))
    row["threshold_used"] = float(thr)
    row["background_window_px"] = int(radius)
    # blobs this large are probably not particles (a big smooth band or a whole grain)
    row["max_particle_area_frac"] = float(parts["area"].max() / valid.sum()) if len(parts) else 0.0
    row["qc_flag"] = ""
    if row["contrast_p99"] < 20:
        row["qc_flag"] += "low_contrast "
    if row["area_fraction"] > 0.35:
        row["qc_flag"] += "very_high_fraction "
    if row["max_particle_area_frac"] > 0.05:
        row["qc_flag"] += "giant_blob "
    if row["n_particles"] < 5:
        row["qc_flag"] += "few_particles "
    if um_per_px and not row["scale_resolved"]:
        row["qc_flag"] += "unresolved_scale "
    row["qc_flag"] = row["qc_flag"].strip()

    if save_overlay and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        rgb[~valid] = (60, 60, 60)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cnts, -1, (0, 0, 255), 1)
        cv2.putText(rgb, f"{row['n_particles']} particles  f={row['area_fraction']*100:.1f}%  "
                         f"d50={row['d_eq_median_px']:.1f}px" if row["n_particles"] else "no particles",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        side = np.hstack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), rgb])
        cv2.imwrite(str(out_dir / f"{path.stem}_particles_qc.png"), side)
    return row


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="image files")
    ap.add_argument("--folder", help="analyze every image in this folder")
    ap.add_argument("--out", default="particle_results", help="output folder")
    ap.add_argument("--um-per-px", type=float, default=None)
    args = ap.parse_args()

    files = [Path(p) for p in args.images]
    if args.folder:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        files += sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in exts)
    if not files:
        ap.error("give at least one image or --folder")

    rows = []
    for f in files:
        try:
            r = analyze_image(f, args.out, args.um_per_px)
            print(f"{f.name}: {r['n_particles']} particles, fraction {r['area_fraction']*100:.1f}%, "
                  f"median d {r['d_eq_median_px']:.1f} px  {r['qc_flag']}")
            rows.append(r)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "particle_results.csv", index=False)
    print(f"\nWrote {out / 'particle_results.csv'}")


if __name__ == "__main__":
    main()