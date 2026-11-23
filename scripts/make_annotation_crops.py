"""
make_annotation_crops.py
========================

Cuts small crops out of your own UHCS images so you can paint the carbide
particles by hand. Those hand annotations are the ground truth the particle
detector is scored against (see score_particle_detector.py).

Run from the folder that holds spheroidite_particle_analysis.py:

    python make_annotation_crops.py

Edit the three paths in CONFIG if yours differ. It writes an
`annotation_kit` folder containing, for each crop:

    <id>_paint_4x.png   the crop enlarged 4x. PAINT ON A COPY OF THIS ONE.
    <id>_orig.png       the crop at original size (reference only)
and index.csv, which records where each crop came from.

HOW TO ANNOTATE (about 15 to 30 minutes per crop)
-------------------------------------------------
1. Open <id>_paint_4x.png in GIMP, Photopea, Paint.NET or ImageJ. Save a
   copy named <id>_annotated.png in the same folder.
2. With a HARD-EDGED pencil or brush, paint every carbide particle in pure
   red (255, 0, 0). Paint the WHOLE particle, interior included. A fast way
   is to outline a particle in red and use the bucket fill inside it.
3. Turn off anti-aliasing and soft brushes. Leave the matrix, scratches,
   grain boundaries and cracks unpainted.
4. Skip a particle if you cannot tell whether it is cementite.
5. Save as PNG, not JPEG.
"""
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import re

# ---------------------------- CONFIG ----------------------------
IMAGE_FOLDER = r"C:\Users\vishw\.cache\kagglehub\datasets\safi842\highcarbon-micrographs\versions\2\For Training\Cropped"
METADATA = r"C:\Users\vishw\ml tut\Project\data\new_metadata.xlsx"
OUT = "annotation_kit"
CROP = 250                       # crop size in original pixels
PICKS = {                        # magnification -> how many images
    "982X": 2, "1964X": 3, "4910X": 3,
}
LABELS = ("spheroidite", "pearlite+spheroidite")
SEED = 7
UM_PER_PX_NUMERATOR = 190.4      # micrometres per pixel = this / magnification
# ----------------------------------------------------------------

import spheroidite_particle_analysis as pa


def find_image(folder, name):
    stem = Path(name).stem
    for ext in (".png", ".jpg", ".jpeg", ""):
        for cand in (folder / f"Cropped{stem}{ext}", folder / f"{stem}{ext}"):
            if cand.exists():
                return cand
    return None


def main():
    rng = np.random.default_rng(SEED)
    folder = Path(IMAGE_FOLDER)
    meta = pd.read_excel(METADATA)
    meta["mag"] = meta["magnification"].astype(str).str.upper().str.replace(" ", "")
    meta = meta[meta.primary_microconstituent.isin(LABELS)]
    out = Path(OUT)
    out.mkdir(exist_ok=True)
    rows = []
    for mag, n in PICKS.items():
        pool = meta[meta.mag == mag]
        pool = pool.iloc[rng.permutation(len(pool))]
        got = 0
        for _, r in pool.iterrows():
            if got >= n:
                break
            src = find_image(folder, r["path"])
            if src is None:
                continue
            img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            valid = pa.valid_region(img)
            h, w = img.shape
            # pick a random window that lies entirely in the valid region
            win = None
            for _ in range(200):
                x0 = int(rng.integers(0, max(1, w - CROP)))
                y0 = int(rng.integers(0, max(1, h - CROP)))
                if valid[y0:y0 + CROP, x0:x0 + CROP].all():
                    win = (x0, y0)
                    break
            if win is None:
                continue
            x0, y0 = win
            crop = img[y0:y0 + CROP, x0:x0 + CROP]
            cid = f"{mag}_{Path(r['path']).stem}_x{x0}_y{y0}"
            cv2.imwrite(str(out / f"{cid}_orig.png"), crop)
            cv2.imwrite(str(out / f"{cid}_paint_4x.png"),
                        cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
            mnum = float(re.sub("X", "", mag))
            rows.append(dict(crop_id=cid, source_file=str(src), x0=x0, y0=y0, size=CROP,
                             magnification=mag, label=r["primary_microconstituent"],
                             um_per_px=UM_PER_PX_NUMERATOR / mnum))
            got += 1
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    print(f"Wrote {len(rows)} crops to {out.resolve()}")
    print("Now paint the particles red on a copy of each *_paint_4x.png and save it as <id>_annotated.png")


if __name__ == "__main__":
    main()
