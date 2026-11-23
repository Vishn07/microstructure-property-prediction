"""
collect_annotations.py

Finds the painted file for each crop in annotation_kit and saves a copy named
<crop id>_annotated.png so score_particle_detector.py can see it.

A file counts as painted if it holds at least MIN_RED_PIXELS pure red pixels.
It checks <id>_annotated.png, <id>_paint_4x.png and <id>_orig.png, and uses
the one with the most red. Nothing is deleted or overwritten except an
existing <id>_annotated.png that has less red than the file it is replacing.

Run from the scripts folder:
    python collect_annotations.py
"""
import csv
from pathlib import Path

import cv2
import numpy as np

KIT = Path("annotation_kit")
MIN_RED_PIXELS = 50


def red_count(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return 0
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    return int(((r > 150) & (g < 110) & (b < 110)).sum())


def main():
    index = KIT / "index.csv"
    if not index.exists():
        print("Cannot find", index, "- run this from the scripts folder.")
        return
    with open(index, newline="", encoding="utf-8") as f:
        ids = [row["crop_id"] for row in csv.DictReader(f)]

    found = 0
    for cid in ids:
        names = [f"{cid}_annotated.png", f"{cid}_paint_4x.png", f"{cid}_orig.png"]
        counts = {n: red_count(KIT / n) for n in names if (KIT / n).exists()}
        if not counts:
            continue
        best = max(counts, key=counts.get)
        if counts[best] < MIN_RED_PIXELS:
            print(f"not painted yet: {cid}")
            continue
        target = KIT / f"{cid}_annotated.png"
        if best != target.name:
            target.write_bytes((KIT / best).read_bytes())
            print(f"saved {target.name} (copied from {best}, {counts[best]} red px)")
        else:
            print(f"ok: {target.name} ({counts[best]} red px)")
        found += 1

    print(f"\n{found} painted crop(s) ready. Now run: python score_particle_detector.py")


if __name__ == "__main__":
    main()
