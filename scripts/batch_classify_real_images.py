import re
import sys
import importlib.util
import traceback
from pathlib import Path
from dataclasses import replace, asdict
 
import pandas as pd
 
HERE = Path(__file__).resolve().parent
 
 
# ============================================================
# 1. SOURCES  ---  edit this for your machine
# ============================================================
# Each source needs:
#   name           short label that goes in the "source" column
#   image_folder   folder containing the actual image files
#   label_lookup   how to find each image's microconstituent text:
#                    "uhcs_metadata"  -> new_metadata.xlsx columns
#                                        'path' and 'primary_microconstituent'
#                    "catalog_csv"    -> any CSV with a filename column and
#                                        a free-text caption/description column
#                    "fixed"          -> every image in this folder goes to
#                                        one classifier, set via fixed_route
#   fixed_route    "pearlite" or "weld", only used when label_lookup="fixed"
 
SOURCES = [
    dict(
        name="uhcs",
        image_folder=r"C:\Users\vishw\.cache\kagglehub\datasets\safi842\highcarbon-micrographs\versions\2\For Training\Cropped",
        label_lookup="uhcs_metadata",
        metadata_path=r"C:\Users\vishw\ml tut\Project\data\new_metadata.xlsx",
        # UHCS "widmanstatten" means cementite/ferrite plates in high-carbon
        # steel, not weld ferrite. The weld classifier gave ~99% acicular on
        # every one of these images, so UHCS only goes to the pearlite one.
        allowed_classifiers=("pearlite",),
        # UHCS images are SEM. In SEM the cementite is BRIGHT, but the
        # pearlite classifier treats DARK pixels as carbide (it was written
        # for etched optical micrographs). Inverting the grey levels before
        # classification puts the carbide on the side the classifier expects.
        invert_for_pearlite=True,
        # Only 48 of 598 UHCS images contain "pearlite" in their label. The
        # pearlite classifier asks "lamellar or spheroidized?", which only
        # makes sense on dense carbide texture, so pure spheroidite images
        # skip it.
        pearlite_only_if_label_contains=("pearlite",),
        # Particle analysis (spheroidite_particle_analysis.py, same folder as
        # this script) runs on every image of this source.
        run_particles=True,
    ),
    dict(
        name="asm9",
        image_folder=r"C:\path\to\asm9_crops",  # <-- EDIT once you extract these
        label_lookup="catalog_csv",
        catalog_path=r"C:\path\to\asm_micrograph_catalog.csv",
        filename_col="filename",   # <-- edit to match your catalog's real column names
        caption_col="description",
    ),
    dict(
        name="atlas7",
        image_folder=r"C:\path\to\atlas7_crops",  # <-- EDIT once you extract these
        label_lookup="catalog_csv",
        catalog_path=r"C:\path\to\atlas7_micrograph_catalog.csv",
        filename_col="filename",
        caption_col="description",
    ),
    # Evans book is mostly full-page tables/graphs, not croppable
    # micrographs (see evans_microstructure_inventory.md), so it is left
    # out here. Add an entry the same way if you extract usable crops
    # from Chapter 17.
]
 
 
# ------------------------------------------------------------
# Test controls
# ------------------------------------------------------------
# RUN_TAG        results go to batch_real_image_results_<RUN_TAG>, so a test
#                run never overwrites an earlier run. Use "" for the plain
#                batch_real_image_results folder.
# ONLY_FILENAMES run just these images (names without extension are fine).
#                Set to None to process everything.
RUN_TAG = "v2"
ONLY_FILENAMES = None   # or a set of names, e.g. {"Croppedmicrograph209", "Croppedmicrograph90"}
 
 
# ============================================================
# 2. CALIBRATION  ---  fill in what you know, leave the rest out
# ============================================================
# Map a magnification string EXACTLY as it appears in your metadata
# (e.g. "1964X") to the real micron value printed on that image's scale
# bar. Leave a magnification out and this script uses each classifier's
# own default calibration and flags the row as uncalibrated.
#
# How to fill this in: for one image per magnification level, read the
# number printed next to its scale bar (e.g. "5 um", "20 um") and put it
# here. You do not need every magnification measured independently,
# since it is the same microscope: if you know 1964X = 5 um, you can
# derive another magnification's bar length by simple proportion.
KNOWN_SCALE_BAR_UM = {
    # "1964X": 5.0,
    # "982X": 10.0,
}
 
 
# Micrometres per pixel by magnification string, for the particle analysis.
# Leave empty and the particle columns are reported in pixels only, with the
# micron columns left blank.
UM_PER_PX_BY_MAG = {
    # "1964X": 0.0509,
}
 
 
def um_per_px_for(mag_text):
    if mag_text is None:
        return None
    return UM_PER_PX_BY_MAG.get(str(mag_text).strip().upper().replace(" ", ""))
 
 
# ============================================================
# 3. Load the two classifier scripts as importable modules
# ============================================================
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
 
 
# The script looks for the professor's folders (01_pearlite and
# 02_acicularferrite) in these places, in order, and uses the first match.
# If yours live somewhere else, put that folder first in the list.
_CANDIDATES = [
    HERE / "GenerativeAI",
    HERE,
    HERE.parent / "scripts" / "GenerativeAI",
    Path(r"C:\Users\vishw\ml tut\Project\scripts\GenerativeAI"),
    Path(r"C:\Users\vishw\Downloads\GenerativeAI"),
]
GENERATIVEAI_DIR = next(
    (c for c in _CANDIDATES
     if (c / "01_pearlite" / "optical_phase_analysis_v5.py").exists()),
    _CANDIDATES[0],
)
 
PEARLITE_PATH = GENERATIVEAI_DIR / "01_pearlite" / "optical_phase_analysis_v5.py"
WELD_PATH = GENERATIVEAI_DIR / "02_acicularferrite" / "weld_microstructure_classifier_v6_3.py"
 
for _p in (PEARLITE_PATH, WELD_PATH):
    if not _p.exists():
        raise FileNotFoundError(
            f"Can't find {_p}\n"
            "Looked in:\n  " + "\n  ".join(str(c) for c in _CANDIDATES) + "\n"
            "Add the folder that holds 01_pearlite/ and 02_acicularferrite/ "
            "to _CANDIDATES in this script."
        )
 
pearlite_mod = _load_module("pearlite_v5", PEARLITE_PATH)
weld_mod = _load_module("weld_v6_3", WELD_PATH)
 
PARTICLE_PATH = HERE / "spheroidite_particle_analysis.py"
particle_mod = _load_module("particles", PARTICLE_PATH) if PARTICLE_PATH.exists() else None
if particle_mod is None:
    print("[note] spheroidite_particle_analysis.py not found next to this script, "
          "particle analysis will be skipped.")
 
 
# ============================================================
# 4. Routing
# ============================================================
PEARLITE_KEYWORDS = ["pearlite", "spheroidite", "spheroidize", "ferrite"]
WELD_KEYWORDS = [
    "bainite", "widmanstatten", "widmanst\u00e4tten", "acicular",
    "martensite", "allotriomorph", "m-a-c", "m a c", "m/a",
]
 
 
def route(label_text):
    text = (label_text or "").lower()
    is_p = any(k in text for k in PEARLITE_KEYWORDS)
    is_w = any(k in text for k in WELD_KEYWORDS)
    if is_p and not is_w:
        return "pearlite"
    if is_w and not is_p:
        return "weld"
    if is_p and is_w:
        return "both"      # mixed label, e.g. "spheroidite+widmanstatten"
    return "unknown"
 
 
def restrict_route(dest, allowed):
    """
    Keep only the classifiers this source is allowed to use.
    'both' becomes whichever allowed one remains. A route to a classifier
    that is not allowed becomes 'not_applicable' (skipped, noted in the CSV).
    """
    if dest == "both":
        keep = [c for c in ("pearlite", "weld") if c in allowed]
        return keep[0] if len(keep) == 1 else ("both" if keep else "not_applicable")
    if dest in ("pearlite", "weld") and dest not in allowed:
        return "not_applicable"
    return dest
 
 
def scale_bar_um_for(mag_text):
    if mag_text is None:
        return None, False
    key = str(mag_text).strip().upper().replace(" ", "")
    if key in KNOWN_SCALE_BAR_UM:
        return KNOWN_SCALE_BAR_UM[key], True
    return None, False
 
 
# ============================================================
# 5. Per-image runners (call the classifiers' own functions directly,
#    do not shell out, so results come back as plain dicts to merge)
# ============================================================
def _inverted_copy(path, tmp_dir):
    """Write a grey-level-inverted copy of an image and return its path.
    The copy keeps the original file name so output folders stay matched."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    out = Path(tmp_dir) / (Path(path).stem + ".png")
    cv2.imwrite(str(out), 255 - img)
    return out
 
 
def run_pearlite(path, out_root, invert=False):
    out_root.mkdir(parents=True, exist_ok=True)
    if invert:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            return dict(pearlite_mod.process_image(_inverted_copy(path, tmp), out_root))
    return dict(pearlite_mod.process_image(Path(path), out_root))
 
 
def run_weld(path, out_root, scale_bar_um=None):
    s = weld_mod.Settings()
    if scale_bar_um is not None:
        s = replace(s, scale_bar_um=scale_bar_um)
    out_root.mkdir(parents=True, exist_ok=True)
    return dict(weld_mod.process(Path(path), out_root / Path(path).stem, s))
 
 
# ============================================================
# 6. Build the (path, label_text, magnification) list per source
# ============================================================
def _resolve_uhcs_filename(folder, metadata_path_value):
    """
    new_metadata.xlsx lists paths like 'micrograph2.png', but the actual
    Kaggle download (safi842/highcarbon-micrographs, 'For Training/Cropped')
    saves the same images as 'Croppedmicrograph2.png' (sometimes with no
    extension, or .jpg). This is the exact candidate list Phase 7 had to
    work out by trial and error, kept here so you do not have to redo it.
    """
    num = metadata_path_value.replace("micrograph", "").replace(".png", "")
    base = f"Croppedmicrograph{num}"
    for candidate in (base + ".png", base + ".jpg", base):
        p = folder / candidate
        if p.exists():
            return p
    # nothing matched; return the .png guess so the caller's existence
    # check reports a clear "missing file" for this one instead of crashing
    return folder / (base + ".png")
 
 
def images_for_source(src):
    folder = Path(src["image_folder"])
    items = []
 
    if src["label_lookup"] == "uhcs_metadata":
        meta = pd.read_excel(src["metadata_path"])
        for _, r in meta.iterrows():
            items.append(dict(
                path=_resolve_uhcs_filename(folder, str(r["path"])),
                label_text=r.get("primary_microconstituent", ""),
                magnification=r.get("magnification", None),
            ))
 
    elif src["label_lookup"] == "catalog_csv":
        cat = pd.read_csv(src["catalog_path"])
        fname_col = src.get("filename_col", cat.columns[0])
        caption_col = src.get("caption_col", cat.columns[-1])
        for _, r in cat.iterrows():
            items.append(dict(
                path=folder / str(r[fname_col]),
                label_text=str(r.get(caption_col, "")),
                magnification=r.get("magnification", None),
            ))
 
    elif src["label_lookup"] == "fixed":
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in exts:
                items.append(dict(
                    path=p,
                    label_text=src.get("fixed_route", ""),
                    magnification=None,
                ))
    else:
        raise ValueError(f"Unknown label_lookup: {src['label_lookup']!r}")
 
    return items
 
 
# ============================================================
# 7. Main loop
# ============================================================
def main():
    out_root = HERE / ("batch_real_image_results" + (f"_{RUN_TAG}" if RUN_TAG else ""))
    out_root.mkdir(exist_ok=True)
    if ONLY_FILENAMES:
        print(f"*** TEST MODE: only {len(ONLY_FILENAMES)} named image(s) will run. "
              f"Set ONLY_FILENAMES = None for the full batch. ***")
 
    all_rows = []
 
    for src in SOURCES:
        folder = Path(src["image_folder"])
        if not folder.exists():
            print(f"[skip source] {src['name']}: folder not found -> {folder}")
            continue
 
        try:
            items = images_for_source(src)
        except Exception as e:
            print(f"[skip source] {src['name']}: could not read its metadata/catalog -> {e}")
            continue
 
        print(f"\n=== {src['name']}: {len(items)} image(s) listed ===")
 
        for item in items:
            path = item["path"]
            if ONLY_FILENAMES and path.stem not in ONLY_FILENAMES:
                continue
            if not path.exists():
                print(f"  [missing file] {path.name}")
                continue
 
            label_text = item.get("label_text") or src.get("fixed_route", "")
            dest = route(label_text) if src["label_lookup"] != "fixed" else src.get("fixed_route", "unknown")
            dest = restrict_route(dest, src.get("allowed_classifiers", ("pearlite", "weld")))
            bar_um, calibrated = scale_bar_um_for(item.get("magnification"))
 
            row = dict(
                source=src["name"],
                filename=path.name,
                microconstituent_label=label_text,
                magnification=item.get("magnification"),
                routed_to=dest,
                scale_calibrated=calibrated,
                pearlite_input_inverted=bool(src.get("invert_for_pearlite", False)) and dest in ("pearlite", "both"),
            )
 
            # pearlite classifier only where the label says pearlite is present
            req = src.get("pearlite_only_if_label_contains")
            run_pe = dest in ("pearlite", "both")
            if run_pe and req and not any(k in str(label_text).lower() for k in req):
                run_pe = False
            run_pa = bool(src.get("run_particles")) and particle_mod is not None
            row["pearlite_classifier_run"] = run_pe
            row["particles_run"] = run_pa
 
            try:
                if run_pa:
                    pa_row = particle_mod.analyze_image(
                        path, out_root / "particles" / src["name"],
                        um_per_px_for(item.get("magnification")))
                    pa_row.pop("filename", None)
                    row.update({f"particles__{k}": v for k, v in pa_row.items()})
 
                if run_pe:
                    p_row = run_pearlite(path, out_root / "pearlite" / src["name"],
                                         invert=src.get("invert_for_pearlite", False))
                    row.update({f"pearlite__{k}": v for k, v in p_row.items()})
 
                if dest in ("weld", "both"):
                    w_row = run_weld(path, out_root / "weld" / src["name"], bar_um)
                    row.update({f"weld__{k}": v for k, v in w_row.items()})
 
                if dest == "not_applicable":
                    row["note"] = "label routes to a classifier this source is not allowed to use, skipped"
 
                if dest == "unknown" and not run_pa:
                    row["note"] = "label matched neither classifier's keyword list, skipped"
 
                print(f"  [{str(dest):9s}] {path.name}")
 
            except Exception as e:
                row["error"] = str(e)
                print(f"  [ERROR] {path.name}: {e}")
                traceback.print_exc()
 
            all_rows.append(row)
 
    master = pd.DataFrame(all_rows)
    out_csv = out_root / "master_real_image_labels.csv"
    master.to_csv(out_csv, index=False)
 
    print("\n" + "=" * 72)
    print(f"Done. {len(master)} image(s) processed.")
    print(f"Master table: {out_csv}")
    if "scale_calibrated" in master.columns and len(master):
        n_uncal = (~master["scale_calibrated"]).sum()
        print(f"Uncalibrated (default scale used): {n_uncal} of {len(master)} row(s).")
        print("Fill in KNOWN_SCALE_BAR_UM at the top of this script to calibrate more.")
    print("=" * 72)
 
 
if __name__ == "__main__":
    main()
 