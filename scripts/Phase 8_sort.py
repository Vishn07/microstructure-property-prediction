"""
Phase 8 Step 3: Build an EXPLICIT, verified alloy-to-folder mapping
====================================================================
The fuzzy matcher made false positives (CPJ-7B matched to CPJ7 folder).
This script builds an explicit mapping with strict rules and shows you
exactly which alloys are safely usable.
"""

import os
import pandas as pd

DATA_ROOT = r"C:\Users\vishw\ml tut\Project\data\cr9 data"
CSV_PATH  = r"C:\Users\vishw\ml tut\Project\data\cr9 data\tensile-properties-from-jeffs-ml-sheet-20220208.csv"


def get_all_image_folders(root):
    folders = {}
    for dirpath, dirnames, filenames in os.walk(root):
        bmps = [f for f in filenames if f.lower().endswith(".bmp")]
        if bmps:
            folders[os.path.basename(dirpath)] = {
                "path": dirpath, "n_images": len(bmps)
            }
    return folders


def main():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Alloy"])
    df["Alloy"] = df["Alloy"].astype(str)

    # Room temp only
    room = df[(df["Temperature, [C]"] >= 20) & (df["Temperature, [C]"] <= 27)].copy()

    folders = get_all_image_folders(DATA_ROOT)
    folder_names = set(folders.keys())

    # ── Explicit strict matching ──────────────────────────────────────────
    # Rule 1: exact match after removing dashes
    # Rule 2: HR##-T1 in CSV maps to HR##-* folder (B, b, BB, B2 suffix)
    # Rule 3: CPJ-7 variants (7B,7C..) only match if EXACT variant folder exists
    # Rule 4: P92STD maps to P92 folder

    def find_folder(alloy):
        a = alloy.upper().replace("-", "").replace(" ", "")

        # Exact normalized match
        for fname in folder_names:
            if fname.upper().replace("-", "").replace(" ", "") == a:
                return fname

        # HR family: HR52T1 -> look for folder starting HR52
        if a.startswith("HR"):
            hr_num = a.replace("HR", "").replace("T1", "")
            for fname in folder_names:
                fn = fname.upper().replace("-", "").replace(" ", "")
                if fn.startswith("HR" + hr_num):
                    # make sure the number matches exactly (HR52 not HR520)
                    rest = fn[len("HR" + hr_num):]
                    if rest == "" or not rest[0].isdigit():
                        return fname

        # P92STD -> P92
        if a == "P92STD" and "P92" in folder_names:
            return "P92"

        return None

    print("=" * 65)
    print("EXPLICIT ROOM-TEMPERATURE MAPPING")
    print("=" * 65)

    usable = []
    skipped = []

    for _, row in room.iterrows():
        alloy = row["Alloy"]
        folder = find_folder(alloy)
        if folder:
            usable.append({
                "alloy":  alloy,
                "folder": folder,
                "path":   folders[folder]["path"],
                "n_img":  folders[folder]["n_images"],
                "YS":     row["Yield Stress, [MPa]"],
                "UTS":    row["Ultimate Tensile Stress, [MPa]"],
                "temp":   row["Temperature, [C]"],
            })
        else:
            skipped.append(alloy)

    # Deduplicate: if same alloy appears twice at room temp, average the props
    # (CPJ-7 and CPJ-4 had duplicate rows)
    seen = {}
    for u in usable:
        key = u["alloy"]
        if key in seen:
            # average the properties
            seen[key]["YS"]  = (seen[key]["YS"]  + u["YS"])  / 2
            seen[key]["UTS"] = (seen[key]["UTS"] + u["UTS"]) / 2
            seen[key]["_dup"] = True
        else:
            seen[key] = u

    final = list(seen.values())

    print(f"\nUsable alloys (have both images and room-temp YS/UTS): {len(final)}")
    print(f"{'Alloy':<12}{'Folder':<12}{'Images':<8}{'YS':<8}{'UTS':<8}")
    print("-" * 50)
    total_images = 0
    for u in sorted(final, key=lambda x: x["alloy"]):
        dup = " (avg of 2)" if u.get("_dup") else ""
        print(f"{u['alloy']:<12}{u['folder']:<12}{u['n_img']:<8}"
              f"{u['YS']:<8.0f}{u['UTS']:<8.0f}{dup}")
        total_images += u["n_img"]

    print("-" * 50)
    print(f"Total alloys: {len(final)}")
    print(f"Total images available: {total_images}")
    print(f"\nSkipped (no matching folder): {len(skipped)}")
    print(f"  {skipped}")

    # Save the mapping for the training script to use
    map_df = pd.DataFrame(final)
    out = os.path.join(os.path.dirname(CSV_PATH), "phase8_alloy_mapping.csv")
    map_df.to_csv(out, index=False)
    print(f"\nMapping saved to: {out}")
    print("Review the table above. If the YS/UTS values look right for each")
    print("alloy, we can proceed to build the training pipeline.")


if __name__ == "__main__":
    main()