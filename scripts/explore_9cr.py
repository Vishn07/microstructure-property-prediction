
import os

# CHANGE THIS to wherever you extracted the download
DATA_ROOT = r"C:\Users\vishw\ml tut\Project\data\cr9 data"

def explore(root):
    if not os.path.exists(root):
        print(f"ERROR: Path does not exist: {root}")
        return

    print(f"Exploring: {root}\n")

    # Walk the directory tree
    total_files = 0
    extensions = {}
    image_files = []
    data_files  = []

    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, "").count(os.sep)
        indent = "  " * depth
        folder_name = os.path.basename(dirpath) or dirpath
        print(f"{indent}{folder_name}/  ({len(filenames)} files)")

        # Show first few files in each folder
        for fname in filenames[:3]:
            print(f"{indent}  {fname}")
        if len(filenames) > 3:
            print(f"{indent}  ... and {len(filenames)-3} more")

        for fname in filenames:
            total_files += 1
            ext = os.path.splitext(fname)[1].lower()
            extensions[ext] = extensions.get(ext, 0) + 1

            if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
                image_files.append(os.path.join(dirpath, fname))
            if ext in [".csv", ".xlsx", ".xls", ".txt", ".json"]:
                data_files.append(os.path.join(dirpath, fname))

    print("=" * 70)
    print(f"\nTotal files: {total_files}")
    print(f"\nFile types found:")
    for ext, count in sorted(extensions.items(), key=lambda x: -x[1]):
        print(f"  {ext or '(no extension)'}: {count}")

    print(f"\nImage files: {len(image_files)}")
    print(f"Data/label files: {len(data_files)}")

    if data_files:
        print(f"\nLikely label files (open these to find YS/UTS):")
        for d in data_files:
            print(f"  {d}")

    # If there's a CSV or Excel, peek at it
    for d in data_files:
        if d.endswith((".csv", ".xlsx", ".xls")):
            print(f"Peeking at: {d}")
            try:
                import pandas as pd
                if d.endswith(".csv"):
                    df = pd.read_csv(d)
                else:
                    df = pd.read_excel(d)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(df.head().to_string())
            except Exception as e:
                print(f"Could not read: {e}")
            break  # just the first one

if __name__ == "__main__":
    explore(DATA_ROOT)
