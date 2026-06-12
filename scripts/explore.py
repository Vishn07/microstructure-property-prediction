import h5py

file_path = "C:\\Users\\vishw\\ml tut\\Project\\data\\MICRO2D_homogenized.h5"  # adjust if your filename is different

def print_structure(name, obj):
    print(name, "→", type(obj).__name__, end="")
    if hasattr(obj, "shape"):
        print(f"  shape: {obj.shape}  dtype: {obj.dtype}", end="")
    print()

with h5py.File(file_path, "r") as f:
    print("=== File structure ===")
    f.visititems(print_structure)