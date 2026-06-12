import h5py
import numpy as np
import cv2
import os

FILE_PATH  = "C:\\Users\\vishw\\ml tut\\Project\\data\\MICRO2D_homogenized.h5"
OUTPUT_DIR = "C:\\Users\\vishw\\ml tut\\Project\\outputs\\test_images"
CLASS      = "GRF"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with h5py.File(FILE_PATH, "r") as f:
    images = f[f"{CLASS}/{CLASS}"]
    mech   = f[f"{CLASS}/homogenized_mechanical"]

    # pick 5 random indices
    indices = np.random.randint(0, len(images), 5)

    for idx in indices:
        img   = images[idx]          # shape (256, 256), values 0 or 1
        E_x   = mech[idx, 0, 0]     # true property value

        # scale to 0-255 for saving as PNG
        img_uint8 = (img * 255).astype(np.uint8)

        filename = f"{OUTPUT_DIR}\\GRF_idx{idx}_Ex{E_x:.0f}.png"
        cv2.imwrite(filename, img_uint8)
        print(f"Saved: {filename}  (true E_x = {E_x:.2f})")

print("\nuse any of these images with predict.py")